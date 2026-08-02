//! Offline peak-lag estimation for the turntable-plus-strobe rig.
//!
//! spec.md §6 L0 states the method in one sentence: *"Cross-correlate gyro
//! angular rate against image-derived rotation rate; the peak lag **is** the
//! offset."* This module is that sentence, with three details the sentence
//! leaves out.
//!
//! **The evaluation window must not move with the lag.** Normalised correlation
//! is only comparable across lags if every lag is scored on the same samples;
//! otherwise the correlation at large lags is computed on a shorter, differently
//! conditioned window and the argmax reflects windowing rather than alignment.
//! So the window is the intersection that is valid for *every* candidate lag,
//! and if that intersection is empty we say so instead of quietly shrinking the
//! search.
//!
//! **The peak must be interpolated.** The lag grid is `step`, but the underlying
//! correlation is a smooth function of a continuous lag, so the sampled peak is
//! almost never at the true one. A parabola through the peak and its two
//! neighbours recovers the sub-step position, and for a signal whose correlation
//! width is many steps that recovery is good to a small fraction of a step.
//!
//! **The estimate needs a variance, not just a number.** spec.md §6 L0 asks for
//! the *variance* of the residual offset explicitly, "because the jitter is the
//! thing we claim to fix". See [`LagEstimate::variance`] for what is computed
//! and what it assumes.

use crate::is_positive_finite;

/// Result of a lag search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LagEstimate {
    /// How far `b` lags `a`, in seconds. Subtract it from `b`'s timestamps to
    /// align the two streams. With `a` = gyro rate and `b` = image-derived
    /// rate this is exactly `td` in the crate's sign convention: positive means
    /// camera stamps lag IMU stamps.
    pub lag_seconds: f64,
    /// Pearson correlation at the interpolated peak, in `[-1, 1]`. A rig run
    /// that produces a peak below ~0.9 is telling you the two signals are not
    /// measuring the same rotation, and the lag is meaningless.
    pub correlation: f64,
    /// Variance of `lag_seconds`, in seconds squared.
    ///
    /// The Gauss-Newton covariance of the two-parameter fit
    /// `a(t) ≈ alpha * b(t + lag)`: `sigma_residual^2 / Σ (alpha * db/dt)^2`.
    /// The denominator is the classic "slope energy" of delay estimation — a
    /// flat signal pins the lag down not at all, a fast-changing one pins it
    /// down well.
    ///
    /// **Assumes the residuals are independent on the resampling grid.** If
    /// `step` is much finer than the native sample spacing they are not, and
    /// this is optimistic by roughly the oversampling factor. Choose `step` near
    /// the faster stream's sample interval.
    pub variance: f64,
}

/// Cross-correlate two irregularly sampled `(time, value)` rate signals and
/// return the lag at the correlation peak.
///
/// Both slices must be sorted by time and hold at least two finite points.
/// Candidate lags run over `[-max_lag, max_lag]` in increments of `step`; both
/// signals are linearly resampled onto a common grid of spacing `step`.
///
/// Returns `None` when the search cannot produce a meaningful answer:
///
/// - either input is too short, unsorted, or non-finite;
/// - `max_lag` or `step` is not positive, or `max_lag < 2 * step`;
/// - the overlap valid for every candidate lag is shorter than a handful of
///   steps;
/// - either signal is constant over the window, so correlation is undefined;
/// - **the peak sits at the edge of the search range**, which means the true lag
///   is outside it and the "peak" is an artifact of where we stopped looking.
///   Widening `max_lag` is the fix, and silently returning the boundary would
///   hide the need for it.
#[must_use]
pub fn cross_correlate_lag(
    a: &[(f64, f64)],
    b: &[(f64, f64)],
    max_lag: f64,
    step: f64,
) -> Option<LagEstimate> {
    if !is_positive_finite(step) || !is_positive_finite(max_lag) {
        return None;
    }
    if !is_usable(a) || !is_usable(b) {
        return None;
    }
    let lag_steps = (max_lag / step).floor() as i64;
    if lag_steps < 2 {
        return None;
    }

    // Window valid for every candidate lag: t in a's span, and t + lag in b's
    // span for all |lag| <= lag_steps * step.
    let reach = lag_steps as f64 * step;
    let t0 = a[0].0.max(b[0].0 + reach);
    let t1 = a[a.len() - 1].0.min(b[b.len() - 1].0 - reach);
    let n = ((t1 - t0) / step).floor() as i64 + 1;
    // Eight is arbitrary but the shape is not: below a handful of samples the
    // correlation is dominated by whichever two points happen to line up.
    if n < 8 {
        return None;
    }
    let n = n as usize;

    // Resample `a` once and centre it, so the per-lag inner loop only touches
    // `b`.
    let mut sampler_a = Resampler::new(a);
    let mut av = Vec::with_capacity(n);
    for i in 0..n {
        av.push(sampler_a.at(t0 + i as f64 * step));
    }
    let mean_a = av.iter().sum::<f64>() / n as f64;
    for v in av.iter_mut() {
        *v -= mean_a;
    }
    let var_a = av.iter().map(|v| v * v).sum::<f64>() / n as f64;
    if !is_positive_finite(var_a) {
        return None; // constant reference signal: nothing to align to
    }

    let mut sampler_b = Resampler::new(b);
    let mut best_index = 0i64;
    let mut best_rho = f64::NEG_INFINITY;
    let mut rho = vec![f64::NEG_INFINITY; (2 * lag_steps + 1) as usize];
    for j in -lag_steps..=lag_steps {
        let lag = j as f64 * step;
        let Some(r) = pearson_at(&av, var_a, &mut sampler_b, t0, step, n, lag) else {
            continue;
        };
        rho[(j + lag_steps) as usize] = r;
        if r > best_rho {
            best_rho = r;
            best_index = j;
        }
    }
    if !best_rho.is_finite() {
        return None;
    }
    // A peak at the boundary is not a peak.
    if best_index <= -lag_steps || best_index >= lag_steps {
        log::warn!("correlate: peak at the edge of +/-{max_lag}s — widen the search range");
        return None;
    }

    let centre = (best_index + lag_steps) as usize;
    let (ym, y0, yp) = (rho[centre - 1], rho[centre], rho[centre + 1]);
    if !ym.is_finite() || !yp.is_finite() {
        return None;
    }
    // Parabola through three equally spaced samples; the vertex is at
    // delta = (y- - y+) / (2 (y- - 2 y0 + y+)) steps from the centre.
    let curvature = ym - 2.0 * y0 + yp;
    let (delta, peak_rho) = if curvature < 0.0 {
        let d = 0.5 * (ym - yp) / curvature;
        // A vertex outside the bracketing samples means the three points are not
        // describing a local maximum; fall back to the grid point.
        if d.abs() <= 0.5 {
            (d, y0 - 0.25 * (ym - yp) * d)
        } else {
            (0.0, y0)
        }
    } else {
        (0.0, y0)
    };
    let lag_seconds = (best_index as f64 + delta) * step;

    let variance = lag_variance(&av, &mut sampler_b, t0, step, n, lag_seconds);

    Some(LagEstimate {
        lag_seconds,
        correlation: peak_rho,
        variance,
    })
}

/// Pearson correlation between the pre-centred `av` and `b` shifted by `lag`.
///
/// `b`'s mean and variance are recomputed per lag because the shifted window
/// covers different data; reusing a single global mean is a common and quietly
/// wrong shortcut.
fn pearson_at(
    av: &[f64],
    var_a: f64,
    sampler_b: &mut Resampler<'_>,
    t0: f64,
    step: f64,
    n: usize,
    lag: f64,
) -> Option<f64> {
    sampler_b.reset();
    let mut sum_b = 0.0;
    let mut sum_bb = 0.0;
    let mut sum_ab = 0.0;
    for (i, &a) in av.iter().enumerate() {
        let bv = sampler_b.at(t0 + i as f64 * step + lag);
        sum_b += bv;
        sum_bb += bv * bv;
        sum_ab += a * bv;
    }
    let inv_n = 1.0 / n as f64;
    let mean_b = sum_b * inv_n;
    let var_b = sum_bb * inv_n - mean_b * mean_b;
    if !is_positive_finite(var_b) {
        return None;
    }
    // `av` is zero-mean, so sum_ab is already the centred cross-product.
    Some(sum_ab * inv_n / (var_a * var_b).sqrt())
}

/// Gauss-Newton covariance of the lag, treating `a(t) = alpha * b(t + lag) + e`.
fn lag_variance(
    av: &[f64],
    sampler_b: &mut Resampler<'_>,
    t0: f64,
    step: f64,
    n: usize,
    lag: f64,
) -> f64 {
    sampler_b.reset();
    let mut bv = Vec::with_capacity(n);
    for i in 0..n {
        bv.push(sampler_b.at(t0 + i as f64 * step + lag));
    }
    let mean_b = bv.iter().sum::<f64>() / n as f64;
    for v in bv.iter_mut() {
        *v -= mean_b;
    }

    let sbb = bv.iter().map(|v| v * v).sum::<f64>();
    if !is_positive_finite(sbb) {
        return f64::INFINITY;
    }
    // Amplitude is a nuisance parameter: gyro rate is rad/s and the
    // image-derived rate is only rad/s if L2's focal length is right.
    let alpha = av.iter().zip(&bv).map(|(a, b)| a * b).sum::<f64>() / sbb;

    // Interior points only: db/dt is a central difference, so the endpoints have
    // no derivative and must be excluded from both sums to keep them consistent.
    if n < 5 {
        return f64::INFINITY;
    }
    let mut residual_sq = 0.0;
    let mut slope_energy = 0.0;
    for i in 1..n - 1 {
        let r = av[i] - alpha * bv[i];
        residual_sq += r * r;
        let db = (bv[i + 1] - bv[i - 1]) / (2.0 * step);
        slope_energy += (alpha * db) * (alpha * db);
    }
    if !is_positive_finite(slope_energy) {
        return f64::INFINITY;
    }
    // Two fitted parameters: alpha and lag.
    let dof = (n - 2).saturating_sub(2);
    if dof == 0 {
        return f64::INFINITY;
    }
    let sigma2 = residual_sq / dof as f64;
    sigma2 / slope_energy
}

/// Whether a signal can be correlated at all.
fn is_usable(s: &[(f64, f64)]) -> bool {
    if s.len() < 2 {
        return false;
    }
    s.windows(2).all(|w| w[0].0 < w[1].0) && s.iter().all(|&(t, v)| t.is_finite() && v.is_finite())
}

/// Linear interpolation over a sorted `(time, value)` series with a forward
/// cursor.
///
/// The cursor turns the inner loop from `O(n log m)` binary searches into
/// `O(n + m)`, which matters because the loop runs once per candidate lag.
/// Callers must feed non-decreasing times between [`Resampler::reset`] calls.
struct Resampler<'a> {
    points: &'a [(f64, f64)],
    cursor: usize,
}

impl<'a> Resampler<'a> {
    fn new(points: &'a [(f64, f64)]) -> Self {
        Resampler { points, cursor: 0 }
    }

    fn reset(&mut self) {
        self.cursor = 0;
    }

    fn at(&mut self, t: f64) -> f64 {
        while self.cursor + 2 < self.points.len() && self.points[self.cursor + 1].0 < t {
            self.cursor += 1;
        }
        let (t0, v0) = self.points[self.cursor];
        let (t1, v1) = self.points[self.cursor + 1];
        let dt = t1 - t0;
        if dt <= 0.0 {
            return v0;
        }
        // Clamped: the caller guarantees `t` is inside the series, and clamping
        // is a saner failure than extrapolating a rate signal.
        let alpha = ((t - t0) / dt).clamp(0.0, 1.0);
        v0 + (v1 - v0) * alpha
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wslam_core::DeterministicRng;

    /// A rotation-rate profile with enough bandwidth to localise a peak.
    ///
    /// A turntable at a *constant* rate is useless for cross-correlation — every
    /// lag correlates perfectly — which is why the rig programme has to include
    /// rate changes. This is the synthetic stand-in for that programme.
    fn rate(t: f64) -> f64 {
        0.8 * (2.0 * std::f64::consts::PI * 0.7 * t).sin()
            + 0.4 * (2.0 * std::f64::consts::PI * 2.3 * t + 1.1).sin()
            + 0.2 * (2.0 * std::f64::consts::PI * 5.1 * t + 0.3).sin()
    }

    /// `a` is stamped correctly; `b` is stamped `lag` seconds late, so the sample
    /// physically taken at `u` is recorded at `u + lag` and therefore
    /// `b(t) = s(t - lag)`.
    /// A `(time, value)` rate signal, the shape both the rig and this module
    /// speak in.
    type RateSignal = Vec<(f64, f64)>;

    fn signals(lag: f64, hz_a: f64, hz_b: f64, duration: f64) -> (RateSignal, RateSignal) {
        let na = (duration * hz_a) as usize;
        let nb = (duration * hz_b) as usize;
        let a = (0..na)
            .map(|i| (i as f64 / hz_a, rate(i as f64 / hz_a)))
            .collect();
        let b = (0..nb)
            .map(|i| {
                let t = i as f64 / hz_b;
                (t, rate(t - lag))
            })
            .collect();
        (a, b)
    }

    #[test]
    fn recovers_a_known_lag_to_sub_step_accuracy() {
        let step = 1.0 / 240.0;
        // Deliberately off-grid: 0.37 of a step past a grid point, so a
        // grid-only search could not possibly get this right.
        let truth = 12.0 * step + 0.37 * step;
        let (a, b) = signals(truth, 200.0, 30.0, 20.0);
        let est = cross_correlate_lag(&a, &b, 0.25, step).expect("peak");
        let err = (est.lag_seconds - truth).abs();
        assert!(
            err < 0.2 * step,
            "lag {} vs truth {truth}, error {err:.3e}s = {:.2} steps",
            est.lag_seconds,
            err / step
        );
        assert!(est.correlation > 0.99, "correlation {}", est.correlation);
        assert!(est.variance >= 0.0 && est.variance.is_finite());
    }

    #[test]
    fn sub_step_refinement_beats_the_grid_across_the_whole_step() {
        // Sweep the true lag through a full grid cell. Without parabolic
        // interpolation the error would be up to half a step by construction.
        let step = 1.0 / 200.0;
        let mut worst = 0.0f64;
        for i in 0..9 {
            let truth = 0.030 + (i as f64 / 8.0 - 0.5) * step;
            let (a, b) = signals(truth, 200.0, 60.0, 20.0);
            let est = cross_correlate_lag(&a, &b, 0.20, step).expect("peak");
            worst = worst.max((est.lag_seconds - truth).abs());
        }
        assert!(
            worst < 0.25 * step,
            "worst sub-step error {worst:.3e}s = {:.2} steps",
            worst / step
        );
    }

    #[test]
    fn recovers_zero_lag() {
        let step = 1.0 / 200.0;
        let (a, b) = signals(0.0, 200.0, 200.0, 20.0);
        let est = cross_correlate_lag(&a, &b, 0.2, step).expect("peak");
        assert!(est.lag_seconds.abs() < 0.2 * step, "{}", est.lag_seconds);
    }

    #[test]
    fn recovers_a_negative_lag() {
        // `b` stamped *early* relative to `a`. The sign has to survive.
        let step = 1.0 / 200.0;
        let truth = -0.0234;
        let (a, b) = signals(truth, 200.0, 60.0, 20.0);
        let est = cross_correlate_lag(&a, &b, 0.15, step).expect("peak");
        assert!(
            (est.lag_seconds - truth).abs() < 0.3 * step,
            "{} vs {truth}",
            est.lag_seconds
        );
    }

    #[test]
    fn is_invariant_to_dc_offset_and_amplitude_scale() {
        // A gyro bias and an unknown rad-per-pixel scaling must not move the
        // peak; that is the entire reason for using normalised correlation with
        // per-lag re-centring.
        let step = 1.0 / 200.0;
        let truth = 0.0217;
        let (a, b) = signals(truth, 200.0, 60.0, 20.0);
        let plain = cross_correlate_lag(&a, &b, 0.15, step).expect("peak");

        let a2: Vec<(f64, f64)> = a.iter().map(|&(t, v)| (t, v + 0.05)).collect();
        let b2: Vec<(f64, f64)> = b.iter().map(|&(t, v)| (t, 3.7 * v - 2.0)).collect();
        let scaled = cross_correlate_lag(&a2, &b2, 0.15, step).expect("peak");

        assert!(
            (scaled.lag_seconds - plain.lag_seconds).abs() < 1e-9,
            "{} vs {}",
            scaled.lag_seconds,
            plain.lag_seconds
        );
    }

    #[test]
    fn variance_grows_with_measurement_noise() {
        let step = 1.0 / 200.0;
        let truth = 0.0217;
        let mut previous = 0.0;
        for &noise in &[0.0, 0.01, 0.05, 0.2] {
            let mut rng = DeterministicRng::new("correlate-noise", 55);
            let (a, b) = signals(truth, 200.0, 60.0, 20.0);
            let b: Vec<(f64, f64)> = b
                .iter()
                .map(|&(t, v)| (t, v + rng.normal_with(0.0, noise)))
                .collect();
            let est = cross_correlate_lag(&a, &b, 0.15, step).expect("peak");
            assert!(
                est.variance >= previous,
                "variance must not shrink as noise grows: {:.3e} < {previous:.3e}",
                est.variance
            );
            previous = est.variance;
            // And the reported sigma must actually bracket the error.
            let err = (est.lag_seconds - truth).abs();
            assert!(
                err < 3.0 * est.variance.sqrt() + 0.3 * step,
                "error {err:.3e} outside 3 sigma {:.3e} at noise {noise}",
                est.variance.sqrt()
            );
        }
    }

    #[test]
    fn variance_shrinks_as_the_record_lengthens() {
        let step = 1.0 / 200.0;
        let mut previous = f64::INFINITY;
        for &duration in &[5.0, 10.0, 20.0, 40.0] {
            let mut rng = DeterministicRng::new("correlate-len", 8);
            let (a, b) = signals(0.02, 200.0, 60.0, duration);
            let b: Vec<(f64, f64)> = b
                .iter()
                .map(|&(t, v)| (t, v + rng.normal_with(0.0, 0.05)))
                .collect();
            let est = cross_correlate_lag(&a, &b, 0.15, step).expect("peak");
            assert!(
                est.variance < previous,
                "variance must shrink with more data: {:.3e} !< {previous:.3e}",
                est.variance
            );
            previous = est.variance;
        }
    }

    #[test]
    fn refuses_when_the_peak_is_at_the_search_boundary() {
        // True lag 80 ms, search only +/-20 ms. The best in-range lag is at the
        // edge, and reporting it would be a confident wrong answer.
        let step = 1.0 / 200.0;
        let (a, b) = signals(0.080, 200.0, 60.0, 20.0);
        assert!(cross_correlate_lag(&a, &b, 0.020, step).is_none());
        // Widening the search fixes it, which is the message the None carries.
        let est = cross_correlate_lag(&a, &b, 0.200, step).expect("peak");
        assert!((est.lag_seconds - 0.080).abs() < 0.3 * step);
    }

    #[test]
    fn refuses_a_constant_signal() {
        // The static-hold degenerate case: no rotation, so no peak exists.
        let a: Vec<(f64, f64)> = (0..2000).map(|i| (i as f64 / 200.0, 0.0)).collect();
        let (_, b) = signals(0.02, 200.0, 60.0, 10.0);
        assert!(cross_correlate_lag(&a, &b, 0.1, 1.0 / 200.0).is_none());
        assert!(cross_correlate_lag(&b, &a, 0.1, 1.0 / 200.0).is_none());
    }

    #[test]
    fn refuses_degenerate_arguments() {
        let (a, b) = signals(0.02, 200.0, 60.0, 10.0);
        assert!(cross_correlate_lag(&[], &b, 0.1, 0.005).is_none());
        assert!(cross_correlate_lag(&a, &[(0.0, 1.0)], 0.1, 0.005).is_none());
        assert!(cross_correlate_lag(&a, &b, 0.1, 0.0).is_none());
        assert!(cross_correlate_lag(&a, &b, 0.1, -0.005).is_none());
        assert!(cross_correlate_lag(&a, &b, 0.0, 0.005).is_none());
        assert!(cross_correlate_lag(&a, &b, f64::NAN, 0.005).is_none());
        // max_lag under two steps leaves no room to bracket a peak.
        assert!(cross_correlate_lag(&a, &b, 0.009, 0.005).is_none());
        // Unsorted input.
        let mut unsorted = a.clone();
        unsorted.swap(3, 9);
        assert!(cross_correlate_lag(&unsorted, &b, 0.1, 0.005).is_none());
        // Non-finite sample.
        let mut nasty = a.clone();
        nasty[10].1 = f64::NAN;
        assert!(cross_correlate_lag(&nasty, &b, 0.1, 0.005).is_none());
    }

    #[test]
    fn refuses_when_the_records_barely_overlap() {
        let step = 1.0 / 200.0;
        let a: Vec<(f64, f64)> = (0..400)
            .map(|i| (i as f64 * step, rate(i as f64 * step)))
            .collect();
        // `b` starts where `a` ends: after reserving +/-max_lag there is nothing
        // left to correlate.
        let b: Vec<(f64, f64)> = (0..400)
            .map(|i| {
                let t = 2.0 + i as f64 * step;
                (t, rate(t))
            })
            .collect();
        assert!(cross_correlate_lag(&a, &b, 0.2, step).is_none());
    }

    #[test]
    fn handles_streams_at_very_different_rates() {
        // The real case: 100 Hz gyro against 30 Hz vision.
        let step = 1.0 / 100.0;
        let truth = 0.0263;
        let (a, b) = signals(truth, 100.0, 30.0, 30.0);
        let est = cross_correlate_lag(&a, &b, 0.15, step).expect("peak");
        assert!(
            (est.lag_seconds - truth).abs() < 0.3 * step,
            "{} vs {truth}",
            est.lag_seconds
        );
    }

    #[test]
    fn resampler_interpolates_linearly_and_clamps_at_the_ends() {
        let pts = [(0.0, 0.0), (1.0, 10.0), (2.0, 20.0)];
        let mut r = Resampler::new(&pts);
        assert!((r.at(0.0) - 0.0).abs() < 1e-15);
        assert!((r.at(0.25) - 2.5).abs() < 1e-15);
        assert!((r.at(1.5) - 15.0).abs() < 1e-15);
        assert!((r.at(2.0) - 20.0).abs() < 1e-15);
        r.reset();
        assert!((r.at(-5.0) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn is_deterministic() {
        let step = 1.0 / 200.0;
        let (a, b) = signals(0.02, 200.0, 60.0, 20.0);
        let x = cross_correlate_lag(&a, &b, 0.15, step).unwrap();
        let y = cross_correlate_lag(&a, &b, 0.15, step).unwrap();
        assert_eq!(x.lag_seconds.to_bits(), y.lag_seconds.to_bits());
        assert_eq!(x.variance.to_bits(), y.variance.to_bits());
    }
}
