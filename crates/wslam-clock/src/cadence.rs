//! Robust linear cadence fitting over event **index**.
//!
//! spec.md §4 L0: `DeviceMotion` events "are generated on a regular hardware
//! cadence and only *delivered* with event-loop jitter". Written out, the
//! delivery stamp of event `k` is
//!
//! ```text
//!     observed_k = slope * k + intercept + delay_k
//! ```
//!
//! where `slope` is the hardware sample period, and `delay_k` is non-negative
//! and heavy-tailed — a main-thread stall delays delivery, and nothing delivers
//! an event early. Two consequences drive every choice in this file:
//!
//! - **The regressor is the index, which is exact.** Differencing consecutive
//!   stamps to get a period differences the noise too, and a single stall
//!   corrupts two samples. Regressing on index does not.
//! - **The noise is one-sided and heavy-tailed, so the fit must be robust.**
//!   Least squares gives a 200 ms stall the same standing as a 2 ms one.
//!   Huber IRLS caps its influence at `huber_k` robust sigmas.
//!
//! The one-sidedness has a consequence worth stating plainly: the mean delivery
//! delay is absorbed into `intercept`, not into `slope`. The cadence is
//! recoverable to microseconds while the *epoch* stays biased by however long
//! the platform typically sits on an event. Removing that bias is not this
//! model's job — it is a constant, and constants are what [`crate::OffsetFilter`]
//! and the rig are for.

use std::collections::VecDeque;

use crate::is_positive_finite;

/// How many times the robust scale is re-estimated per fit.
///
/// The scale is held **fixed** while the line is solved for, and only then
/// re-derived: re-deriving it from the weights it is about to produce is the
/// classic way to make IRLS oscillate. Two rounds is enough because the first
/// one starts from the previous push's solution, which is already almost right;
/// the second exists for the cold start, where the seed comes from unweighted
/// least squares and a stall in the window inflates the initial scale.
///
/// It is also the cost knob. The scale estimate is a median-of-medians and
/// dominates the per-push work, so this bounds the hot path at three sorts of
/// the window rather than one per location iteration.
const MAX_SCALE_ROUNDS: usize = 2;

/// Location-solve iterations per scale round.
const MAX_LOCATION_ITERATIONS: usize = 4;

/// IRLS convergence tolerance on `|Δslope| + |Δintercept|`, in seconds.
/// A picosecond is nine orders below any browser stamp's resolution.
const IRLS_TOLERANCE: f64 = 1e-12;

/// Floor on the robust scale used as the Huber divisor, in seconds.
///
/// Without it, a window whose residuals happen to have zero MAD (a majority
/// exactly on the line, which synthetic data produces routinely) would divide by
/// zero and drive every off-line sample's weight to zero. One nanosecond is
/// below the resolution of every browser timestamp we will ever see, so treating
/// scatter under it as exact costs nothing.
const SIGMA_FLOOR: f64 = 1e-9;

/// Consistency constant making the median absolute deviation an unbiased
/// estimator of sigma for Gaussian data. Same value as
/// [`wslam_core::stats::mad_sigma`], which this file is a buffer-reusing
/// specialisation of; `robust_sigma_agrees_with_core_mad_sigma` holds them
/// together.
const MAD_TO_SIGMA: f64 = 1.4826;

/// Tuning for [`CadenceModel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CadenceConfig {
    /// Samples required before the model will predict at all.
    pub min_samples: usize,
    /// Sliding window length in samples. Bounds both cost and how fast the model
    /// can follow a genuine cadence change (a `DeviceMotion` frequency switch, or
    /// a camera dropping from 60 to 30 fps under thermal pressure).
    pub window: usize,
    /// Huber tuning constant, in units of robust sigma. Residuals beyond it get
    /// linear rather than quadratic loss.
    pub huber_k: f64,
}

impl Default for CadenceConfig {
    fn default() -> Self {
        CadenceConfig {
            min_samples: 16,
            window: 256,
            // 1.345 is the classic Huber constant: 95% of the efficiency of
            // least squares on clean Gaussian data, with bounded influence.
            huber_k: 1.345,
        }
    }
}

impl CadenceConfig {
    /// Size the window for a nominal rate and a wall-clock span.
    ///
    /// `min_samples` is set to a quarter of the window so the model starts
    /// predicting after ~1/4 of the span rather than waiting for a full one.
    #[must_use]
    pub fn for_rate(nominal_hz: f64, window_seconds: f64) -> Self {
        let window = ((nominal_hz * window_seconds).round() as usize).max(8);
        CadenceConfig {
            min_samples: (window / 4).max(4),
            window,
            ..Self::default()
        }
    }
}

/// A fitted line in the recentred frame: `y = slope * x + intercept`, with
/// `x = index - index_origin` and `y = seconds - time_origin`.
#[derive(Debug, Clone, Copy)]
struct Line {
    slope: f64,
    intercept: f64,
}

impl Line {
    #[inline]
    fn at(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }
}

/// Leverage terms carried alongside the fit so prediction uncertainty can be
/// reported without re-walking the window.
#[derive(Debug, Clone, Copy)]
struct Leverage {
    /// Sum of Huber weights. Bounded above by the sample count because weights
    /// are at most 1, so using it in place of `n` errs conservative — the right
    /// direction for a number we publish.
    sum_w: f64,
    /// Weighted mean of `x`.
    mean_x: f64,
    /// `Σ w (x - mean_x)^2`.
    sxx: f64,
}

/// Robust incremental fit of `t = slope * index + intercept`.
///
/// Push samples as they arrive; the fit is refreshed on every push, warm-started
/// from the previous solution so the usual cost is a couple of passes over the
/// window rather than a full re-solve.
#[derive(Debug, Clone)]
pub struct CadenceModel {
    config: CadenceConfig,
    /// `(x, y)` in the recentred frame, oldest first.
    samples: VecDeque<(f64, f64)>,
    index_origin: u64,
    time_origin: f64,
    line: Option<Line>,
    leverage: Option<Leverage>,
    /// Robust sigma of the fit residuals, seconds. This is the delivery jitter
    /// we exist to measure.
    residual_sigma: f64,
    /// Scratch for the median-of-absolute-deviations, kept across pushes so the
    /// hot path does not allocate at IMU rate.
    scratch: Vec<f64>,
    /// Per-sample Huber weights for the current iterate, same order as
    /// `samples`.
    weights: Vec<f64>,
}

impl CadenceModel {
    /// Construct an empty model.
    ///
    /// A self-contradictory config is clamped rather than rejected: the
    /// contracted constructor returns `Self`, and a clock that refuses to start
    /// is worse than one that quietly widens a too-small window.
    #[must_use]
    pub fn new(config: CadenceConfig) -> Self {
        // Three points is the least that leaves a residual to measure scatter
        // from; two would fit exactly and report zero jitter forever.
        let min_samples = config.min_samples.max(3);
        let window = config.window.max(min_samples);
        let huber_k = if config.huber_k.is_finite() && config.huber_k > 0.0 {
            config.huber_k
        } else {
            CadenceConfig::default().huber_k
        };
        CadenceModel {
            config: CadenceConfig {
                min_samples,
                window,
                huber_k,
            },
            samples: VecDeque::with_capacity(window),
            index_origin: 0,
            time_origin: 0.0,
            line: None,
            leverage: None,
            residual_sigma: f64::INFINITY,
            scratch: Vec::with_capacity(window),
            weights: Vec::with_capacity(window),
        }
    }

    /// Add one observation: event `index` was stamped at `observed_seconds`.
    ///
    /// Non-finite stamps are dropped. The shim is required to pass raw values
    /// through untouched (spec.md §7), so a NaN here means the platform produced
    /// one, and admitting it would poison the fit permanently.
    pub fn push(&mut self, index: u64, observed_seconds: f64) {
        if !observed_seconds.is_finite() {
            log::warn!("cadence: dropping non-finite stamp at index {index}");
            return;
        }
        if self.samples.is_empty() {
            self.index_origin = index;
            self.time_origin = observed_seconds;
        }
        // f64 subtraction rather than u64: an out-of-order or reset index must
        // produce a negative regressor, not an underflow to 1.8e19.
        let x = index as f64 - self.index_origin as f64;
        let y = observed_seconds - self.time_origin;

        if self.samples.len() == self.config.window {
            self.samples.pop_front();
        }
        self.samples.push_back((x, y));
        self.refit();
    }

    /// Predicted stamp for `index`, in the same units `push` was given.
    ///
    /// `None` until [`CadenceModel::is_converged`]. Extrapolates freely outside
    /// the window — [`CadenceModel::prediction_variance`] is how far you should
    /// trust it.
    #[must_use]
    pub fn predict(&self, index: u64) -> Option<f64> {
        let line = self.line?;
        let x = index as f64 - self.index_origin as f64;
        Some(self.time_origin + line.at(x))
    }

    /// Seconds per event — the recovered hardware cadence.
    #[must_use]
    pub fn slope(&self) -> Option<f64> {
        self.line.map(|l| l.slope)
    }

    /// Fitted stamp at the origin index, in the units `push` was given.
    #[must_use]
    pub fn intercept(&self) -> Option<f64> {
        self.line.map(|l| self.time_origin + l.intercept)
    }

    /// Robust variance of the fit residuals, in seconds squared.
    ///
    /// **This is the number spec.md §6 L0 asks us to report**: the distribution
    /// of delivery jitter, which is the thing we claim to fix. It is a *robust*
    /// variance (scaled MAD, squared) on purpose — a single 200 ms stall in a
    /// four-second window moves the ordinary sample variance by three orders of
    /// magnitude and turns the reported figure into a description of the worst
    /// event rather than of typical delivery. Use
    /// [`CadenceModel::residual_percentile`] for the tail; that is what it is
    /// for.
    ///
    /// Infinite before convergence.
    #[must_use]
    pub fn residual_variance(&self) -> f64 {
        self.residual_sigma * self.residual_sigma
    }

    /// Quantile of the **signed** residuals in the current window, seconds.
    ///
    /// "Report the distribution, not a mean" (spec.md §6 L0) needs more than a
    /// scale parameter when the noise is one-sided and heavy-tailed: the p50 and
    /// the p99 of delivery delay are different facts about the platform.
    #[must_use]
    pub fn residual_percentile(&self, q: f64) -> Option<f64> {
        let line = self.line?;
        let residuals: Vec<f64> = self.samples.iter().map(|&(x, y)| y - line.at(x)).collect();
        wslam_core::stats::percentile(&residuals, q)
    }

    /// Variance of the fitted line's prediction at `index`, seconds squared.
    ///
    /// The weighted-least-squares prediction variance
    /// `sigma^2 (1/Σw + (x - x̄)^2 / Σw(x - x̄)^2)`. Grows quadratically as you
    /// extrapolate away from the window, which is exactly the honest behaviour:
    /// a cadence model asked about an index it has never seen near should say so.
    ///
    /// Infinite before convergence.
    #[must_use]
    pub fn prediction_variance(&self, index: u64) -> f64 {
        let Some(lev) = self.leverage else {
            return f64::INFINITY;
        };
        if lev.sum_w <= 0.0 || lev.sxx <= 0.0 {
            return f64::INFINITY;
        }
        let x = index as f64 - self.index_origin as f64;
        let d = x - lev.mean_x;
        self.residual_variance() * (1.0 / lev.sum_w + d * d / lev.sxx)
    }

    /// Samples currently in the window.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Whether the model has enough well-conditioned data to predict.
    #[must_use]
    pub fn is_converged(&self) -> bool {
        self.line.is_some()
    }

    /// Forget everything. Called on stream restart so a stale cadence cannot
    /// stamp a fresh session.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.line = None;
        self.leverage = None;
        self.residual_sigma = f64::INFINITY;
        self.index_origin = 0;
        self.time_origin = 0.0;
    }

    /// Configuration in force, after the constructor's clamping.
    #[must_use]
    pub fn config(&self) -> CadenceConfig {
        self.config
    }

    // -- internals ----------------------------------------------------------

    fn refit(&mut self) {
        if self.samples.len() < self.config.min_samples {
            self.invalidate();
            return;
        }

        // Warm start: the window moved by one sample, so the previous solution
        // is almost the answer. Falling back to ordinary least squares only
        // happens on the very first fit or after a reset.
        let Some(mut line) = self.line.or_else(|| ordinary_least_squares(&self.samples)) else {
            self.invalidate();
            return;
        };

        let k = self.config.huber_k;
        let mut leverage = None;
        'outer: for _ in 0..MAX_SCALE_ROUNDS {
            let sigma = robust_sigma(&self.samples, &line, &mut self.scratch).max(SIGMA_FLOOR);
            for _ in 0..MAX_LOCATION_ITERATIONS {
                self.weights.clear();
                self.weights.extend(
                    self.samples
                        .iter()
                        .map(|&(x, y)| huber_weight(y - line.at(x), sigma, k)),
                );
                let Some((next, lev)) = weighted_fit(&self.samples, &self.weights) else {
                    self.invalidate();
                    return;
                };
                let moved =
                    (next.slope - line.slope).abs() + (next.intercept - line.intercept).abs();
                line = next;
                leverage = Some(lev);
                if moved < IRLS_TOLERANCE {
                    // Converged at this scale. A second scale round would only
                    // move things if the scale itself were still wrong, which
                    // the warm start makes unlikely after the first.
                    continue 'outer;
                }
            }
        }

        if !line.slope.is_finite() || !line.intercept.is_finite() {
            self.invalidate();
            return;
        }

        self.residual_sigma = robust_sigma(&self.samples, &line, &mut self.scratch);
        self.line = Some(line);
        self.leverage = leverage;
    }

    fn invalidate(&mut self) {
        self.line = None;
        self.leverage = None;
        self.residual_sigma = f64::INFINITY;
    }
}

/// Huber weight for a residual: 1 inside `k` sigmas, `k*sigma/|r|` outside, so
/// the effective loss is quadratic in the core and linear in the tail.
#[inline]
fn huber_weight(residual: f64, sigma: f64, k: f64) -> f64 {
    let a = residual.abs() / sigma;
    if a <= k {
        1.0
    } else {
        k / a
    }
}

/// Scaled median absolute deviation of the residuals about `line`.
///
/// A buffer-reusing specialisation of [`wslam_core::stats::mad_sigma`]: this
/// runs on every pushed event at IMU rate, and the core version allocates two
/// vectors per call.
fn robust_sigma(samples: &VecDeque<(f64, f64)>, line: &Line, scratch: &mut Vec<f64>) -> f64 {
    if samples.is_empty() {
        return f64::INFINITY;
    }
    scratch.clear();
    scratch.extend(samples.iter().map(|&(x, y)| y - line.at(x)));
    let median = median_in_place(scratch);
    for v in scratch.iter_mut() {
        *v = (*v - median).abs();
    }
    MAD_TO_SIGMA * median_in_place(scratch)
}

/// Median by sorting in place. `values` must be non-empty.
fn median_in_place(values: &mut [f64]) -> f64 {
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    // Matches wslam_core::stats::percentile(_, 0.5): linear interpolation
    // between the two central order statistics for even n.
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

/// Unweighted fit, used only to seed IRLS.
fn ordinary_least_squares(samples: &VecDeque<(f64, f64)>) -> Option<Line> {
    let n = samples.len() as f64;
    if n < 2.0 {
        return None;
    }
    let mean_x = samples.iter().map(|s| s.0).sum::<f64>() / n;
    let mean_y = samples.iter().map(|s| s.1).sum::<f64>() / n;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for &(x, y) in samples {
        let dx = x - mean_x;
        sxx += dx * dx;
        sxy += dx * (y - mean_y);
    }
    if !is_positive_finite(sxx) {
        return None;
    }
    let slope = sxy / sxx;
    Some(Line {
        slope,
        intercept: mean_y - slope * mean_x,
    })
}

/// Weighted least squares in the numerically stable centred form.
///
/// Returns `None` when the design is rank deficient — every sample at one index,
/// or every weight driven to zero. Both are real: the first is a stalled event
/// stream, the second is a window of pure outliers.
fn weighted_fit(samples: &VecDeque<(f64, f64)>, weights: &[f64]) -> Option<(Line, Leverage)> {
    let mut sum_w = 0.0;
    let mut swx = 0.0;
    let mut swy = 0.0;
    for (&(x, y), &w) in samples.iter().zip(weights) {
        sum_w += w;
        swx += w * x;
        swy += w * y;
    }
    if !is_positive_finite(sum_w) {
        return None;
    }
    let mean_x = swx / sum_w;
    let mean_y = swy / sum_w;

    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (&(x, y), &w) in samples.iter().zip(weights) {
        let dx = x - mean_x;
        sxx += w * dx * dx;
        sxy += w * dx * (y - mean_y);
    }
    if !is_positive_finite(sxx) {
        return None;
    }
    let slope = sxy / sxx;
    Some((
        Line {
            slope,
            intercept: mean_y - slope * mean_x,
        },
        Leverage { sum_w, mean_x, sxx },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::IMU_60HZ;
    use wslam_core::DeterministicRng;

    /// The naive estimate a shim would produce: average successive delivery
    /// stamps. Algebraically `(last - first) / (n - 1)`, so it is a two-sample
    /// estimator wearing an n-sample disguise — every intermediate stamp
    /// cancels, and the two that survive carry the full jitter.
    fn raw_successive_difference_slope(stamps: &[f64]) -> f64 {
        let diffs: Vec<f64> = stamps.windows(2).map(|w| w[1] - w[0]).collect();
        diffs.iter().sum::<f64>() / diffs.len() as f64
    }

    fn fill(model: &mut CadenceModel, period: f64, n: u64) {
        for k in 0..n {
            model.push(k, 100.0 + k as f64 * period);
        }
    }

    #[test]
    fn recovers_exact_slope_and_intercept_from_a_noiseless_cadence() {
        let mut m = CadenceModel::new(CadenceConfig::default());
        let period = 1.0 / 60.0;
        fill(&mut m, period, 120);
        assert!(m.is_converged());
        // Closed form: the data lies exactly on a line, so the fit is that line.
        assert!(
            (m.slope().unwrap() - period).abs() < 1e-15,
            "{:?}",
            m.slope()
        );
        assert!((m.intercept().unwrap() - 100.0).abs() < 1e-12);
        // And prediction is exact, in and out of the window.
        for k in [0u64, 7, 119, 5_000] {
            let want = 100.0 + k as f64 * period;
            assert!((m.predict(k).unwrap() - want).abs() < 1e-9, "index {k}");
        }
        assert!(m.residual_variance() < 1e-24);
    }

    #[test]
    fn fitted_slope_beats_the_raw_timestamp_slope_by_a_wide_margin() {
        let mut rng = DeterministicRng::new("cadence-jitter", 0xC10C_C10C);
        let period = 1.0 / 60.0;
        let n = 240u64;

        let mut m = CadenceModel::new(CadenceConfig::default());
        let mut stamps = Vec::new();
        for k in 0..n {
            let t = 42.0 + k as f64 * period + IMU_60HZ.sample(&mut rng);
            stamps.push(t);
            m.push(k, t);
        }

        let fitted_err = (m.slope().unwrap() - period).abs();
        let raw_err = (raw_successive_difference_slope(&stamps) - period).abs();
        assert!(
            fitted_err * 20.0 < raw_err,
            "fitted slope error {fitted_err:.3e}s vs raw {raw_err:.3e}s — \
             expected at least 20x better"
        );
        // In absolute terms: the cadence is recovered to well under a microsecond
        // per sample from stamps scattered by milliseconds.
        assert!(fitted_err < 1e-6, "fitted slope error {fitted_err:.3e}s");
    }

    #[test]
    fn fitted_model_removes_jitter_rather_than_tracking_it() {
        // The claim under test: the *corrected* stamp is closer to truth than
        // the delivered stamp was. Compared as standard deviations because the
        // mean delivery delay is one-sided and lands in the intercept, where the
        // offset filter and the rig deal with it (see module header).
        let mut rng = DeterministicRng::new("cadence-removes-jitter", 7);
        let period = 1.0 / 60.0;
        let n = 300u64;
        let mut m = CadenceModel::new(CadenceConfig::for_rate(60.0, 5.0));

        let mut raw_err = Vec::new();
        let mut fitted_err = Vec::new();
        for k in 0..n {
            let truth = 42.0 + k as f64 * period;
            let observed = truth + IMU_60HZ.sample(&mut rng);
            m.push(k, observed);
            // Skip warm-up: before min_samples there is no model to judge.
            if m.is_converged() {
                raw_err.push(observed - truth);
                fitted_err.push(m.predict(k).unwrap() - truth);
            }
        }
        assert!(fitted_err.len() > 200);

        let injected = stddev(&raw_err);
        let residual = stddev(&fitted_err);
        assert!(
            residual < injected,
            "residual std {residual:.3e}s must be below injected jitter std {injected:.3e}s"
        );
        // Not marginally below: the fit averages ~200 in-window samples.
        assert!(
            residual * 8.0 < injected,
            "residual std {residual:.3e}s vs injected {injected:.3e}s"
        );
    }

    #[test]
    fn residual_variance_measures_the_injected_jitter() {
        // residual_variance() is a *measurement* of the delivery jitter, not of
        // the corrected error. With Gaussian jitter the robust scale is a
        // consistent estimator of sigma, so it must come back near the truth.
        let mut rng = DeterministicRng::new("cadence-report", 4242);
        let sigma = 3.0e-3;
        let mut m = CadenceModel::new(CadenceConfig::for_rate(60.0, 8.0));
        for k in 0..480u64 {
            m.push(k, k as f64 / 60.0 + rng.normal_with(0.0, sigma));
        }
        let reported = m.residual_variance().sqrt();
        assert!(
            (reported - sigma).abs() < 0.15 * sigma,
            "reported sigma {reported:.4e} vs injected {sigma:.4e}"
        );
    }

    #[test]
    fn a_two_hundred_millisecond_stall_does_not_move_the_slope() {
        // The degenerate delivery event spec.md §4 L0 exists to survive: the
        // main thread blocks, one event lands 200 ms late.
        let period = 1.0 / 60.0;
        let n = 240u64;
        let stall_at = 120u64;

        let mut robust = CadenceModel::new(CadenceConfig::default());
        let mut clean = CadenceModel::new(CadenceConfig::default());
        let mut stamps = Vec::new();
        for k in 0..n {
            let truth = k as f64 * period;
            let observed = if k == stall_at { truth + 0.200 } else { truth };
            stamps.push(observed);
            robust.push(k, observed);
            clean.push(k, truth);
        }

        let robust_err = (robust.slope().unwrap() - period).abs();
        let ols_err = (ordinary_least_squares_slope(&stamps) - period).abs();
        assert!(
            robust_err < 1e-9,
            "one stall moved the robust slope by {robust_err:.3e}s per sample"
        );
        // And it *would* have moved least squares, which is the point.
        assert!(
            ols_err > 100.0 * robust_err.max(1e-15),
            "least squares error {ols_err:.3e} should be far worse than robust {robust_err:.3e}"
        );
        // Predictions are unmoved too, out to a full window of extrapolation.
        let want = clean.predict(400).unwrap();
        assert!((robust.predict(400).unwrap() - want).abs() < 1e-6);
    }

    fn ordinary_least_squares_slope(stamps: &[f64]) -> f64 {
        let q: VecDeque<(f64, f64)> = stamps
            .iter()
            .enumerate()
            .map(|(i, &t)| (i as f64, t))
            .collect();
        super::ordinary_least_squares(&q).unwrap().slope
    }

    #[test]
    fn a_burst_of_stalls_still_leaves_the_majority_in_control() {
        // Huber bounds influence but does not have a 50% breakdown point; 20% of
        // the window stalled is the regime we actually need to survive.
        let period = 1.0 / 30.0;
        let mut m = CadenceModel::new(CadenceConfig::default());
        for k in 0..200u64 {
            let truth = k as f64 * period;
            let observed = if k % 5 == 0 { truth + 0.080 } else { truth };
            m.push(k, observed);
        }
        assert!(
            (m.slope().unwrap() - period).abs() < 1e-6,
            "slope {:?}",
            m.slope()
        );
    }

    #[test]
    fn refuses_to_predict_before_min_samples() {
        let cfg = CadenceConfig {
            min_samples: 10,
            window: 64,
            huber_k: 1.345,
        };
        let mut m = CadenceModel::new(cfg);
        for k in 0..9u64 {
            m.push(k, k as f64 * 0.01);
            assert!(!m.is_converged(), "converged at {} samples", k + 1);
            assert!(m.predict(0).is_none());
            assert!(m.slope().is_none());
            assert!(m.residual_variance().is_infinite());
            assert!(m.prediction_variance(0).is_infinite());
        }
        m.push(9, 0.09);
        assert!(m.is_converged());
    }

    #[test]
    fn refuses_to_fit_without_index_spread() {
        // Degenerate design matrix: every event carrying the same index. Must
        // report "no model", not NaN.
        let mut m = CadenceModel::new(CadenceConfig::default());
        for i in 0..64 {
            m.push(7, 1.0 + i as f64 * 1e-3);
        }
        assert!(!m.is_converged());
        assert!(m.predict(7).is_none());
        assert!(m.prediction_variance(7).is_infinite());
    }

    #[test]
    fn drops_non_finite_stamps_instead_of_poisoning_the_fit() {
        let mut m = CadenceModel::new(CadenceConfig::default());
        fill(&mut m, 1.0 / 60.0, 100);
        let before = m.slope().unwrap();
        m.push(100, f64::NAN);
        m.push(101, f64::INFINITY);
        assert_eq!(m.sample_count(), 100);
        assert!((m.slope().unwrap() - before).abs() < 1e-18);
    }

    #[test]
    fn window_evicts_and_follows_a_genuine_cadence_change() {
        // Thermal throttling drops capture from 60 to 30 fps. The window must
        // forget the old cadence within roughly its own length.
        let cfg = CadenceConfig::for_rate(60.0, 2.0); // window = 120
        let mut m = CadenceModel::new(cfg);
        let mut t = 0.0;
        for k in 0..200u64 {
            t += 1.0 / 60.0;
            m.push(k, t);
        }
        assert!((m.slope().unwrap() - 1.0 / 60.0).abs() < 1e-12);
        for k in 200..400u64 {
            t += 1.0 / 30.0;
            m.push(k, t);
        }
        assert_eq!(m.sample_count(), 120);
        assert!(
            (m.slope().unwrap() - 1.0 / 30.0).abs() < 1e-9,
            "slope after change: {:?}",
            m.slope()
        );
    }

    #[test]
    fn prediction_variance_is_smallest_at_the_window_centre() {
        let mut rng = DeterministicRng::new("leverage", 1);
        let mut m = CadenceModel::new(CadenceConfig::for_rate(60.0, 4.0));
        let n = 240u64;
        for k in 0..n {
            m.push(k, k as f64 / 60.0 + rng.normal_with(0.0, 1e-3));
        }
        let centre = m.prediction_variance(n / 2);
        let edge = m.prediction_variance(n - 1);
        let far = m.prediction_variance(n * 4);
        assert!(centre < edge, "{centre:.3e} !< {edge:.3e}");
        assert!(edge < far, "{edge:.3e} !< {far:.3e}");
        // sigma^2/n at the centre, to within the leverage term.
        let expect = m.residual_variance() / n as f64;
        assert!(
            centre < 3.0 * expect && centre > 0.3 * expect,
            "{centre:.3e} vs {expect:.3e}"
        );
    }

    #[test]
    fn prediction_variance_shrinks_as_samples_accumulate() {
        let mut rng = DeterministicRng::new("leverage-n", 2);
        let cfg = CadenceConfig {
            min_samples: 8,
            window: 4096,
            huber_k: 1.345,
        };
        let mut m = CadenceModel::new(cfg);
        let mut prev = f64::INFINITY;
        for k in 0..800u64 {
            m.push(k, k as f64 / 60.0 + rng.normal_with(0.0, 1e-3));
            if k % 200 == 199 {
                let v = m.prediction_variance(k);
                assert!(v < prev, "variance grew at {k}: {v:.3e} !< {prev:.3e}");
                prev = v;
            }
        }
    }

    #[test]
    fn residual_percentile_exposes_the_one_sided_tail() {
        // Delivery delay is non-negative, so the residual distribution is skewed
        // and the p99 is much further from the median than the p1 is. Reporting
        // only a variance would hide that, which is why the percentile accessor
        // exists.
        let mut rng = DeterministicRng::new("tail", 99);
        let mut m = CadenceModel::new(CadenceConfig::for_rate(60.0, 8.0));
        for k in 0..480u64 {
            m.push(k, k as f64 / 60.0 + IMU_60HZ.sample(&mut rng));
        }
        let p50 = m.residual_percentile(0.5).unwrap();
        let p01 = m.residual_percentile(0.01).unwrap();
        let p99 = m.residual_percentile(0.99).unwrap();
        assert!(p01 < p50 && p50 < p99);
        assert!(
            (p99 - p50) > 3.0 * (p50 - p01),
            "expected a right-skewed tail: p01 {p01:.4}, p50 {p50:.4}, p99 {p99:.4}"
        );
    }

    #[test]
    fn robust_sigma_agrees_with_core_mad_sigma() {
        // The buffer-reusing copy must stay pinned to the house implementation.
        let mut rng = DeterministicRng::new("mad", 5);
        let samples: VecDeque<(f64, f64)> = (0..101)
            .map(|i| (i as f64, rng.normal_with(0.0, 0.01)))
            .collect();
        let line = Line {
            slope: 0.0,
            intercept: 0.0,
        };
        let mut scratch = Vec::new();
        let mine = robust_sigma(&samples, &line, &mut scratch);
        let values: Vec<f64> = samples.iter().map(|s| s.1).collect();
        let theirs = wslam_core::stats::mad_sigma(&values).unwrap();
        assert!((mine - theirs).abs() < 1e-15, "{mine} vs {theirs}");
    }

    #[test]
    fn huber_weight_is_unity_inside_the_core_and_decays_outside() {
        let k = 1.345;
        assert_eq!(huber_weight(0.0, 1.0, k), 1.0);
        assert_eq!(huber_weight(1.0, 1.0, k), 1.0);
        assert_eq!(huber_weight(-1.3, 1.0, k), 1.0);
        // At 10 sigma the influence is capped: w * r = k * sigma, a constant.
        let w = huber_weight(10.0, 1.0, k);
        assert!((w * 10.0 - k).abs() < 1e-15);
        assert!(huber_weight(100.0, 1.0, k) < w);
    }

    #[test]
    fn config_is_clamped_not_rejected() {
        let m = CadenceModel::new(CadenceConfig {
            min_samples: 0,
            window: 1,
            huber_k: -3.0,
        });
        let c = m.config();
        assert!(c.min_samples >= 3);
        assert!(c.window >= c.min_samples);
        assert!(c.huber_k > 0.0);
    }

    #[test]
    fn reset_forgets_the_previous_stream() {
        let mut m = CadenceModel::new(CadenceConfig::default());
        fill(&mut m, 1.0 / 60.0, 100);
        m.reset();
        assert_eq!(m.sample_count(), 0);
        assert!(!m.is_converged());
        assert!(m.predict(0).is_none());
        // A completely different cadence and epoch after the reset.
        for k in 0..100u64 {
            m.push(k, 9_000.0 + k as f64 / 30.0);
        }
        assert!((m.slope().unwrap() - 1.0 / 30.0).abs() < 1e-15);
        assert!((m.predict(0).unwrap() - 9_000.0).abs() < 1e-9);
    }

    #[test]
    fn identical_input_gives_identical_output() {
        // spec.md §6: replay must be bit-exact.
        let run = || {
            let mut rng = DeterministicRng::new("determinism", 31337);
            let mut m = CadenceModel::new(CadenceConfig::default());
            for k in 0..300u64 {
                m.push(k, k as f64 / 60.0 + IMU_60HZ.sample(&mut rng));
            }
            (
                m.slope().unwrap(),
                m.residual_variance(),
                m.predict(1000).unwrap(),
            )
        };
        let a = run();
        let b = run();
        assert_eq!(a.0.to_bits(), b.0.to_bits());
        assert_eq!(a.1.to_bits(), b.1.to_bits());
        assert_eq!(a.2.to_bits(), b.2.to_bits());
    }

    fn stddev(xs: &[f64]) -> f64 {
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (xs.len() - 1) as f64).sqrt()
    }

    proptest::proptest! {
        /// Round-trip property: the model is an affine map from index to time,
        /// so on data that lies exactly on such a map it must reproduce it —
        /// for any cadence, any epoch, any starting index, in and out of the
        /// window. This is the invariant every other test in this file assumes.
        #[test]
        fn noiseless_data_round_trips_through_the_fit(
            // 10 Hz to 240 Hz covers everything a browser reports, and then some.
            period in 1.0f64 / 240.0..0.1f64,
            epoch in -1.0e4f64..1.0e4f64,
            start in 0u64..1_000_000u64,
            n in 8usize..64usize,
        ) {
            let cfg = CadenceConfig { min_samples: 4, window: 64, huber_k: 1.345 };
            let mut m = CadenceModel::new(cfg);
            for i in 0..n as u64 {
                m.push(start + i, epoch + i as f64 * period);
            }
            proptest::prop_assert!(m.is_converged());

            let slope = m.slope().unwrap();
            proptest::prop_assert!(
                (slope - period).abs() <= 1.0e-9 * period,
                "slope {slope} vs period {period}"
            );
            // Interpolation, and extrapolation ten windows past the data.
            for probe in [0u64, (n / 2) as u64, (n - 1) as u64, n as u64 * 10] {
                let want = epoch + probe as f64 * period;
                let got = m.predict(start + probe).unwrap();
                proptest::prop_assert!(
                    (got - want).abs() <= 1.0e-9 * (1.0 + want.abs()),
                    "predict({probe}) = {got}, want {want}"
                );
            }
            // Exact data has no scatter to report.
            proptest::prop_assert!(m.residual_variance() <= 1.0e-18);
        }
    }
}
