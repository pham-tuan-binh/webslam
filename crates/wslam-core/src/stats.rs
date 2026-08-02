//! Statistics needed to *validate* the estimator, not to run it.
//!
//! spec.md §6 L6 asks for NEES against chi-squared bounds and empirical
//! coverage at 68/95/99. Those need a chi-squared CDF and its inverse, and this
//! module provides them without dragging in a stats crate that would have to
//! compile to wasm.

use crate::math::Scalar;

/// Streaming mean and variance (Welford). Numerically stable over long runs,
/// which matters for a 15-minute thermal soak (spec.md §6, System level).
#[derive(Debug, Clone, Copy, Default)]
pub struct Welford {
    n: u64,
    mean: Scalar,
    m2: Scalar,
}

impl Welford {
    /// Empty accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a sample.
    pub fn push(&mut self, x: Scalar) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as Scalar;
        self.m2 += delta * (x - self.mean);
    }

    /// Sample count.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.n
    }
    /// Running mean; 0 when empty.
    #[must_use]
    pub fn mean(&self) -> Scalar {
        self.mean
    }
    /// Unbiased sample variance; 0 with fewer than two samples.
    #[must_use]
    pub fn variance(&self) -> Scalar {
        if self.n < 2 {
            0.0
        } else {
            self.m2 / (self.n - 1) as Scalar
        }
    }
    /// Sample standard deviation.
    #[must_use]
    pub fn stddev(&self) -> Scalar {
        self.variance().sqrt()
    }
}

/// Percentile of an unsorted slice, using linear interpolation between order
/// statistics. `q` in `[0, 1]`. Returns `None` for an empty slice.
///
/// The p99 frame time is the number spec.md §6 L4 asks for, not the mean.
#[must_use]
pub fn percentile(values: &[Scalar], q: Scalar) -> Option<Scalar> {
    if values.is_empty() {
        return None;
    }
    let mut v: Vec<Scalar> = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = q.clamp(0.0, 1.0);
    let pos = q * (v.len() - 1) as Scalar;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as Scalar;
    Some(v[lo] * (1.0 - frac) + v[hi] * frac)
}

/// Median.
#[must_use]
pub fn median(values: &[Scalar]) -> Option<Scalar> {
    percentile(values, 0.5)
}

/// Median absolute deviation, scaled to be a consistent estimator of sigma for
/// Gaussian data. The robust scale estimate RANSAC thresholds are derived from.
#[must_use]
pub fn mad_sigma(values: &[Scalar]) -> Option<Scalar> {
    let m = median(values)?;
    let devs: Vec<Scalar> = values.iter().map(|v| (v - m).abs()).collect();
    Some(1.4826 * median(&devs)?)
}

/// Regularised lower incomplete gamma `P(a, x)`.
///
/// Series expansion for `x < a + 1`, continued fraction otherwise — the
/// standard split, because each converges quickly only on its own side.
#[must_use]
pub fn gamma_p(a: Scalar, x: Scalar) -> Scalar {
    if x < 0.0 || a <= 0.0 {
        return Scalar::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // Series.
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..500 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        sum * (-x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        // Continued fraction for Q(a, x) = 1 - P(a, x), modified Lentz.
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..500 {
            let an = -(i as Scalar) * (i as Scalar - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        1.0 - (-x + a * x.ln() - ln_gamma(a)).exp() * h
    }
}

/// Natural log of the gamma function (Lanczos, g = 7, n = 9).
#[must_use]
pub fn ln_gamma(x: Scalar) -> Scalar {
    const C: [Scalar; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_1,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: Gamma(x) Gamma(1-x) = pi / sin(pi x)
        (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + 7.5;
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as Scalar);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Chi-squared CDF with `k` degrees of freedom.
#[must_use]
pub fn chi2_cdf(x: Scalar, k: usize) -> Scalar {
    if x <= 0.0 {
        0.0
    } else {
        gamma_p(k as Scalar * 0.5, x * 0.5)
    }
}

/// Chi-squared inverse CDF (quantile) with `k` degrees of freedom.
///
/// Bisection on the CDF. Slow and completely adequate: this runs once per
/// report, not once per frame.
#[must_use]
pub fn chi2_quantile(p: Scalar, k: usize) -> Scalar {
    if !(0.0..1.0).contains(&p) {
        return Scalar::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    let (mut lo, mut hi) = (0.0, (k as Scalar).max(1.0));
    while chi2_cdf(hi, k) < p {
        hi *= 2.0;
        if hi > 1e12 {
            return hi;
        }
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if chi2_cdf(mid, k) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Two-sided acceptance interval for the *average* NEES over `n` independent
/// trials with state dimension `k`.
///
/// Under the consistency hypothesis, `n * mean_nees ~ chi2(n*k)`, so the bound
/// tightens as trials accumulate. Returns `(lower, upper)` in units of mean
/// NEES — compare directly against the average, and the ideal value is `k`.
#[must_use]
pub fn nees_bounds(n_trials: usize, state_dim: usize, alpha: Scalar) -> (Scalar, Scalar) {
    let dof = n_trials * state_dim;
    let lo = chi2_quantile(alpha * 0.5, dof) / n_trials as Scalar;
    let hi = chi2_quantile(1.0 - alpha * 0.5, dof) / n_trials as Scalar;
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn welford_matches_two_pass() {
        let xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mut w = Welford::new();
        for x in xs {
            w.push(x);
        }
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
        assert_relative_eq!(w.mean(), mean, epsilon = 1e-12);
        assert_relative_eq!(w.variance(), var, epsilon = 1e-12);
        assert_eq!(w.count(), 8);
    }

    #[test]
    fn welford_empty_and_singleton_are_safe() {
        let mut w = Welford::new();
        assert_eq!(w.variance(), 0.0);
        w.push(5.0);
        assert_eq!(w.variance(), 0.0);
        assert_eq!(w.mean(), 5.0);
    }

    #[test]
    fn percentile_endpoints_and_interpolation() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(percentile(&v, 0.0).unwrap(), 1.0);
        assert_relative_eq!(percentile(&v, 1.0).unwrap(), 4.0);
        assert_relative_eq!(median(&v).unwrap(), 2.5, epsilon = 1e-12);
        assert!(percentile(&[], 0.5).is_none());
    }

    #[test]
    fn ln_gamma_matches_known_values() {
        assert_relative_eq!(ln_gamma(1.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(ln_gamma(2.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(ln_gamma(5.0), 24f64.ln(), epsilon = 1e-11);
        // Gamma(0.5) = sqrt(pi)
        assert_relative_eq!(
            ln_gamma(0.5),
            std::f64::consts::PI.sqrt().ln(),
            epsilon = 1e-11
        );
    }

    #[test]
    fn chi2_cdf_matches_published_quantiles() {
        // chi2(0.95, k) from standard tables.
        for (k, x) in [
            (1usize, 3.841_459),
            (2, 5.991_465),
            (6, 12.591_587),
            (10, 18.307_038),
        ] {
            assert_relative_eq!(chi2_cdf(x, k), 0.95, epsilon = 1e-6);
        }
    }

    #[test]
    fn chi2_quantile_inverts_the_cdf() {
        for k in [1usize, 3, 6, 12, 60] {
            for p in [0.025, 0.5, 0.95, 0.975] {
                let x = chi2_quantile(p, k);
                assert_relative_eq!(chi2_cdf(x, k), p, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn nees_bounds_bracket_the_state_dimension() {
        // A consistent estimator averages NEES == state_dim; the interval must
        // contain it, and must tighten as trials accumulate.
        let (lo10, hi10) = nees_bounds(10, 6, 0.05);
        let (lo100, hi100) = nees_bounds(100, 6, 0.05);
        assert!(lo10 < 6.0 && 6.0 < hi10, "10 trials: [{lo10}, {hi10}]");
        assert!(lo100 < 6.0 && 6.0 < hi100, "100 trials: [{lo100}, {hi100}]");
        assert!(
            hi100 - lo100 < hi10 - lo10,
            "more trials must tighten the bound"
        );
        // Published value for the classic 100-trial, 6-DoF case is ~[5.35, 6.68].
        assert_relative_eq!(lo100, 5.35, epsilon = 0.05);
        assert_relative_eq!(hi100, 6.68, epsilon = 0.05);
    }

    #[test]
    fn mad_sigma_is_robust_to_an_outlier() {
        let clean = [1.0, 2.0, 3.0, 4.0, 5.0];
        let dirty = [1.0, 2.0, 3.0, 4.0, 1000.0];
        let a = mad_sigma(&clean).unwrap();
        let b = mad_sigma(&dirty).unwrap();
        assert!((a - b).abs() < 0.75 * a, "MAD moved too much: {a} -> {b}");
    }
}
