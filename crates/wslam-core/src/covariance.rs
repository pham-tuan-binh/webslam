//! Covariance propagation and consistency checking.
//!
//! spec.md §6 L6: *"We claim the covariance is meaningful. Prove it."* The
//! NEES and coverage machinery lives here so that every layer computes them the
//! same way and the harness has nothing to reimplement.

use crate::math::{Mat6, Scalar, Se3, Vec6};
use crate::stats::chi2_cdf;

/// Force a matrix to be symmetric. Floating-point accumulation in a Kalman
/// update drifts out of symmetry, and an asymmetric "covariance" produces
/// negative NEES that looks like a modelling bug.
#[must_use]
pub fn symmetrize(m: &Mat6) -> Mat6 {
    (m + m.transpose()) * 0.5
}

/// Nudge a covariance back to positive-definiteness by clamping its
/// eigenvalues below at `floor`.
///
/// Returns `None` if the symmetric eigendecomposition fails, which for a 6x6
/// real symmetric matrix means the input contained non-finite entries.
#[must_use]
pub fn enforce_psd(m: &Mat6, floor: Scalar) -> Option<Mat6> {
    if m.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let sym = symmetrize(m);
    let eig = sym.symmetric_eigen();
    let mut d = eig.eigenvalues;
    let mut changed = false;
    for v in d.iter_mut() {
        if *v < floor {
            *v = floor;
            changed = true;
        }
    }
    if !changed {
        return Some(sym);
    }
    Some(eig.eigenvectors * Mat6::from_diagonal(&d) * eig.eigenvectors.transpose())
}

/// Whether a matrix is a usable covariance: finite, symmetric to `tol`, and
/// with non-negative diagonal.
#[must_use]
pub fn is_valid_covariance(m: &Mat6, tol: Scalar) -> bool {
    if m.iter().any(|v| !v.is_finite()) {
        return false;
    }
    for i in 0..6 {
        if m[(i, i)] < 0.0 {
            return false;
        }
        for j in (i + 1)..6 {
            if (m[(i, j)] - m[(j, i)]).abs() > tol * (1.0 + m[(i, j)].abs()) {
                return false;
            }
        }
    }
    true
}

/// Linear covariance propagation: `J P J^T + Q`.
#[must_use]
pub fn propagate(p: &Mat6, j: &Mat6, q: &Mat6) -> Mat6 {
    symmetrize(&(j * p * j.transpose() + q))
}

/// Transport a covariance from one frame to another through an SE(3) adjoint.
///
/// Given `P` expressed in the tangent space at `T_a` and a rigid change of
/// reference `T_ba`, the covariance at `T_b` is `Adj(T_ba) P Adj(T_ba)^T`.
#[must_use]
pub fn transport(p: &Mat6, t_ba: &Se3) -> Mat6 {
    let adj = t_ba.adjoint();
    symmetrize(&(adj * p * adj.transpose()))
}

/// Normalised estimation error squared for a single pose trial.
///
/// `error` is the right-minus residual `estimate.minus(&truth)` in `[rho; phi]`
/// order, and `covariance` is the estimator's claimed covariance in the same
/// order and frame. Under a consistent estimator, NEES averages to 6.
///
/// Returns `None` if the covariance is not invertible, which is itself a
/// finding worth reporting rather than papering over.
#[must_use]
pub fn nees(error: &Vec6, covariance: &Mat6) -> Option<Scalar> {
    let inv = symmetrize(covariance).try_inverse()?;
    let v = (error.transpose() * inv * error)[(0, 0)];
    if v.is_finite() && v >= 0.0 {
        Some(v)
    } else {
        None
    }
}

/// NEES between an estimated and a true pose.
#[must_use]
pub fn pose_nees(estimate: &Se3, truth: &Se3, covariance: &Mat6) -> Option<Scalar> {
    nees(&estimate.minus(truth), covariance)
}

/// Accumulates NEES and coverage statistics across trials.
///
/// This is the object spec.md §6 L6 describes: NEES over >= 100 trials plus
/// empirical coverage at 68/95/99.
#[derive(Debug, Clone, Default)]
pub struct ConsistencyAccumulator {
    values: Vec<Scalar>,
    state_dim: usize,
    rejected: usize,
}

impl ConsistencyAccumulator {
    /// New accumulator for a given state dimension (6 for a full pose).
    #[must_use]
    pub fn new(state_dim: usize) -> Self {
        ConsistencyAccumulator {
            values: Vec::new(),
            state_dim,
            rejected: 0,
        }
    }

    /// Record one trial. Trials whose covariance was singular are counted
    /// separately rather than dropped silently.
    pub fn push(&mut self, nees_value: Option<Scalar>) {
        match nees_value {
            Some(v) => self.values.push(v),
            None => self.rejected += 1,
        }
    }

    /// Record a pose trial directly.
    pub fn push_pose(&mut self, estimate: &Se3, truth: &Se3, covariance: &Mat6) {
        self.push(pose_nees(estimate, truth, covariance));
    }

    /// Number of usable trials.
    #[must_use]
    pub fn count(&self) -> usize {
        self.values.len()
    }

    /// Trials discarded for a singular covariance.
    #[must_use]
    pub fn rejected(&self) -> usize {
        self.rejected
    }

    /// Average NEES. A consistent estimator gives `state_dim`.
    #[must_use]
    pub fn mean_nees(&self) -> Scalar {
        if self.values.is_empty() {
            Scalar::NAN
        } else {
            self.values.iter().sum::<Scalar>() / self.values.len() as Scalar
        }
    }

    /// Empirical coverage at a nominal confidence level: the fraction of trials
    /// whose NEES falls inside the corresponding chi-squared quantile.
    #[must_use]
    pub fn coverage(&self, nominal: Scalar) -> Scalar {
        if self.values.is_empty() {
            return Scalar::NAN;
        }
        let threshold = crate::stats::chi2_quantile(nominal, self.state_dim);
        let inside = self.values.iter().filter(|&&v| v <= threshold).count();
        inside as Scalar / self.values.len() as Scalar
    }

    /// The full report spec.md §6 L6 asks for.
    #[must_use]
    pub fn report(&self, alpha: Scalar) -> ConsistencyReport {
        let (lo, hi) = crate::stats::nees_bounds(self.values.len().max(1), self.state_dim, alpha);
        let mean = self.mean_nees();
        ConsistencyReport {
            trials: self.values.len(),
            rejected: self.rejected,
            state_dim: self.state_dim,
            mean_nees: mean,
            bounds: (lo, hi),
            consistent: mean.is_finite() && mean >= lo && mean <= hi,
            overconfident: mean.is_finite() && mean > hi,
            coverage_68: self.coverage(0.68),
            coverage_95: self.coverage(0.95),
            coverage_99: self.coverage(0.99),
        }
    }
}

/// Outcome of a consistency evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConsistencyReport {
    /// Usable trials.
    pub trials: usize,
    /// Trials rejected for a singular covariance.
    pub rejected: usize,
    /// State dimension the NEES was computed against.
    pub state_dim: usize,
    /// Average NEES; ideal value is `state_dim`.
    pub mean_nees: Scalar,
    /// Two-sided acceptance interval for the mean.
    pub bounds: (Scalar, Scalar),
    /// Whether the mean falls inside the interval.
    pub consistent: bool,
    /// Whether the mean exceeds the upper bound. Called out separately because
    /// spec.md §6 L6 says overconfidence is *"worse than no covariance at all"*
    /// — a conservative estimator is merely unhelpful, not misleading.
    pub overconfident: bool,
    /// Empirical coverage at nominal 68%.
    pub coverage_68: Scalar,
    /// Empirical coverage at nominal 95%.
    pub coverage_95: Scalar,
    /// Empirical coverage at nominal 99%.
    pub coverage_99: Scalar,
}

impl ConsistencyReport {
    /// Whether every coverage level is within `tol` of nominal — the M6 exit
    /// criterion is "coverage within 2% of nominal", i.e. `tol = 0.02`.
    #[must_use]
    pub fn coverage_within(&self, tol: Scalar) -> bool {
        (self.coverage_68 - 0.68).abs() <= tol
            && (self.coverage_95 - 0.95).abs() <= tol
            && (self.coverage_99 - 0.99).abs() <= tol
    }
}

impl std::fmt::Display for ConsistencyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NEES {:.3} (ideal {}, bounds [{:.3}, {:.3}], n={}{}) {} | coverage 68/95/99 = {:.1}%/{:.1}%/{:.1}%",
            self.mean_nees,
            self.state_dim,
            self.bounds.0,
            self.bounds.1,
            self.trials,
            if self.rejected > 0 {
                format!(", {} rejected", self.rejected)
            } else {
                String::new()
            },
            if self.consistent {
                "CONSISTENT"
            } else if self.overconfident {
                "OVERCONFIDENT"
            } else {
                "conservative"
            },
            100.0 * self.coverage_68,
            100.0 * self.coverage_95,
            100.0 * self.coverage_99,
        )
    }
}

/// Mahalanobis distance squared for an arbitrary-dimension residual.
#[must_use]
pub fn mahalanobis_sq<const N: usize>(
    error: &nalgebra::SVector<Scalar, N>,
    covariance: &nalgebra::SMatrix<Scalar, N, N>,
) -> Option<Scalar> {
    let inv = covariance.clone_owned().try_inverse()?;
    let v = (error.transpose() * inv * error)[(0, 0)];
    v.is_finite().then_some(v)
}

/// Chi-squared gate: is this residual consistent with its covariance at the
/// given confidence? The standard outlier test in a filter update.
#[must_use]
pub fn chi2_gate<const N: usize>(
    error: &nalgebra::SVector<Scalar, N>,
    covariance: &nalgebra::SMatrix<Scalar, N, N>,
    confidence: Scalar,
) -> bool {
    match mahalanobis_sq(error, covariance) {
        Some(d2) => chi2_cdf(d2, N) <= confidence,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{So3, Vec3};
    use crate::rng::DeterministicRng;
    use approx::assert_relative_eq;

    #[test]
    fn symmetrize_is_idempotent() {
        let mut m = Mat6::zeros();
        for i in 0..6 {
            for j in 0..6 {
                m[(i, j)] = (i * 7 + j * 3) as f64;
            }
        }
        let s = symmetrize(&m);
        assert_relative_eq!(s, symmetrize(&s), epsilon = 1e-15);
        assert_relative_eq!(s, s.transpose(), epsilon = 1e-15);
    }

    #[test]
    fn enforce_psd_lifts_negative_eigenvalues_only() {
        let m = Mat6::from_diagonal(&Vec6::new(1.0, -0.5, 2.0, 0.0, 3.0, -1e-9));
        let fixed = enforce_psd(&m, 1e-12).unwrap();
        let eig = fixed.symmetric_eigen().eigenvalues;
        assert!(eig.iter().all(|&v| v >= 1e-12 - 1e-15), "{eig:?}");
        // Already-positive entries survive.
        assert_relative_eq!(fixed[(2, 2)], 2.0, epsilon = 1e-9);
    }

    #[test]
    fn enforce_psd_rejects_non_finite() {
        let mut m = Mat6::identity();
        m[(0, 0)] = f64::NAN;
        assert!(enforce_psd(&m, 1e-12).is_none());
    }

    #[test]
    fn is_valid_covariance_catches_the_usual_failures() {
        assert!(is_valid_covariance(&Mat6::identity(), 1e-9));
        let mut neg = Mat6::identity();
        neg[(3, 3)] = -1.0;
        assert!(!is_valid_covariance(&neg, 1e-9));
        let mut asym = Mat6::identity();
        asym[(0, 1)] = 1.0;
        assert!(!is_valid_covariance(&asym, 1e-9));
        let mut nan = Mat6::identity();
        nan[(2, 2)] = f64::NAN;
        assert!(!is_valid_covariance(&nan, 1e-9));
    }

    #[test]
    fn propagation_of_identity_is_identity_plus_noise() {
        let p = Mat6::identity() * 0.5;
        let q = Mat6::identity() * 0.25;
        assert_relative_eq!(
            propagate(&p, &Mat6::identity(), &q),
            Mat6::identity() * 0.75,
            epsilon = 1e-15
        );
    }

    #[test]
    fn transport_preserves_trace_under_pure_rotation() {
        // A rotation is orthogonal, so Adj P Adj^T is a similarity transform
        // and the trace is invariant.
        let p = Mat6::from_diagonal(&Vec6::new(1.0, 2.0, 3.0, 0.1, 0.2, 0.3));
        let t = Se3::from_rotation(So3::exp(&Vec3::new(0.3, -0.2, 0.5)));
        let q = transport(&p, &t);
        assert_relative_eq!(q.trace(), p.trace(), epsilon = 1e-9);
    }

    #[test]
    fn nees_of_zero_error_is_zero() {
        assert_relative_eq!(
            nees(&Vec6::zeros(), &Mat6::identity()).unwrap(),
            0.0,
            epsilon = 1e-15
        );
    }

    #[test]
    fn nees_of_one_sigma_in_each_axis_is_state_dim() {
        let cov = Mat6::from_diagonal(&Vec6::new(4.0, 4.0, 4.0, 0.01, 0.01, 0.01));
        let err = Vec6::new(2.0, 2.0, 2.0, 0.1, 0.1, 0.1); // exactly 1 sigma each
        assert_relative_eq!(nees(&err, &cov).unwrap(), 6.0, epsilon = 1e-12);
    }

    #[test]
    fn nees_rejects_singular_covariance() {
        assert!(nees(&Vec6::repeat(1.0), &Mat6::zeros()).is_none());
    }

    #[test]
    fn a_correctly_calibrated_estimator_passes_the_consistency_check() {
        // Synthesise trials whose error really is drawn from the claimed
        // covariance. This is the Tier-1 self-test of the L6 machinery: if a
        // perfectly calibrated estimator failed here, the harness would be
        // wrong rather than the estimator.
        let mut rng = DeterministicRng::new("nees-selftest", 20260801);
        let sigma = Vec6::new(0.05, 0.05, 0.08, 0.01, 0.01, 0.02);
        let cov = Mat6::from_diagonal(&sigma.map(|s| s * s));
        let truth = Se3::new(
            So3::exp(&Vec3::new(0.1, 0.2, -0.3)),
            Vec3::new(1.0, 2.0, 3.0),
        );

        let mut acc = ConsistencyAccumulator::new(6);
        for _ in 0..2000 {
            let delta = Vec6::from_iterator((0..6).map(|i| rng.normal() * sigma[i]));
            acc.push_pose(&truth.plus(&delta), &truth, &cov);
        }

        let r = acc.report(0.05);
        assert!(r.consistent, "{r}");
        assert!(!r.overconfident, "{r}");
        assert!(r.coverage_within(0.03), "{r}");
        assert_eq!(r.rejected, 0);
    }

    #[test]
    fn an_overconfident_estimator_is_flagged_as_such() {
        let mut rng = DeterministicRng::new("nees-overconfident", 7);
        let true_sigma = Vec6::repeat(0.1);
        // Claim 4x smaller sigma than reality.
        let claimed = Mat6::from_diagonal(&true_sigma.map(|s| (s * 0.25).powi(2)));
        let truth = Se3::identity();

        let mut acc = ConsistencyAccumulator::new(6);
        for _ in 0..500 {
            let delta = Vec6::from_iterator((0..6).map(|i| rng.normal() * true_sigma[i]));
            acc.push_pose(&truth.plus(&delta), &truth, &claimed);
        }
        let r = acc.report(0.05);
        assert!(r.overconfident, "{r}");
        assert!(!r.consistent, "{r}");
        assert!(r.coverage_95 < 0.5, "{r}");
    }

    #[test]
    fn chi2_gate_accepts_inliers_and_rejects_outliers() {
        let cov = nalgebra::Matrix2::new(1.0, 0.0, 0.0, 1.0);
        let inlier = nalgebra::Vector2::new(0.5, 0.5);
        let outlier = nalgebra::Vector2::new(10.0, 10.0);
        assert!(chi2_gate(&inlier, &cov, 0.99));
        assert!(!chi2_gate(&outlier, &cov, 0.99));
    }
}
