//! The camera-IMU offset `td` as an online filter state.
//!
//! Li & Mourikis (IJRR 2014) put `td` directly in the estimator state, prove it
//! is recoverable under generic motion, and — the part that matters here —
//! *identify the motions under which it is not*. Their §V degenerate cases are
//! the ones a phone spends most of its life in: no motion at all, motion along a
//! single axis, and constant angular velocity. A filter that keeps applying
//! measurements through such a stretch does not stall; it wanders, because the
//! measurements carry no information about `td` but do carry noise, and the
//! covariance keeps shrinking as if they did. The result is a confident wrong
//! offset with no symptom until the pose is wrong too.
//!
//! Hence [`OffsetFilter::set_degenerate`]. While it is set the filter propagates
//! — the uncertainty **grows**, which is the honest thing for it to do — and
//! refuses to correct. Downstream, [`crate::FittedTimeBase`] reports that
//! inflated variance through [`wslam_core::TimeBase::offset_variance`] and closes
//! tier 3 until real excitation returns.

/// Consecutive gate rejections after which the filter concludes that the world
/// moved rather than that the measurement is wrong, and reopens.
///
/// Without this a gate plus an over-confident posterior is a trap: `td` genuinely
/// jumps (the camera pipeline reconfigures, the page is backgrounded and
/// resumed), every subsequent measurement fails the gate, and the filter defends
/// a stale estimate forever.
const REJECTIONS_BEFORE_REOPENING: u32 = 5;

/// Scalar Kalman filter on the camera-IMU temporal offset, seconds.
///
/// Positive `td` means camera stamps lag IMU stamps; see the crate header for
/// the sign convention and how it composes.
#[derive(Debug, Clone)]
pub struct OffsetFilter {
    offset: f64,
    variance: f64,
    process_noise: f64,
    initial_variance: f64,
    degenerate: bool,
    /// Gate on normalised innovation squared. Infinite means no gating, which is
    /// the default — see [`OffsetFilter::set_innovation_gate`].
    gate: f64,
    updates: u64,
    consecutive_rejections: u32,
}

impl OffsetFilter {
    /// A filter starting at `td = 0` with the given prior variance.
    ///
    /// `process_noise` is the random-walk variance added **per epoch**, not per
    /// second. The contracted `update` carries no timestep, and the quantity is
    /// naturally per-epoch anyway: `td` drifts with the skew between two sensor
    /// clocks that are each ticking at their own fixed cadence. Callers who skip
    /// an epoch should call [`OffsetFilter::propagate`] so the estimate still
    /// ages.
    ///
    /// A sensible prior for a browser is `(50 ms)^2`: Huai et al. measured up to
    /// 30 ms with native API access (arXiv:2001.00470) and spec.md §5 says
    /// plainly that "the browser will be worse".
    #[must_use]
    pub fn new(initial_variance: f64, process_noise: f64) -> Self {
        let initial_variance = if initial_variance.is_finite() && initial_variance > 0.0 {
            initial_variance
        } else {
            f64::INFINITY
        };
        let process_noise = if process_noise.is_finite() && process_noise >= 0.0 {
            process_noise
        } else {
            0.0
        };
        OffsetFilter {
            offset: 0.0,
            variance: initial_variance,
            process_noise,
            initial_variance,
            degenerate: false,
            gate: f64::INFINITY,
            updates: 0,
            consecutive_rejections: 0,
        }
    }

    /// Fold in a measurement of the offset.
    ///
    /// Always propagates first, so calling this once per epoch keeps the
    /// covariance honest whether or not the correction is applied. Under
    /// degeneracy the correction is skipped and only the propagation happens.
    pub fn update(&mut self, measured_offset: f64, measurement_variance: f64) {
        self.propagate();

        if self.degenerate {
            return;
        }
        if !measured_offset.is_finite()
            || !measurement_variance.is_finite()
            || measurement_variance <= 0.0
        {
            log::warn!(
                "offset: ignoring measurement {measured_offset} with variance {measurement_variance}"
            );
            return;
        }

        let innovation = measured_offset - self.offset;
        let innovation_variance = self.variance + measurement_variance;
        if innovation * innovation > self.gate * innovation_variance {
            self.consecutive_rejections += 1;
            if self.consecutive_rejections >= REJECTIONS_BEFORE_REOPENING {
                // Inflate to at least the size of what we keep rejecting, so the
                // gate lets the next one through instead of defending a stale
                // estimate indefinitely.
                self.variance = self.variance.max(innovation * innovation);
            }
            return;
        }
        self.consecutive_rejections = 0;

        if self.variance.is_finite() {
            let gain = self.variance / innovation_variance;
            self.offset += gain * innovation;
            // Scalar form; (1-K)P stays positive by construction because K < 1
            // whenever measurement_variance > 0, so Joseph form buys nothing.
            self.variance *= 1.0 - gain;
        } else {
            // Uninformative prior. Written out rather than reached through the
            // gain, because K = P/(P+R) evaluates to inf/inf = NaN and (1-K)P to
            // 0*inf = NaN, and a NaN covariance is silent for a long time.
            self.offset = measured_offset;
            self.variance = measurement_variance;
        }
        self.updates += 1;
    }

    /// Age the estimate by one epoch without a measurement.
    pub fn propagate(&mut self) {
        self.variance += self.process_noise;
    }

    /// Current offset estimate, seconds.
    #[must_use]
    pub fn offset(&self) -> f64 {
        self.offset
    }

    /// Variance of the offset estimate, seconds squared.
    #[must_use]
    pub fn variance(&self) -> f64 {
        self.variance
    }

    /// Suspend estimation under degenerate motion, per Li & Mourikis §V.
    ///
    /// The caller decides what "degenerate" means, because it depends on state
    /// this layer cannot see: near-zero excitation, single-axis translation, or
    /// constant angular rate. [`wslam_core::StateWindow::mean_excitation`] and
    /// [`wslam_core::StateWindow::peak_angular_rate`] are the intended inputs.
    pub fn set_degenerate(&mut self, degenerate: bool) {
        if degenerate != self.degenerate {
            log::debug!("offset: degenerate={degenerate}");
        }
        self.degenerate = degenerate;
    }

    /// Whether estimation is currently suspended.
    #[must_use]
    pub fn is_degenerate(&self) -> bool {
        self.degenerate
    }

    /// Number of measurements actually applied. Rejected and suspended epochs do
    /// not count, which makes this the right thing to gate a convergence claim
    /// on.
    #[must_use]
    pub fn update_count(&self) -> u64 {
        self.updates
    }

    /// Seed a prior from an offline calibration — a rig measurement, or a value
    /// persisted from an earlier session on the same device.
    ///
    /// Overwrites rather than fuses: a prior is a statement about where the
    /// filter should start, and fusing it with whatever the filter had drifted
    /// to would defeat the point.
    pub fn set_prior(&mut self, offset: f64, variance: f64) {
        if !offset.is_finite() || !variance.is_finite() || variance <= 0.0 {
            log::warn!("offset: ignoring invalid prior {offset} +/- {variance}");
            return;
        }
        self.offset = offset;
        self.variance = variance;
        self.consecutive_rejections = 0;
    }

    /// Reject measurements whose normalised innovation squared exceeds `gate`.
    ///
    /// Off by default (`gate = infinity`). A cross-correlation peak on a
    /// low-excitation stretch can land anywhere, and gating is the standard
    /// defence — but a gate on a scalar filter with a small posterior is also a
    /// way to lock out the truth, so it is opt-in and paired with the
    /// reopening rule above. `gate = 25` is 5 sigma.
    pub fn set_innovation_gate(&mut self, gate: f64) {
        self.gate = if gate.is_finite() && gate > 0.0 {
            gate
        } else {
            f64::INFINITY
        };
    }

    /// Return to the prior, keeping the tuning. Used on stream restart.
    pub fn reset(&mut self) {
        self.offset = 0.0;
        self.variance = self.initial_variance;
        self.degenerate = false;
        self.updates = 0;
        self.consecutive_rejections = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wslam_core::DeterministicRng;

    #[test]
    fn single_update_matches_the_closed_form() {
        let (p0, r, z) = (4.0e-4, 1.0e-4, 0.030);
        let mut f = OffsetFilter::new(p0, 0.0);
        f.update(z, r);
        let gain = p0 / (p0 + r); // 0.8
        assert!((f.offset() - gain * z).abs() < 1e-18, "{}", f.offset());
        assert!((f.variance() - (1.0 - gain) * p0).abs() < 1e-22);
        assert_eq!(f.update_count(), 1);
    }

    #[test]
    fn two_sequential_updates_equal_one_batch_fusion() {
        // With no process noise the Kalman recursion is exactly information
        // accumulation, and that has a closed form we can check against.
        let (p0, r1, r2) = (2.5e-3, 4.0e-5, 9.0e-5);
        let (z1, z2) = (0.021, 0.026);
        let mut f = OffsetFilter::new(p0, 0.0);
        f.update(z1, r1);
        f.update(z2, r2);

        let precision = 1.0 / p0 + 1.0 / r1 + 1.0 / r2;
        let mean = (0.0 / p0 + z1 / r1 + z2 / r2) / precision;
        assert!(
            (f.offset() - mean).abs() < 1e-15,
            "{} vs {mean}",
            f.offset()
        );
        assert!((f.variance() - 1.0 / precision).abs() < 1e-18);
    }

    #[test]
    fn converges_to_a_constant_offset_under_noise() {
        let truth = 0.0247;
        let r: f64 = 4.0e-6; // (2 ms)^2
        let mut rng = DeterministicRng::new("offset", 17);
        let mut f = OffsetFilter::new(2.5e-3, 1.0e-10);
        for _ in 0..400 {
            f.update(truth + rng.normal_with(0.0, r.sqrt()), r);
        }
        assert!(
            (f.offset() - truth).abs() < 3.0 * f.variance().sqrt(),
            "estimate {} vs truth {truth}, sigma {:.3e}",
            f.offset(),
            f.variance().sqrt()
        );
        assert!((f.offset() - truth).abs() < 5.0e-4);
        assert!(f.variance() < 1.0e-7);
    }

    #[test]
    fn variance_grows_while_degenerate_and_shrinks_on_updates() {
        let mut f = OffsetFilter::new(2.5e-3, 1.0e-8);
        for _ in 0..50 {
            f.update(0.020, 1.0e-6);
        }
        let converged_offset = f.offset();
        let converged_variance = f.variance();
        assert!(converged_variance < 2.5e-3);

        f.set_degenerate(true);
        for _ in 0..50 {
            // Deliberately informative-looking measurements: under degeneracy
            // they must be ignored no matter how tight they claim to be.
            f.update(0.200, 1.0e-12);
        }
        let degenerate_variance = f.variance();
        assert!(
            degenerate_variance > converged_variance,
            "variance must grow while suspended: {degenerate_variance:.3e} !> {converged_variance:.3e}"
        );
        assert_eq!(
            f.offset().to_bits(),
            converged_offset.to_bits(),
            "a suspended filter must not move the estimate"
        );
        assert_eq!(f.update_count(), 50, "suspended epochs must not count");

        f.set_degenerate(false);
        for _ in 0..50 {
            f.update(0.020, 1.0e-6);
        }
        assert!(
            f.variance() < degenerate_variance,
            "variance must shrink once updates resume: {:.3e} !< {degenerate_variance:.3e}",
            f.variance()
        );
    }

    #[test]
    fn propagate_alone_only_grows_the_variance() {
        let q = 1.0e-9;
        let mut f = OffsetFilter::new(1.0e-6, q);
        f.update(0.01, 1.0e-8);
        let (o, v) = (f.offset(), f.variance());
        f.propagate();
        f.propagate();
        assert_eq!(f.offset().to_bits(), o.to_bits());
        assert!((f.variance() - (v + 2.0 * q)).abs() < 1e-22);
    }

    #[test]
    fn ignores_impossible_measurement_variances() {
        let mut f = OffsetFilter::new(1.0e-4, 0.0);
        let before = f.offset();
        f.update(0.05, 0.0);
        f.update(0.05, -1.0);
        f.update(0.05, f64::NAN);
        f.update(f64::NAN, 1.0e-6);
        f.update(f64::INFINITY, 1.0e-6);
        assert_eq!(f.offset().to_bits(), before.to_bits());
        assert_eq!(f.update_count(), 0);
        assert!(f.variance().is_finite());
    }

    #[test]
    fn an_unusable_prior_variance_becomes_an_uninformative_one() {
        // Zero prior variance would freeze the filter at zero forever, which is
        // the one failure mode that looks like success.
        let mut f = OffsetFilter::new(0.0, 0.0);
        assert!(f.variance().is_infinite());
        f.update(0.031, 1.0e-6);
        assert!((f.offset() - 0.031).abs() < 1e-12, "{}", f.offset());
        assert!((f.variance() - 1.0e-6).abs() < 1e-18);
    }

    #[test]
    fn innovation_gate_rejects_a_wild_measurement_then_reopens() {
        let mut f = OffsetFilter::new(1.0e-4, 0.0);
        f.set_innovation_gate(25.0); // 5 sigma
        for _ in 0..30 {
            f.update(0.020, 1.0e-8);
        }
        let settled = f.offset();
        assert!((settled - 0.020).abs() < 1e-4);

        // One absurd measurement must not move it.
        f.update(5.0, 1.0e-8);
        assert!((f.offset() - settled).abs() < 1e-9);

        // But a persistent new truth must eventually be believed, or the gate is
        // a lock rather than a filter.
        for _ in 0..40 {
            f.update(0.120, 1.0e-8);
        }
        assert!(
            (f.offset() - 0.120).abs() < 1e-3,
            "gate locked out a real jump: {}",
            f.offset()
        );
    }

    #[test]
    fn gate_off_by_default_accepts_everything() {
        let mut f = OffsetFilter::new(1.0e-4, 0.0);
        for _ in 0..30 {
            f.update(0.020, 1.0e-8);
        }
        let before = f.offset();
        f.update(5.0, 1.0e-8);
        assert!(f.offset() > before);
    }

    #[test]
    fn prior_overwrites_and_reset_restores() {
        let mut f = OffsetFilter::new(2.5e-3, 0.0);
        f.set_prior(0.018, 1.0e-6);
        assert!((f.offset() - 0.018).abs() < 1e-18);
        assert!((f.variance() - 1.0e-6).abs() < 1e-24);
        f.set_prior(f64::NAN, 1.0e-6);
        f.set_prior(0.018, -1.0);
        assert!(
            (f.offset() - 0.018).abs() < 1e-18,
            "invalid priors must be ignored"
        );
        f.reset();
        assert_eq!(f.offset(), 0.0);
        assert!((f.variance() - 2.5e-3).abs() < 1e-18);
        assert!(!f.is_degenerate());
    }

    proptest::proptest! {
        /// With no process noise the scalar Kalman recursion *is* information
        /// accumulation, so any order and any number of updates must land on the
        /// closed-form inverse-variance fusion of the prior and every
        /// measurement. Checking against the closed form rather than against a
        /// second run of the same recursion is the point.
        #[test]
        fn sequential_updates_equal_closed_form_information_fusion(
            prior_variance in 1.0e-6f64..1.0e-1f64,
            measurements in proptest::collection::vec(
                (-0.1f64..0.1f64, 1.0e-8f64..1.0e-3f64),
                1..8,
            ),
        ) {
            let mut f = OffsetFilter::new(prior_variance, 0.0);
            for &(z, r) in &measurements {
                f.update(z, r);
            }

            let mut precision = 1.0 / prior_variance;
            let mut weighted = 0.0; // prior mean is zero
            for &(z, r) in &measurements {
                precision += 1.0 / r;
                weighted += z / r;
            }
            let mean = weighted / precision;
            let variance = 1.0 / precision;

            proptest::prop_assert!(
                (f.offset() - mean).abs() <= 1.0e-9 * (1.0 + mean.abs()),
                "offset {} vs closed form {mean}",
                f.offset()
            );
            proptest::prop_assert!(
                (f.variance() - variance).abs() <= 1.0e-9 * variance,
                "variance {} vs closed form {variance}",
                f.variance()
            );
        }

        /// A suspended filter is a no-op on the estimate, whatever it is fed.
        #[test]
        fn suspension_is_a_no_op_on_the_estimate(
            measurements in proptest::collection::vec(
                (-10.0f64..10.0f64, 1.0e-12f64..1.0e-2f64),
                1..16,
            ),
        ) {
            let mut f = OffsetFilter::new(2.5e-3, 1.0e-8);
            f.update(0.02, 1.0e-6);
            let held = f.offset();
            f.set_degenerate(true);
            for &(z, r) in &measurements {
                f.update(z, r);
            }
            proptest::prop_assert_eq!(f.offset().to_bits(), held.to_bits());
            proptest::prop_assert!(f.variance() > 0.0);
        }
    }

    #[test]
    fn degeneracy_toggles_are_reported() {
        let mut f = OffsetFilter::new(1.0e-4, 0.0);
        assert!(!f.is_degenerate());
        f.set_degenerate(true);
        assert!(f.is_degenerate());
        f.set_degenerate(false);
        assert!(!f.is_degenerate());
    }
}
