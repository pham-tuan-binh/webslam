//! Synthetic delivery-jitter profiles for the Tier-1 tests.
//!
//! spec.md §6 Tier 1 wants "clock-model recovery from synthetic jitter" with a
//! closed-form answer. That needs a jitter model, and the model has to have the
//! two properties that make browser delivery jitter hard, or the tests prove
//! nothing:
//!
//! - **One-sided.** An event can be delivered late; it cannot be delivered
//!   early. So the noise has a positive mean, which biases the *intercept* of a
//!   cadence fit while leaving the *slope* clean. Symmetric Gaussian noise would
//!   quietly hide that.
//! - **Heavy-tailed.** Typical delivery is a few milliseconds late; a layout
//!   pass, a GC, or a backgrounded tab produces occasional stalls two orders
//!   larger. A profile without stalls makes least squares look fine and makes
//!   the robust fit look like ceremony.
//!
//! The numbers below are plausible, not measured. They exist so the estimator
//! can be tested, and they are not a claim about any browser — see the crate
//! header.

use wslam_core::DeterministicRng;

/// Exponential body plus a Bernoulli stall, in seconds.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DeliveryJitter {
    /// Mean of the exponential body, seconds.
    pub mean: f64,
    /// Probability that an event is additionally stalled.
    pub stall_probability: f64,
    /// Stall duration is uniform over this range, seconds.
    pub stall_min: f64,
    /// Upper end of the stall range, seconds.
    pub stall_max: f64,
}

impl DeliveryJitter {
    /// Draw one delay. Always non-negative.
    pub(crate) fn sample(&self, rng: &mut DeterministicRng) -> f64 {
        // Inverse-CDF exponential; `uniform()` is [0,1) so `1 - u` is (0,1] and
        // the log is safe.
        let base = -self.mean * (1.0 - rng.uniform()).ln();
        let stall = if rng.uniform() < self.stall_probability {
            rng.uniform_range(self.stall_min, self.stall_max)
        } else {
            0.0
        };
        base + stall
    }
}

/// `DeviceMotion` delivery on a busy main thread at 60 Hz.
pub(crate) const IMU_60HZ: DeliveryJitter = DeliveryJitter {
    mean: 3.0e-3,
    stall_probability: 0.03,
    stall_min: 20.0e-3,
    stall_max: 120.0e-3,
};

/// `requestVideoFrameCallback` `mediaTime` at 30 Hz.
///
/// Far gentler than the motion stream, and deliberately so: `mediaTime` rides
/// the media clock rather than the event loop (spec.md §4 L0), which is the
/// whole reason the video side uses it. What is left is capture-clock
/// quantisation plus the occasional dropped-and-resynchronised frame.
pub(crate) const CAMERA_30HZ: DeliveryJitter = DeliveryJitter {
    mean: 0.3e-3,
    stall_probability: 0.01,
    stall_min: 1.0e-3,
    stall_max: 4.0e-3,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitter_is_never_negative() {
        let mut rng = DeterministicRng::new("synth", 1);
        for _ in 0..20_000 {
            assert!(IMU_60HZ.sample(&mut rng) >= 0.0);
        }
    }

    #[test]
    fn jitter_is_right_skewed_with_a_heavy_tail() {
        let mut rng = DeterministicRng::new("synth", 2);
        let xs: Vec<f64> = (0..20_000).map(|_| IMU_60HZ.sample(&mut rng)).collect();
        let median = wslam_core::stats::median(&xs).unwrap();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let p999 = wslam_core::stats::percentile(&xs, 0.999).unwrap();
        assert!(mean > median, "mean {mean} must exceed median {median}");
        assert!(p999 > 20.0 * median, "tail p99.9 {p999} vs median {median}");
    }
}
