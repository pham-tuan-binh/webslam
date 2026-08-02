//! Noise model and gating thresholds.

use wslam_core::math::Scalar;

/// Tuning for [`crate::OrientationFilter`].
///
/// The two noise terms are in different units on purpose, because they enter the
/// filter at different places:
///
/// - `gyro_noise` is a **spectral density** (rad/s/sqrt(Hz)). It is integrated
///   over the propagation interval, so the process noise it contributes scales
///   with `dt` and the filter is invariant to the sample rate.
/// - `accel_noise` is the standard deviation of a **single sample** (m/s^2). It
///   is a measurement covariance, not a density, and it does not scale with
///   `dt`. Treating it as a density would make a 200 Hz phone trust its
///   accelerometer twice as much as a 100 Hz one for no physical reason.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrientationConfig {
    /// Gyroscope angular random walk, rad/s/sqrt(Hz).
    pub gyro_noise: Scalar,
    /// Gyroscope bias random walk, rad/s^2/sqrt(Hz).
    pub gyro_bias_walk: Scalar,
    /// Accelerometer standard deviation of a single sample, m/s^2.
    pub accel_noise: Scalar,
    /// Reject a gravity update when `| |a| - g |` exceeds this, m/s^2.
    ///
    /// Under linear acceleration the accelerometer does not measure gravity, and
    /// an ungated update tilts the estimate toward the acceleration. The gate is
    /// a magnitude test because that is the only cue available from one sample —
    /// and it is an imperfect one: horizontal acceleration `a` perpendicular to
    /// gravity changes the magnitude by only `a^2 / 2g` while tilting the
    /// direction by `atan(a/g)`. A 3 m/s^2 sideways shove passes a 0.5 m/s^2 gate
    /// while pointing 17 degrees off vertical. What keeps that survivable is that
    /// the update is a low-gain filter step, not an assignment, and that human
    /// linear acceleration is close to zero-mean over a second. Sustained
    /// one-directional acceleration — a car pulling away — will bias roll/pitch,
    /// and no magnitude gate can prevent it.
    pub gravity_gate: Scalar,
    /// Angular rate below which the device counts as rotationally static, rad/s.
    ///
    /// Enables a zero-angular-rate update on the gyro bias: a device that is not
    /// turning reports its own bias, which is the single most direct observation
    /// of the bias state the filter will ever get. Set to zero to disable.
    ///
    /// Keep it small. A hand-held phone tremors at 0.02-0.05 rad/s, so a
    /// generous threshold quietly turns real hand motion into bias.
    pub static_threshold: Scalar,
}

impl Default for OrientationConfig {
    /// Defaults for a phone IMU as delivered by `DeviceMotion`.
    ///
    /// The gyro figures are an order of magnitude worse than a bare MEMS part's
    /// datasheet (an LSM6DSO is ~7e-5 rad/s/sqrt(Hz)) because the browser hands
    /// us quantised, platform-filtered, jitter-delivered samples rather than the
    /// raw sensor, and a filter that believes the datasheet under-weights the
    /// accelerometer and drifts.
    fn default() -> Self {
        OrientationConfig {
            gyro_noise: 2.0e-3,
            gyro_bias_walk: 1.0e-4,
            accel_noise: 0.15,
            gravity_gate: 0.5,
            static_threshold: 0.02,
        }
    }
}

impl OrientationConfig {
    /// The ablation arm: accept every accelerometer sample regardless of
    /// magnitude.
    ///
    /// Exists so that "the gate earns its place" is a measurement rather than an
    /// assertion — the same discipline spec.md §6 L2 imposes on the distortion
    /// and lever-arm ablations. Not a shipping configuration.
    #[must_use]
    pub fn ungated() -> Self {
        OrientationConfig {
            gravity_gate: Scalar::INFINITY,
            ..Self::default()
        }
    }

    /// Replace any value that would produce a non-finite or non-positive
    /// covariance with the default, logging what was changed.
    ///
    /// [`crate::OrientationFilter::new`] cannot return a `Result` (the interface
    /// is frozen in `docs/CONTRACT.md`), and a filter constructed from a NaN
    /// noise term produces NaN poses forever without ever saying why. Sanitising
    /// loudly is the least-bad option available at this signature.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let d = Self::default();
        let fix = |name: &str, value: Scalar, fallback: Scalar| -> Scalar {
            if value.is_finite() && value > 0.0 {
                value
            } else {
                log::warn!("OrientationConfig::{name} = {value} is unusable; using {fallback}");
                fallback
            }
        };
        OrientationConfig {
            gyro_noise: fix("gyro_noise", self.gyro_noise, d.gyro_noise),
            gyro_bias_walk: fix("gyro_bias_walk", self.gyro_bias_walk, d.gyro_bias_walk),
            accel_noise: fix("accel_noise", self.accel_noise, d.accel_noise),
            // Infinity is meaningful here: it is the ungated ablation.
            gravity_gate: if self.gravity_gate > 0.0 && !self.gravity_gate.is_nan() {
                self.gravity_gate
            } else {
                log::warn!("OrientationConfig::gravity_gate must be positive; using default");
                d.gravity_gate
            },
            // Zero is meaningful here: it disables the zero-rate update.
            static_threshold: if self.static_threshold.is_finite() && self.static_threshold >= 0.0 {
                self.static_threshold
            } else {
                log::warn!("OrientationConfig::static_threshold must be finite and >= 0; using 0");
                0.0
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_all_usable() {
        let c = OrientationConfig::default();
        assert_eq!(c, c.sanitized());
        assert!(c.static_threshold > 0.0);
    }

    #[test]
    fn sanitize_replaces_nonsense_but_keeps_the_meaningful_extremes() {
        let bad = OrientationConfig {
            gyro_noise: f64::NAN,
            gyro_bias_walk: -1.0,
            accel_noise: 0.0,
            gravity_gate: f64::INFINITY,
            static_threshold: 0.0,
        }
        .sanitized();
        let d = OrientationConfig::default();
        assert_eq!(bad.gyro_noise, d.gyro_noise);
        assert_eq!(bad.gyro_bias_walk, d.gyro_bias_walk);
        assert_eq!(bad.accel_noise, d.accel_noise);
        // Infinite gate = ungated ablation, zero threshold = zero-rate disabled.
        assert!(bad.gravity_gate.is_infinite());
        assert_eq!(bad.static_threshold, 0.0);
    }

    #[test]
    fn sanitize_rejects_a_nan_gate_and_a_negative_threshold() {
        let fixed = OrientationConfig {
            gravity_gate: f64::NAN,
            static_threshold: -1.0,
            ..OrientationConfig::default()
        }
        .sanitized();
        assert_eq!(
            fixed.gravity_gate,
            OrientationConfig::default().gravity_gate
        );
        assert_eq!(fixed.static_threshold, 0.0);
    }

    #[test]
    fn ungated_accepts_everything() {
        assert!(OrientationConfig::ungated().gravity_gate.is_infinite());
    }
}
