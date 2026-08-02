//! Inertial samples, in the shape the browser actually delivers them.
//!
//! `DeviceMotionEvent` gives `acceleration` (gravity removed by the platform),
//! `accelerationIncludingGravity`, `rotationRate` in **degrees per second**, and
//! a nominal `interval`. We carry all of it, converted to SI radians, and we
//! keep the raw delivery index because L0 fits its clock model over index rather
//! than over the jittery delivery stamp (spec.md §4 L0).

use crate::math::{Scalar, Vec3};
use crate::time::Timestamp;

/// One `DeviceMotionEvent`, converted to SI units but otherwise untouched.
///
/// The TypeScript shim performs no smoothing, reordering or buffering
/// (spec.md §7: *"a shim that helpfully cleans up its inputs destroys the
/// signal it is supposed to deliver"*), so the jitter visible here is real.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionEvent {
    /// Delivery index, monotonically increasing, never reset.
    pub index: u64,
    /// Raw arrival stamp in milliseconds as observed by the shim. **Not** in
    /// the unified timebase — that is what [`crate::TimeBase::map_motion`] is
    /// for. Present so L0 can measure the jitter it exists to correct.
    pub arrival_millis: f64,
    /// Angular rate in rad/s, body frame. Converted from the browser's deg/s.
    pub gyro: Vec3,
    /// Specific force in m/s^2 including gravity, body frame.
    pub accel_with_gravity: Vec3,
    /// Platform's gravity-compensated acceleration in m/s^2, if provided.
    /// iOS and Android both supply it, but its filter is undocumented, so L1
    /// prefers `accel_with_gravity` and does its own gravity handling.
    pub accel_linear: Option<Vec3>,
    /// Nominal sampling interval in seconds as reported by the platform.
    pub nominal_interval: f64,
}

impl MotionEvent {
    /// Build from browser-native units: degrees per second and m/s^2.
    #[must_use]
    pub fn from_browser(
        index: u64,
        arrival_millis: f64,
        gyro_deg_per_s: Vec3,
        accel_with_gravity: Vec3,
        accel_linear: Option<Vec3>,
        nominal_interval: f64,
    ) -> Self {
        MotionEvent {
            index,
            arrival_millis,
            gyro: gyro_deg_per_s.map(Scalar::to_radians),
            accel_with_gravity,
            accel_linear,
            nominal_interval,
        }
    }
}

/// A motion sample once it has been placed in the unified timebase.
///
/// This is what L1 and L0 consume. The split from [`MotionEvent`] is deliberate:
/// only the clock layer may construct one, so an un-timestamped sample cannot
/// reach the estimator by accident.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuSample {
    /// Capture time in the unified timebase.
    pub timestamp: Timestamp,
    /// Angular rate, rad/s, body frame.
    pub gyro: Vec3,
    /// Specific force including gravity, m/s^2, body frame.
    pub accel: Vec3,
}

impl ImuSample {
    /// Construct a sample already in the unified timebase.
    #[must_use]
    pub fn new(timestamp: Timestamp, gyro: Vec3, accel: Vec3) -> Self {
        ImuSample {
            timestamp,
            gyro,
            accel,
        }
    }

    /// Linear interpolation between two samples at time `t`.
    ///
    /// Used to align IMU to frame times. `t` outside `[a, b]` extrapolates,
    /// which the caller should avoid but which is better than clamping
    /// silently.
    #[must_use]
    pub fn lerp(a: &ImuSample, b: &ImuSample, t: Timestamp) -> ImuSample {
        let span = b.timestamp.since(a.timestamp);
        let alpha = if span.abs() < 1e-12 {
            0.0
        } else {
            t.since(a.timestamp) / span
        };
        ImuSample {
            timestamp: t,
            gyro: a.gyro + (b.gyro - a.gyro) * alpha,
            accel: a.accel + (b.accel - a.accel) * alpha,
        }
    }

    /// Magnitude of specific force minus gravity, a cheap proxy for
    /// translational excitation.
    ///
    /// spec.md §6 L5 requires reporting *"error against measured excitation"*
    /// rather than a single aggregate, because monocular inertial scale becomes
    /// unobservable as translational acceleration vanishes. This is the measure.
    #[must_use]
    pub fn excitation(&self, gravity_magnitude: Scalar) -> Scalar {
        (self.accel.norm() - gravity_magnitude).abs()
    }
}

/// Standard gravity, m/s^2.
pub const GRAVITY: Scalar = 9.80665;

/// Source of inertial samples. Live and replay implementations, same as
/// [`crate::FrameSource`].
pub trait MotionSource {
    /// Drain all motion samples that have arrived, in timestamp order.
    fn drain(&mut self, out: &mut Vec<ImuSample>);

    /// Whether the platform granted motion permission. `false` forces the
    /// orchestrator down to sensor tier 1 (vision only) rather than failing —
    /// spec.md §4 lists tier 1 as "fallback when motion permission is denied".
    fn is_available(&self) -> bool;
}

/// In-memory motion source for replay and tests.
#[derive(Debug, Clone, Default)]
pub struct ReplayMotionSource {
    samples: std::collections::VecDeque<ImuSample>,
    available: bool,
}

impl ReplayMotionSource {
    /// Build from a sorted sample list.
    #[must_use]
    pub fn new(mut samples: Vec<ImuSample>) -> Self {
        samples.sort_by_key(|s| s.timestamp);
        ReplayMotionSource {
            samples: samples.into(),
            available: true,
        }
    }

    /// A source that reports motion as unavailable — the tier-1 fallback path.
    #[must_use]
    pub fn denied() -> Self {
        ReplayMotionSource {
            samples: Default::default(),
            available: false,
        }
    }

    /// Drain only the samples at or before `until`, so replay can advance the
    /// IMU stream in lockstep with frames.
    pub fn drain_until(&mut self, until: Timestamp, out: &mut Vec<ImuSample>) {
        while let Some(front) = self.samples.front() {
            if front.timestamp <= until {
                out.push(self.samples.pop_front().expect("front exists"));
            } else {
                break;
            }
        }
    }

    /// Samples not yet drained.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.samples.len()
    }
}

impl MotionSource for ReplayMotionSource {
    fn drain(&mut self, out: &mut Vec<ImuSample>) {
        out.extend(self.samples.drain(..));
    }
    fn is_available(&self) -> bool {
        self.available
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn browser_units_convert_to_radians() {
        let e = MotionEvent::from_browser(
            0,
            1.0,
            Vec3::new(180.0, -90.0, 0.0),
            Vec3::new(0.0, 0.0, 9.81),
            None,
            0.016,
        );
        assert_relative_eq!(e.gyro.x, std::f64::consts::PI, epsilon = 1e-12);
        assert_relative_eq!(e.gyro.y, -std::f64::consts::FRAC_PI_2, epsilon = 1e-12);
    }

    #[test]
    fn lerp_hits_endpoints_and_midpoint() {
        let a = ImuSample::new(
            Timestamp::from_seconds(1.0),
            Vec3::zeros(),
            Vec3::new(0.0, 0.0, 10.0),
        );
        let b = ImuSample::new(
            Timestamp::from_seconds(2.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 12.0),
        );
        assert_relative_eq!(
            ImuSample::lerp(&a, &b, a.timestamp).accel,
            a.accel,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            ImuSample::lerp(&a, &b, b.timestamp).gyro,
            b.gyro,
            epsilon = 1e-12
        );
        let mid = ImuSample::lerp(&a, &b, Timestamp::from_seconds(1.5));
        assert_relative_eq!(mid.accel.z, 11.0, epsilon = 1e-12);
        assert_relative_eq!(mid.gyro.x, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn lerp_handles_coincident_timestamps() {
        let t = Timestamp::from_seconds(1.0);
        let a = ImuSample::new(t, Vec3::zeros(), Vec3::zeros());
        let b = ImuSample::new(t, Vec3::new(9.0, 9.0, 9.0), Vec3::zeros());
        // Must not produce NaN.
        assert!(ImuSample::lerp(&a, &b, t)
            .gyro
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn excitation_is_zero_when_stationary() {
        let s = ImuSample::new(Timestamp::ZERO, Vec3::zeros(), Vec3::new(0.0, 0.0, GRAVITY));
        assert_relative_eq!(s.excitation(GRAVITY), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn drain_until_respects_the_boundary() {
        let samples: Vec<ImuSample> = (0..10)
            .map(|i| {
                ImuSample::new(
                    Timestamp::from_seconds(i as f64 * 0.1),
                    Vec3::zeros(),
                    Vec3::zeros(),
                )
            })
            .collect();
        let mut src = ReplayMotionSource::new(samples);
        let mut out = Vec::new();
        src.drain_until(Timestamp::from_seconds(0.35), &mut out);
        assert_eq!(out.len(), 4); // 0.0, 0.1, 0.2, 0.3
        assert_eq!(src.remaining(), 6);
    }

    #[test]
    fn denied_source_reports_unavailable() {
        assert!(!ReplayMotionSource::denied().is_available());
    }
}
