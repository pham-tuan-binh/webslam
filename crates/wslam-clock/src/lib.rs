//! # wslam-clock — L0, the unified timebase
//!
//! spec.md §4 L0: *"Everything above it is wrong if this is wrong."*
//!
//! ## The idea in one paragraph
//!
//! A browser hands us two timestamp streams and neither one is the time the
//! sample was taken. `DeviceMotion` events are generated on a regular hardware
//! cadence and only *delivered* with event-loop jitter, so their arrival stamp
//! is `true_time + delay` with `delay` non-negative and heavy-tailed. Video
//! frames carry `requestVideoFrameCallback`'s `mediaTime`, which rides the media
//! clock rather than the wall clock and is therefore much cleaner — but it lives
//! in a different epoch. So L0 does three things:
//!
//! 1. [`CadenceModel`] fits `t = slope * index + intercept` over the event
//!    **index**, robustly. Index is exact; the stamp is not. The fit recovers
//!    the hardware cadence and rejects the stalls.
//! 2. [`OffsetFilter`] carries the residual constant camera-IMU offset `td` as a
//!    filter state (Li & Mourikis, IJRR 2014; Qin & Shen, IROS 2018), and
//!    **suspends** estimation under the degenerate motions those papers
//!    identify rather than integrating quietly through them.
//! 3. [`FittedTimeBase`] composes the two and implements
//!    [`wslam_core::TimeBase`], which is the only door time enters the pipeline
//!    through (spec.md §6, "Determinism is a prerequisite").
//!
//! [`cross_correlate_lag`] is the offline half: on the turntable-plus-strobe rig
//! we cross-correlate gyro angular rate against image-derived rotation rate and
//! *the peak lag is the offset* (spec.md §6 L0). That measurement is what feeds
//! [`FittedTimeBase::observe_offset`].
//!
//! ## Sign convention, fixed once
//!
//! `td` is positive when **camera stamps lag IMU stamps**: for one physical
//! instant the camera reads later than the IMU, so the correction subtracts `td`
//! from camera stamps. This matches [`wslam_core::TimeBase::camera_imu_offset`]
//! and the `a`/`b` argument order of [`cross_correlate_lag`], which returns "how
//! far `b` lags `a`". Feed gyro rate as `a` and image rate as `b` and the
//! returned lag is `td` with no sign gymnastics.
//!
//! ## What the tests in this crate do and do not prove
//!
//! spec.md §6 L0 sets the bar at beating the 30 ms camera-IMU offset Huai et al.
//! measured on real phones *with full native API access* (arXiv:2001.00470), and
//! asks for the **variance** of the residual offset because "the jitter is the
//! thing we claim to fix, so report the distribution, not a mean".
//!
//! `timebase::tests::beats_the_thirty_millisecond_bar_on_synthetic_jitter`
//! measures exactly that quantity and passes comfortably. **That is not the
//! claim.** It is a test that the estimator does what the model says on a jitter
//! profile we invented; it cannot be evidence about a jitter profile we did not.
//! The real number comes from the rig — phone on the turntable viewing the
//! strobe, per device, never aggregated (spec.md §6 L0, "Report per device").
//! Until that runs, this crate has demonstrated correctness and nothing about
//! iOS Safari.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod cadence;
pub mod correlate;
pub mod offset;
pub mod timebase;

#[cfg(test)]
mod synth;

pub use cadence::{CadenceConfig, CadenceModel};
pub use correlate::{cross_correlate_lag, LagEstimate};
pub use offset::OffsetFilter;
pub use timebase::{ClockConfig, FittedTimeBase};

/// Strictly positive and finite.
///
/// Every denominator in this crate is guarded with this. Spelled as a function
/// rather than inline because the two obvious inline forms are both wrong or
/// disallowed: `x <= 0.0` silently admits NaN (a NaN denominator produces a NaN
/// estimate that survives for a long time before anyone notices), and the
/// NaN-safe `!(x > 0.0)` trips `clippy::neg_cmp_op_on_partial_ord`.
#[inline]
pub(crate) fn is_positive_finite(x: f64) -> bool {
    x.is_finite() && x > 0.0
}
