//! # wslam-orientation — L1
//!
//! Gyro integration with accelerometer gravity correction: drift-free in roll
//! and pitch, slowly drifting in yaw. Three of six degrees of freedom, solved,
//! with no vision required (spec.md §4 L1).
//!
//! ```
//! use wslam_core::{ImuSample, Timestamp, Vec3};
//! use wslam_core::imu::GRAVITY;
//! use wslam_orientation::{OrientationConfig, OrientationFilter};
//!
//! let mut filter = OrientationFilter::new(OrientationConfig::default());
//! for i in 0..100 {
//!     filter.integrate(&ImuSample::new(
//!         Timestamp::from_seconds(i as f64 * 0.01),
//!         Vec3::new(0.0, 0.0, 0.3),           // turning 0.3 rad/s about body z
//!         Vec3::new(0.0, 0.0, GRAVITY),       // level, at rest translationally
//!     ));
//! }
//! assert!(filter.is_initialized());
//! // Roll and pitch are pinned by gravity; yaw has integrated 0.3 rad/s.
//! assert!((filter.gravity_body() - Vec3::new(0.0, 0.0, -1.0)).norm() < 1e-6);
//! ```
//!
//! ## Why a Kalman filter and not a complementary filter
//!
//! A complementary filter would produce the same attitude for a fraction of the
//! code. It would not produce a covariance, and the covariance is the product:
//! spec.md §6 L6 commits to validating it by NEES against ground truth, and
//! L6 publishes it with every pose. A tuned blend coefficient has nothing to
//! publish. So this is an error-state Kalman filter on SO(3) with the gyro bias
//! in the state — propagated with the gyro, corrected with gravity.
//!
//! ## The one thing to understand
//!
//! Gravity observes exactly two rotational degrees of freedom. The third — yaw
//! about world `+Z` — is unobservable to L1 forever, not merely
//! poorly-observed. Every design decision here follows from that:
//!
//! - the accelerometer Jacobian is `hat(v)` with `v = R^T e_z`, whose null space
//!   *is* the yaw direction, so the update cannot see yaw by construction;
//! - the Kalman gain's attitude rows are projected off `v` so that correlations
//!   in the covariance cannot leak a yaw correction out of a measurement that
//!   carries no yaw information;
//! - the yaw variance therefore starts at `pi^2/3` (uniform on the circle) and
//!   grows monotonically until [`OrientationFilter::correct_yaw`] — the L3 hook
//!   — supplies a heading.
//!
//! ## Frames, fixed once
//!
//! World is Z-up; the accelerometer reports specific force, so a device at rest
//! reads `+g` along world `+Z`. Attitude is `R_world_body` and errors are
//! right-multiplicative, matching `wslam_core::math`. See [`gravity`] for the
//! full statement and for the tilt/yaw decomposition everything else uses.
//!
//! ## Determinism
//!
//! There is no clock and no RNG in this crate. Time enters only as
//! `wslam_core::Timestamp` on the samples themselves (spec.md §6, *"Every
//! timestamp enters through the clock layer"*), and every intermediate is a pure
//! function of the sample stream, so a replay reproduces a live run bit for bit.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod config;
mod filter;
pub mod gravity;
mod history;
mod rates;

pub use config::OrientationConfig;
pub use filter::{FilterStats, OrientationFilter};
pub use history::HISTORY_CAPACITY;
