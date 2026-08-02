//! # wslam-core
//!
//! Shared vocabulary for every layer of web-slam. This crate depends on nothing
//! else in the workspace (spec.md §7, "Dependency direction is one-way and
//! enforced"), so it is the one place where cross-layer types may live.
//!
//! Three rules from spec.md §6 ("Determinism is a prerequisite") are structural
//! here rather than aspirational:
//!
//! 1. **No wall clock in the pipeline.** There is no `Instant::now` or
//!    `Date.now` reachable from any pipeline type. Time enters exclusively
//!    through [`TimeBase`], which maps sensor-native stamps into the unified
//!    timebase. [`HostClock`] exists for profiling only and is documented as
//!    off-pipeline.
//! 2. **Every RNG is seeded and the seed is logged.** [`DeterministicRng`] is
//!    the only randomness source; it cannot be constructed without a seed.
//! 3. **The frame source is an interface.** [`FrameSource`] has live and replay
//!    implementations; the same binary runs both.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod camera;
pub mod covariance;
pub mod error;
pub mod frame;
pub mod imu;
pub mod math;
pub mod pose;
pub mod rng;
pub mod stats;
pub mod time;
pub mod window;

pub use camera::{CameraIntrinsics, CameraModel, RadialTangential};
pub use error::{Error, Result};
pub use frame::{Frame, FrameId, FrameSource, GrayImage, ReplayFrameSource};
pub use imu::{ImuSample, MotionEvent, MotionSource};
pub use math::{Mat3, Mat4, Mat6, Quat, Scalar, Se3, Sim3, So3, Vec2, Vec3, Vec6};
pub use pose::{LimitedReason, Pose, ScaleEstimate, ScaleKind, TrackingState};
pub use rng::DeterministicRng;
pub use time::{HostClock, Nanos, PassthroughTimeBase, TimeBase, Timestamp};
pub use window::{StateWindow, WindowSample};

/// Version of the on-disk / on-wire map format. Bumped on any breaking change
/// to the serialised map; readers reject formats they do not understand.
pub const MAP_FORMAT_VERSION: u16 = 1;

/// Semantic version of the *public* API surface described in spec.md §3.
///
/// The debug surface (`slam.debug.*`) is versioned separately and explicitly
/// unstable; see [`DEBUG_API_VERSION`].
pub const PUBLIC_API_VERSION: &str = "0.1.0";

/// Version of the deliberately unstable debug surface (spec.md §3).
pub const DEBUG_API_VERSION: &str = "0.1.0-unstable";
