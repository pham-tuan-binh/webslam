//! The dev viewer. spec.md §8: *"internal, ugly, indispensable."*
//!
//! Uses [rerun](https://rerun.io) for the native loop — Rust-native spatial
//! logging, loggable directly from `cargo test`, giving point clouds, camera
//! frusta, time-scrubbing and plots without writing a viewer.
//!
//! ## Why there is a trait here
//!
//! `rerun` is a large dependency and it is optional (`--features
//! rerun-viewer`). Rather than scatter `#[cfg]` through the harness, everything
//! logs to a [`ViewerSink`]; [`NullSink`] compiles to nothing. That also means
//! the *call sites* in the replay harness are identical whether or not anyone
//! is watching, so instrumentation cannot rot behind a feature flag.
//!
//! ## What it must show
//!
//! Straight from spec.md §8, and [`SessionFrame`] has a field for each:
//!
//! - camera feed with tracked features, coloured by state
//! - sparse landmark cloud and keyframe frusta in 3D
//! - estimated trajectory overlaid on ground truth
//! - **covariance ellipsoid on the current pose** — the differentiator, and we
//!   should be looking at it daily
//! - scale source badge and current scale variance
//! - per-stage frame timing: upload / pyramid / corners / flow / PnP
//! - tracking state machine
//! - from M5: pose-graph edges **including loop candidates rejected by
//!   geometric verification**, which is how the false-positive threshold gets
//!   tuned by eye rather than by guesswork

#![warn(missing_docs)]

use wslam_core::{GrayImage, Mat6, ScaleEstimate, Se3, Timestamp, TrackingState, Vec2, Vec3};

mod ellipsoid;
pub use ellipsoid::{uncertainty_ellipsoid, Ellipsoid};

#[cfg(feature = "rerun-viewer")]
mod rerun_sink;
#[cfg(feature = "rerun-viewer")]
pub use rerun_sink::RerunSink;

/// State of one tracked feature, for overlay colouring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureOverlayState {
    /// Detected this frame.
    New,
    /// Successfully tracked from the previous frame.
    Tracked,
    /// Tracked, but rejected by the pose solver's outlier test.
    Outlier,
    /// Tracked last frame, gone this one.
    Lost,
}

impl FeatureOverlayState {
    /// RGB for the overlay.
    ///
    /// Fixed here so the dev viewer and the web demo agree — a reviewer should
    /// never have to ask what a colour means, and two people describing the
    /// same screenshot should be describing the same thing.
    #[must_use]
    pub fn colour(self) -> [u8; 3] {
        match self {
            FeatureOverlayState::New => [94, 234, 212], // teal: fresh
            FeatureOverlayState::Tracked => [255, 255, 255], // white: nominal
            FeatureOverlayState::Outlier => [248, 113, 113], // red: rejected
            FeatureOverlayState::Lost => [107, 114, 128], // grey: gone
        }
    }
}

/// A 2D feature to overlay on the camera image.
#[derive(Debug, Clone, Copy)]
pub struct FeatureOverlay {
    /// Pixel position.
    pub px: Vec2,
    /// State, which selects the colour.
    pub state: FeatureOverlayState,
}

/// One pose-graph edge.
#[derive(Debug, Clone, Copy)]
pub struct GraphEdge {
    /// Source keyframe id.
    pub from: u64,
    /// Destination keyframe id.
    pub to: u64,
    /// `true` for a loop candidate, `false` for sequential odometry.
    pub is_loop: bool,
    /// Whether geometric verification accepted it.
    ///
    /// Rejected candidates are logged deliberately (spec.md §8): drawing them
    /// is how the verification threshold gets tuned by eye rather than by
    /// guesswork.
    pub accepted: bool,
    /// Place-recognition score.
    pub score: f64,
}

/// Per-stage frame timing, in milliseconds.
///
/// spec.md §8: *"Required to manage the WebGPU budget."*
#[derive(Debug, Clone, Copy, Default)]
pub struct StageTimingsView {
    /// Image upload to the GPU.
    pub upload_ms: f64,
    /// Pyramid construction.
    pub pyramid_ms: f64,
    /// Corner detection.
    pub corners_ms: f64,
    /// Optical flow.
    pub flow_ms: f64,
    /// Pose solve.
    pub pnp_ms: f64,
    /// Whole frame.
    pub total_ms: f64,
}

/// Everything worth looking at from one frame.
///
/// Fields are optional so a caller can log what it has: the replay harness
/// fills all of them, a unit test might fill two.
#[derive(Debug, Clone, Default)]
pub struct SessionFrame<'a> {
    /// Capture time; becomes the rerun timeline position.
    pub timestamp: Timestamp,
    /// Frame index; a second rerun timeline, for stepping frame by frame.
    pub frame_index: u64,
    /// The camera image.
    pub image: Option<&'a GrayImage>,
    /// Features to overlay on it.
    pub features: &'a [FeatureOverlay],
    /// Estimated pose, up to scale or metric depending on the scale source.
    pub pose: Option<Se3>,
    /// Pose covariance, `[translation, rotation]`.
    pub covariance: Option<Mat6>,
    /// Ground-truth pose, when a rig or dataset supplies one.
    pub ground_truth: Option<Se3>,
    /// Sparse landmark positions.
    pub landmarks: &'a [Vec3],
    /// Keyframe poses.
    pub keyframes: &'a [(u64, Se3)],
    /// Pose-graph edges, including rejected loop candidates.
    pub edges: &'a [GraphEdge],
    /// Current scale estimate and its provenance.
    pub scale: Option<ScaleEstimate>,
    /// Tracking state.
    pub state: Option<TrackingState>,
    /// Per-stage timings.
    pub timings: Option<StageTimingsView>,
}

/// Somewhere to log a session.
pub trait ViewerSink {
    /// Log one frame.
    fn log_frame(&mut self, frame: &SessionFrame<'_>);

    /// Log a scalar time series, for anything without a dedicated field —
    /// scale error, NEES, inlier ratio.
    fn log_scalar(&mut self, path: &str, timestamp: Timestamp, value: f64);

    /// Log a one-off note: a relocalization, a rejected closure, a config
    /// change.
    fn log_note(&mut self, path: &str, timestamp: Timestamp, text: &str);

    /// Flush and finish. Called once at the end of a session.
    fn finish(&mut self) {}
}

/// A sink that discards everything.
///
/// The default, so the harness's instrumentation costs nothing when nobody is
/// watching and cannot bit-rot behind a `#[cfg]`.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullSink;

impl ViewerSink for NullSink {
    fn log_frame(&mut self, _frame: &SessionFrame<'_>) {}
    fn log_scalar(&mut self, _path: &str, _timestamp: Timestamp, _value: f64) {}
    fn log_note(&mut self, _path: &str, _timestamp: Timestamp, _text: &str) {}
}

/// Build the configured sink.
///
/// With `--features rerun-viewer` and a path, writes an `.rrd`; with the
/// feature and no path, connects to a running viewer; without the feature,
/// returns [`NullSink`] and warns once.
///
/// It never fails: a harness run must not die because logging was unavailable.
#[must_use]
pub fn make_sink(rrd_path: Option<&std::path::Path>) -> Box<dyn ViewerSink> {
    #[cfg(feature = "rerun-viewer")]
    {
        match RerunSink::new("web-slam", rrd_path) {
            Ok(sink) => return Box::new(sink),
            Err(err) => log::warn!("rerun unavailable ({err}); logging disabled"),
        }
    }
    #[cfg(not(feature = "rerun-viewer"))]
    {
        if rrd_path.is_some() {
            log::warn!(
                "--rrd was given but this build lacks the rerun-viewer feature; \
                 rebuild with --features rerun-viewer"
            );
        }
    }
    Box::new(NullSink)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_sink_accepts_everything_without_panicking() {
        let mut sink = NullSink;
        sink.log_frame(&SessionFrame::default());
        sink.log_scalar("x", Timestamp::ZERO, f64::NAN);
        sink.log_note("y", Timestamp::ZERO, "hello");
        sink.finish();
    }

    #[test]
    fn overlay_colours_are_distinguishable() {
        // A reviewer scanning an overlay must be able to tell rejected features
        // from tracked ones at a glance, so the palette is pinned by a test.
        let states = [
            FeatureOverlayState::New,
            FeatureOverlayState::Tracked,
            FeatureOverlayState::Outlier,
            FeatureOverlayState::Lost,
        ];
        for (i, a) in states.iter().enumerate() {
            for b in &states[i + 1..] {
                let (ca, cb) = (a.colour(), b.colour());
                let distance: i32 = (0..3).map(|k| (ca[k] as i32 - cb[k] as i32).abs()).sum();
                assert!(
                    distance > 80,
                    "{a:?} and {b:?} are too close: {ca:?} vs {cb:?}"
                );
            }
        }
    }

    #[test]
    fn make_sink_never_fails() {
        let mut sink = make_sink(None);
        sink.log_frame(&SessionFrame::default());
        sink.finish();
    }
}
