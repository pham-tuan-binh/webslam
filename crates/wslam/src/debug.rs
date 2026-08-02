//! The debug surface.
//!
//! spec.md §3: namespaced, tree-shakeable, and **explicitly unstable** —
//! versioned separately from the core API so the viewer can move fast without
//! pinning us. It is also first-class rather than an afterthought, because our
//! own viewer and demo are its first consumers.
//!
//! Every accessor borrows rather than copies where it can. A viewer polling at
//! 60 Hz should not make the allocator the bottleneck.

use wslam_core::{Se3, Vec2, Vec3};
use wslam_track::{FeatureState, StageTimings};

use crate::WebSlam;

/// One tracked 2D feature, with the state that selects its overlay colour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DebugFeature {
    /// Stable identity across frames.
    pub id: u64,
    /// Position in full-resolution pixels.
    pub px: Vec2,
    /// Lifecycle state.
    pub state: FeatureState,
    /// Frames survived.
    pub age: u32,
}

/// A keyframe, for drawing frusta.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DebugKeyframe {
    /// Keyframe id.
    pub id: u64,
    /// Capture time, milliseconds in the unified timebase.
    pub timestamp_ms: f64,
    /// `T_world_camera`.
    pub pose: Se3,
}

/// One pose-graph edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DebugPoseGraphEdge {
    /// Source keyframe id.
    pub from: u64,
    /// Destination keyframe id.
    pub to: u64,
    /// `true` for a loop candidate, `false` for sequential odometry.
    pub is_loop: bool,
    /// Whether geometric verification accepted it.
    pub accepted: bool,
    /// Place-recognition score; 1.0 for odometry edges.
    pub score: f64,
}

/// A borrowed view of everything the viewer needs.
///
/// Cheap to construct — the expensive accessors allocate only when called, so a
/// consumer that wants only `timings()` pays for only `timings()`.
pub struct DebugSnapshot<'a> {
    slam: &'a WebSlam,
}

impl<'a> DebugSnapshot<'a> {
    pub(crate) fn new(slam: &'a WebSlam) -> Self {
        DebugSnapshot { slam }
    }

    /// Sparse landmark positions from the active local map.
    ///
    /// Up to scale unless the session has a metric anchor — the same units as
    /// `pose.position`.
    #[must_use]
    pub fn landmarks(&self) -> Vec<Vec3> {
        self.slam
            .tracker_ref()
            .local_map()
            .landmarks()
            .iter()
            .map(|l| l.position)
            .collect()
    }

    /// Landmarks packed as `xyz` floats, ready to upload as a vertex buffer.
    #[must_use]
    pub fn landmarks_packed(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.slam.tracker_ref().local_map().len() * 3);
        for l in self.slam.tracker_ref().local_map().landmarks() {
            out.push(l.position.x as f32);
            out.push(l.position.y as f32);
            out.push(l.position.z as f32);
        }
        out
    }

    /// Stored keyframes, oldest first.
    #[must_use]
    pub fn keyframes(&self) -> Vec<DebugKeyframe> {
        match self.slam.map_ref() {
            Some(db) => {
                let mut out: Vec<DebugKeyframe> = db
                    .keyframes()
                    .map(|kf| DebugKeyframe {
                        id: kf.id.0,
                        timestamp_ms: kf.timestamp.millis(),
                        pose: kf.pose,
                    })
                    .collect();
                out.sort_by_key(|k| k.id);
                out
            }
            None => Vec::new(),
        }
    }

    /// The estimated trajectory so far.
    #[must_use]
    pub fn trajectory(&self) -> &[Vec3] {
        self.slam.trajectory_ref()
    }

    /// Trajectory packed as `xyz` floats.
    #[must_use]
    pub fn trajectory_packed(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.slam.trajectory_ref().len() * 3);
        for p in self.slam.trajectory_ref() {
            out.push(p.x as f32);
            out.push(p.y as f32);
            out.push(p.z as f32);
        }
        out
    }

    /// Current 2D features, with per-feature state for overlay colouring.
    #[must_use]
    pub fn features(&self) -> Vec<DebugFeature> {
        self.slam
            .tracker_ref()
            .features()
            .iter()
            .map(|f| DebugFeature {
                id: f.id,
                px: f.px,
                state: f.state,
                age: f.age,
            })
            .collect()
    }

    /// Pose-graph edges, **including loop candidates that were rejected**.
    ///
    /// spec.md §8 requires the rejections specifically: drawing them is how the
    /// verification threshold gets tuned by eye rather than by guesswork. A
    /// graph view that shows only accepted closures cannot answer "was that
    /// threshold too tight?".
    #[must_use]
    pub fn pose_graph(&self) -> Vec<DebugPoseGraphEdge> {
        let mut out: Vec<DebugPoseGraphEdge> = self
            .slam
            .graph_ref()
            .edges()
            .iter()
            .map(|e| DebugPoseGraphEdge {
                from: e.from.0,
                to: e.to.0,
                // Sequential ids adjacent in the graph are odometry; anything
                // else got there through place recognition.
                is_loop: e.to.0.abs_diff(e.from.0) != 1,
                accepted: true,
                score: 1.0,
            })
            .collect();
        out.extend_from_slice(self.slam.rejected_loops_ref());
        out
    }

    /// Per-stage timing for the most recent frame.
    ///
    /// The only place in the pipeline that consults a wall clock, and it is
    /// confined to reporting (spec.md §6, and `wslam_core::time`).
    #[must_use]
    pub fn timings(&self) -> StageTimings {
        *self.slam.tracker_ref().timings()
    }

    /// Current intrinsics, refined by L2 if it ran.
    #[must_use]
    pub fn intrinsics(&self) -> wslam_core::CameraIntrinsics {
        self.slam.intrinsics()
    }

    /// Map size, for the memory-growth metric spec.md §6 L4 asks for.
    #[must_use]
    pub fn map_memory_bytes(&self) -> usize {
        self.slam.map_ref().map_or(0, |db| db.memory_bytes())
    }

    /// Keyframe count.
    #[must_use]
    pub fn keyframe_count(&self) -> usize {
        self.slam.map_ref().map_or(0, |db| db.keyframe_count())
    }

    /// Configured versus effective sensor tier. They differ when motion
    /// permission was denied.
    #[must_use]
    pub fn tiers(&self) -> (crate::SensorTier, crate::SensorTier) {
        (self.slam.config_ref().tier, self.slam.effective_tier())
    }

    /// Version of this surface. Distinct from the stable API version, and
    /// deliberately so.
    #[must_use]
    pub fn version(&self) -> &'static str {
        wslam_core::DEBUG_API_VERSION
    }
}

impl std::fmt::Debug for DebugSnapshot<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DebugSnapshot")
            .field("landmarks", &self.slam.tracker_ref().local_map().len())
            .field("keyframes", &self.keyframe_count())
            .field("features", &self.slam.tracker_ref().features().len())
            .field("map_bytes", &self.map_memory_bytes())
            .finish()
    }
}

#[cfg(test)]
mod tests {

    use crate::{SlamConfig, WebSlam};

    #[test]
    fn the_debug_version_is_separate_from_the_public_one() {
        // spec.md §3: "versioned separately from the core API so the viewer can
        // move fast without pinning us."
        let slam = WebSlam::new(SlamConfig::new(320, 240)).unwrap();
        assert_ne!(slam.debug().version(), wslam_core::PUBLIC_API_VERSION);
        assert!(slam.debug().version().contains("unstable"));
    }

    #[test]
    fn packed_accessors_agree_with_their_structured_forms() {
        let slam = WebSlam::new(SlamConfig::new(320, 240)).unwrap();
        let debug = slam.debug();
        assert_eq!(debug.landmarks_packed().len(), debug.landmarks().len() * 3);
        assert_eq!(
            debug.trajectory_packed().len(),
            debug.trajectory().len() * 3
        );
    }

    #[test]
    fn an_empty_session_reports_zeroes_rather_than_panicking() {
        // A viewer should never have to special-case "before the first frame".
        let slam = WebSlam::new(SlamConfig::new(320, 240)).unwrap();
        let debug = slam.debug();
        assert!(debug.landmarks().is_empty());
        assert!(debug.keyframes().is_empty());
        assert!(debug.features().is_empty());
        assert!(debug.pose_graph().is_empty());
        assert_eq!(debug.keyframe_count(), 0);
        assert_eq!(debug.timings().total_ms, 0.0);
    }

    #[test]
    fn map_memory_is_zero_when_mapping_is_disabled() {
        let cfg = SlamConfig {
            map: crate::MapConfig {
                enabled: false,
                ..Default::default()
            },
            ..SlamConfig::new(320, 240)
        };
        let slam = WebSlam::new(cfg).unwrap();
        assert_eq!(slam.debug().map_memory_bytes(), 0);
        assert!(slam.debug().keyframes().is_empty());
    }

    #[test]
    fn tiers_report_configured_and_effective_separately() {
        let slam = WebSlam::new(SlamConfig::new(320, 240)).unwrap();
        let (configured, effective) = slam.debug().tiers();
        assert_eq!(configured, crate::SensorTier::VisionOrientation);
        // Before any frame the effective tier has not been demoted yet.
        assert_eq!(effective, crate::SensorTier::VisionOrientation);
    }
}
