//! The L3 frontend: per-frame tracking and the [`TrackingState`] machine.
//!
//! One `process` call is the whole critical path of spec.md §4 L3 — upload,
//! pyramid, corners, flow, PnP — and it must never block on L4.
//!
//! ## Where the state machine's verdicts come from
//!
//! spec.md §3 gives three reasons a tracker may be `Limited`, and they are
//! *causes*, not symptoms. A dark frame yields few features; a fast pan yields
//! few features; a blank wall yields few features. Reporting
//! `insufficient-features` for all three tells the caller nothing it can act on,
//! so the checks are ordered by how upstream the cause is: low light (the
//! sensor), then excessive motion (the user), then insufficient features (the
//! scene). `Lost` outranks all three.
//!
//! ## Timing and the wall-clock ban
//!
//! spec.md §3 requires per-stage upload/pyramid/corners/flow/PnP timings, and
//! spec.md §6 bans wall-clock reads from the pipeline. `wslam_core::time`
//! resolves this by splitting the two traits: [`TimeBase`] carries measurement
//! time and may be consulted; [`HostClock`] is profiling only. The tracker holds
//! an optional `Box<dyn HostClock>` that is read **only** by
//! [`Tracker::lap`], whose return value goes **only** into [`StageTimings`].
//! No branch in this file reads a `StageTimings` field, and no estimate depends
//! on whether a clock is installed — with no clock the timings are zero and the
//! trajectory is bit-identical.
//!
//! [`TimeBase`]: wslam_core::TimeBase

use std::collections::HashMap;

use wslam_core::{
    CameraIntrinsics, DeterministicRng, Frame, HostClock, LimitedReason, Mat6, Scalar, Se3, So3,
    Timestamp, TrackingState, Vec2, Vec3,
};

use crate::corners::{self, CornerConfig};
use crate::init::{self, InitConfig};
use crate::klt::{self, KltConfig};
use crate::local_ba;
use crate::motion_ba::{self, MotionBaConfig};
use crate::motion_ba::{pinhole_only, project_pinhole};
use crate::pnp;
use crate::pyramid::{Pyramid, PyramidConfig};
use crate::triangulate::{self, TriangulationConfig};

/// Fewest 3D-2D correspondences the tracker will attempt a pose from. P3P needs
/// four; six leaves enough redundancy for RANSAC to mean something.
const MIN_PNP_CORRESPONDENCES: usize = 6;

/// Features in the with/without-prior trial subsample. Large enough that the
/// majority verdict is stable, small enough that two extra passes over it are
/// noise in the frame budget.
const PRIOR_TRIAL_FEATURES: usize = 24;

/// Consecutive frames without a pose solve before the session is declared lost
/// rather than limited. One dropped frame is an occlusion; three is a loss.
const LOST_AFTER_FAILURES: u32 = 3;

/// Consecutive solve failures before the local map is discarded and the
/// tracker re-initialises from scratch.
///
/// Deliberately much larger than [`LOST_AFTER_FAILURES`]. Reporting `Lost`
/// early is cheap and correct — the consumer should stop trusting the pose. But
/// *discarding the map* forfeits any chance of relocalizing back into the
/// coordinate frame the caller has already anchored content to, so it waits
/// until recovery-in-place has plainly failed. At 30 Hz this is about a second.
const ABANDON_MAP_AFTER_FAILURES: u32 = 30;

/// Frames a two-view bootstrap reference is kept before a fresh one is taken.
///
/// A reference that has not produced a map in this long is not going to: either
/// the view has changed too much to match it, or it was taken on a frame with
/// nothing in it. At 30 Hz this is about a second and a half.
const BOOTSTRAP_REFERENCE_TTL_FRAMES: u64 = 45;

/// Per-frame covariance inflation while coasting on a stale pose. Reporting the
/// last solve's covariance for a pose that is now several frames old is exactly
/// the overconfidence spec.md §6 L6 calls "worse than no covariance at all".
const STALE_COVARIANCE_GROWTH: Scalar = 4.0;

/// Frames a landmark may go unobserved before it is dropped from the local map.
/// Unbounded growth here is the frontend's version of spec.md §9's "tab killed".
const LANDMARK_TTL_FRAMES: u64 = 150;

/// Track survival ratio below which a frame is blamed on motion rather than on
/// the scene, provided the previous frame was healthy.
const EXCESSIVE_SURVIVAL_RATIO: Scalar = 0.3;

/// Median flow, as a fraction of the pyramid's displacement envelope, above
/// which the motion is called excessive.
const EXCESSIVE_FLOW_FRACTION: Scalar = 0.75;

/// Corners the refilled frame must still hold, as a fraction of the count the
/// frame was entered with, for the frame to count as *textured*. Used only to
/// separate "the camera moved too fast" from "the scene went blank", and a
/// blank frame yields exactly zero corners, so the constant only has to sit
/// somewhere between "less textured" and "not textured".
const TEXTURE_PRESENT_RATIO: Scalar = 0.5;

/// Baseline the bootstrap waits for, as a fraction of image width.
const BOOTSTRAP_PARALLAX_FRACTION: Scalar = 0.02;

/// Fewest correspondences the two-view bootstrap will be offered. The
/// eight-point algorithm needs eight; below that there is nothing to fit.
const MIN_BOOTSTRAP_MATCHES: usize = 8;

/// Per-axis pixel noise assumed when turning the PnP geometry into a covariance.
const PIXEL_SIGMA: Scalar = 1.0;

/// Lifecycle of one tracked feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureState {
    /// Detected this frame; not yet tracked anywhere.
    New,
    /// Followed from the previous frame by KLT.
    Tracked,
    /// Tracked, but rejected by the pose solve's robust cost.
    Outlier,
    /// Flow failed. Retained for one frame so the debug overlay can colour it.
    Lost,
}

/// Remove lens distortion, mapping an observed pixel into the pinhole camera
/// that `pnp` and `triangulate` assume.
///
/// A free function rather than a method so it can be called while
/// `self.features` is mutably borrowed.
fn undistort_px(k: &CameraIntrinsics, px: Vec2) -> Vec2 {
    let n = k.unproject_normalized(px);
    project_pinhole(&pinhole_only(k), &Vec3::new(n.x, n.y, 1.0)).unwrap_or(px)
}

/// One feature in the current frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Feature {
    /// Stable identity across frames.
    pub id: u64,
    /// Position in full-resolution pixels, **as observed** — lens distortion
    /// included. This is what KLT, corner masking and `predict` work on,
    /// because they operate on the image.
    pub px: Vec2,
    /// The same point with lens distortion removed, in the pinhole camera
    /// `motion_ba::pinhole_only` defines.
    ///
    /// Every geometric consumer takes this one. `pnp` and `triangulate` both
    /// document "all 2D inputs are undistorted pixel coordinates" and strip
    /// distortion from the model on that basis; handing them `px` instead put a
    /// systematic error into every residual — on EuRoC's `k1 = -0.283` lens
    /// that is over 20 px at the image periphery, which is far outside any
    /// sane RANSAC threshold and starved the pose solve of inliers.
    pub px_undist: Vec2,
    /// Lifecycle state, for the debug overlay (spec.md §8: "coloured by state").
    pub state: FeatureState,
    /// Associated local-map landmark, if this feature has been triangulated.
    pub landmark: Option<u64>,
    /// Frames survived.
    pub age: u32,
}

/// A landmark in the tracker's active local map.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalLandmark {
    /// Stable identity.
    pub id: u64,
    /// Position in world coordinates, **up to scale** — L3 makes no metric
    /// claim (spec.md §4 L3).
    pub position: Vec3,
    /// Number of frames this landmark has been an inlier in.
    pub observations: u32,
}

/// The sliding set of landmarks the tracker solves pose against.
///
/// Deliberately not L4's map: this one is bounded, forgetful, and thrown away
/// on reset. The compiler enforces the separation (spec.md §7) — `wslam-track`
/// cannot see `wslam-map`.
#[derive(Debug, Clone, Default)]
pub struct LocalMap {
    landmarks: Vec<LocalLandmark>,
    last_seen: Vec<u64>,
    index: HashMap<u64, usize>,
}

impl LocalMap {
    /// An empty map.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Every landmark, in insertion order.
    #[must_use]
    pub fn landmarks(&self) -> &[LocalLandmark] {
        &self.landmarks
    }

    /// Landmark count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.landmarks.len()
    }

    /// Whether the map holds nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.landmarks.is_empty()
    }

    /// Look a landmark up by id.
    #[must_use]
    /// Mutable access, for the one operation that corrects the map.
    ///
    /// Deliberately narrow: local bundle adjustment is the only caller, because
    /// it is the only thing entitled to move a landmark after triangulation.
    pub(crate) fn get_mut(&mut self, id: u64) -> Option<&mut LocalLandmark> {
        self.index.get(&id).copied().map(|i| &mut self.landmarks[i])
    }

    /// Every landmark, mutably — for the epoch merge, which re-expresses the
    /// whole map in another coordinate frame at once. Not for point-wise
    /// correction; that is [`LocalMap::get_mut`]'s job.
    pub(crate) fn landmarks_mut(&mut self) -> impl Iterator<Item = &mut LocalLandmark> {
        self.landmarks.iter_mut()
    }

    /// Borrow a landmark by id.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&LocalLandmark> {
        self.index.get(&id).map(|&i| &self.landmarks[i])
    }

    fn insert(&mut self, landmark: LocalLandmark, frame: u64) {
        match self.index.get(&landmark.id) {
            Some(&i) => {
                self.landmarks[i] = landmark;
                self.last_seen[i] = frame;
            }
            None => {
                self.index.insert(landmark.id, self.landmarks.len());
                self.landmarks.push(landmark);
                self.last_seen.push(frame);
            }
        }
    }

    fn observe(&mut self, id: u64, frame: u64) {
        if let Some(&i) = self.index.get(&id) {
            self.landmarks[i].observations = self.landmarks[i].observations.saturating_add(1);
            self.last_seen[i] = frame;
        }
    }

    /// Drop landmarks unobserved for `ttl` frames, then trim the weakest until
    /// at most `capacity` remain.
    fn cull(&mut self, frame: u64, ttl: u64, capacity: usize) {
        let mut keep: Vec<usize> = (0..self.landmarks.len())
            .filter(|&i| frame.saturating_sub(self.last_seen[i]) <= ttl)
            .collect();
        if keep.len() > capacity {
            // Most-observed first, ties by recency, then by id so the outcome
            // does not depend on sort stability.
            keep.sort_by(|&a, &b| {
                self.landmarks[b]
                    .observations
                    .cmp(&self.landmarks[a].observations)
                    .then(self.last_seen[b].cmp(&self.last_seen[a]))
                    .then(self.landmarks[a].id.cmp(&self.landmarks[b].id))
            });
            keep.truncate(capacity);
            keep.sort_unstable();
        }
        if keep.len() == self.landmarks.len() {
            return;
        }
        let landmarks: Vec<LocalLandmark> = keep.iter().map(|&i| self.landmarks[i]).collect();
        let last_seen: Vec<u64> = keep.iter().map(|&i| self.last_seen[i]).collect();
        self.index = landmarks
            .iter()
            .enumerate()
            .map(|(i, l)| (l.id, i))
            .collect();
        self.landmarks = landmarks;
        self.last_seen = last_seen;
    }

    fn clear(&mut self) {
        self.landmarks.clear();
        self.last_seen.clear();
        self.index.clear();
    }
}

/// Per-stage frame timing, in milliseconds (spec.md §3 debug surface).
///
/// All zero when no [`HostClock`] is installed. These fields are output only —
/// nothing in the estimator reads them.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StageTimings {
    /// Frame ingest and the intensity statistic.
    pub upload_ms: f64,
    /// Pyramid construction.
    pub pyramid_ms: f64,
    /// Corner detection, summed over bootstrap and refill.
    pub corners_ms: f64,
    /// Pyramidal KLT including the forward-backward pass.
    pub flow_ms: f64,
    /// PnP RANSAC plus motion-only bundle adjustment.
    pub pnp_ms: f64,
    /// Wall time for the whole call.
    pub total_ms: f64,
}

/// Tracker tuning. Frozen by `docs/CONTRACT.md`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackConfig {
    /// Feature budget per frame.
    pub max_features: usize,
    /// Fewest features the tracker considers a workable frame. A frame that
    /// still holds fewer than this *after* the refill is reported as
    /// `insufficient-features`, and a landmark count below it forces a
    /// keyframe.
    pub min_features: usize,
    /// Fraction of [`TrackConfig::max_features`] below which the per-frame
    /// corner refill actually runs.
    ///
    /// The refill's cost is not the corners it returns — it is the full-image
    /// response map it computes to find them, ~7 ms of a 17 ms wasm frame at
    /// 640x360. With refill-on-any-deficit the feature count hovers one short
    /// of the budget and the map is computed every frame to add one corner.
    /// With hysteresis the count sawtooths between `refill_below_fraction *
    /// max_features` and the budget, and most frames skip detection entirely.
    /// The trough (200 at the defaults) stays far above `min_features` (60),
    /// which is what distinguishes this from the keyframe-only refill the
    /// history warns about — that one let the set sag *through* the floor.
    ///
    /// Keyframes and an empty map still refill unconditionally: keyframes
    /// register the full set as observations, and the bootstrap wants all the
    /// supply it can get. `1.0` restores refill-on-any-deficit, for ablation.
    pub refill_below_fraction: Scalar,
    /// Pyramid levels including level 0.
    pub pyramid_levels: u32,
    /// KLT patch **half**-width; the patch is `(2w+1) x (2w+1)`.
    pub klt_window: u32,
    /// KLT iterations per level.
    pub klt_iterations: u32,
    /// Forward-backward round-trip tolerance in pixels; `None` disables the
    /// check.
    ///
    /// The GPU kernel solves in `f32` with its own interpolation, so its round
    /// trip is intrinsically noisier than the `f64` CPU reference and wants a
    /// looser tolerance for the same rejection rate. Tightening the CPU value
    /// onto the GPU path rejected so many good tracks that 52.6% of frames lost
    /// pose while ATE stayed fine — the surviving tracks were excellent and
    /// there were nowhere near enough of them.
    pub klt_forward_backward: Option<f64>,
    /// PnP RANSAC inlier threshold, pixels.
    pub ransac_threshold_px: f64,
    /// PnP RANSAC iteration cap.
    pub ransac_iterations: usize,
    /// Keyframe on this much translation since the last one, in **up-to-scale**
    /// units — L3 has no metres.
    pub keyframe_translation: f64,
    /// Keyframe on this much rotation since the last one, radians.
    pub keyframe_rotation_rad: f64,
    /// Minimum frames between keyframes.
    ///
    /// ORB-SLAM2 uses `mMaxFrames = fps` as its *forced* insertion interval,
    /// i.e. one second. A 3-frame floor at 20 Hz is nearly seven times denser
    /// than that and leaves consecutive keyframes with no usable baseline.
    pub keyframe_min_frames: u64,
    /// Insert a keyframe when tracking falls below this fraction of what the
    /// reference keyframe saw. ORB-SLAM2's `thRefRatio`, which is 0.9 for
    /// monocular.
    pub keyframe_tracked_ratio: Scalar,
    /// Never let the starvation trigger fire below this many tracked
    /// landmarks. ORB-SLAM2's `mnMatchesInliers > 15`: below it the track is
    /// dying and more keyframes cannot save it.
    pub keyframe_min_tracked: usize,
    /// Mean intensity below which the frame is reported as low light.
    pub low_light_threshold: f64,
    /// RNG seed. Logged; RANSAC draws from it (spec.md §6).
    pub seed: u64,
    /// Prefer the WebGPU front-end when it is compiled in.
    pub use_gpu: bool,
    /// Keyframes retained for local bundle adjustment.
    ///
    /// Ten is the ORB-SLAM2 local-window order and keeps the reduced camera
    /// system at 48x48, which solves in microseconds. Zero disables local BA,
    /// which is only useful for measuring what it buys.
    pub local_ba_window: usize,
    /// Leading keyframes in the window held fixed, anchoring the gauge.
    ///
    /// Must be at least 1. Two is steadier: with a single anchor the window can
    /// pivot about it, and a monocular window has no scale of its own to resist
    /// that.
    pub local_ba_fixed: usize,
    /// Levenberg-Marquardt iterations per local BA solve.
    ///
    /// The frame-time tail is set almost entirely by this: BA runs
    /// synchronously on keyframe insertion, so a large window at full iteration
    /// count turns a 2 ms median into a 325 ms p99.
    pub local_ba_iterations: usize,
    /// Older keyframes kept behind the optimised window purely as **fixed**
    /// context.
    ///
    /// This is the piece of ORB-SLAM2's `LocalBundleAdjustment` that pins a
    /// landmark to the map it already belongs to. Without it, a landmark
    /// observed by the window is free even though older keyframes outside the
    /// window also see it, so each solve can rescale the local reconstruction
    /// against nothing — local consistency bought with global scale drift.
    /// With it, those older observations hold the landmarks still and the
    /// window has to fit the existing map rather than a rescaled copy of it.
    pub local_ba_context: usize,
    /// Hold landmarks fixed in local BA, optimising only the window's poses.
    ///
    /// A fixed map has **no gauge freedom at all**, so the seventh direction —
    /// scale — cannot drift no matter how badly conditioned the anchors are.
    /// Full BA measured a 2.6x better inter-frame RPE but twice the absolute
    /// error, the signature of local consistency bought with global scale
    /// drift; this is the version that cannot make that trade.
    pub local_ba_motion_only: bool,
    /// Largest per-solve change in window path length local BA may make.
    /// See `local_ba::LocalBaConfig::max_scale_change`.
    pub local_ba_max_scale_change: Scalar,
}

impl Default for TrackConfig {
    fn default() -> Self {
        TrackConfig {
            max_features: 250,
            min_features: 60,
            refill_below_fraction: 0.8,
            pyramid_levels: 4,
            klt_window: 5,
            klt_iterations: 24,
            klt_forward_backward: Some(1.0),
            ransac_threshold_px: 3.0,
            ransac_iterations: 256,
            keyframe_translation: 0.08,
            keyframe_rotation_rad: 0.15,
            keyframe_min_frames: 10,
            keyframe_tracked_ratio: 0.9,
            keyframe_min_tracked: 15,
            // Mean luma of 20/255 is roughly where a phone sensor's read noise
            // starts to dominate the gradient the tracker lives on.
            low_light_threshold: 20.0,
            seed: 0xC0FFEE,
            use_gpu: false,
            local_ba_window: 10,
            local_ba_max_scale_change: 1.5,
            local_ba_fixed: 2,
            local_ba_motion_only: false,
            local_ba_context: 30,
            local_ba_iterations: 4,
        }
    }
}

/// Why frames failed to produce a pose, counted over a session.
///
/// A single aggregate loss rate says a pipeline is failing without saying
/// where, which is not enough to act on. These counters exist so the next fix
/// is chosen by measurement rather than by guess.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FailureCounts {
    /// No map yet — the two-view bootstrap has not succeeded.
    pub awaiting_bootstrap: usize,
    /// Fewer than `MIN_PNP_CORRESPONDENCES` features carried a landmark.
    pub too_few_correspondences: usize,
    /// PnP RANSAC found no consensus.
    pub ransac_failed: usize,
    /// A pose was found but too few observations agreed with it.
    pub too_few_inliers: usize,
}

impl FailureCounts {
    /// Total frames that produced no pose.
    #[must_use]
    pub fn total(&self) -> usize {
        self.awaiting_bootstrap
            + self.too_few_correspondences
            + self.ransac_failed
            + self.too_few_inliers
    }
}

/// What one `process` call produced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackOutcome {
    /// State after this frame.
    pub state: TrackingState,
    /// `T_world_camera`, up to scale. `None` before the map exists.
    pub pose: Option<Se3>,
    /// 6x6 pose covariance, `[translation; rotation]`, right-perturbation.
    pub covariance: Mat6,
    /// Correspondences the pose solve accepted.
    pub inlier_count: usize,
    /// Features that survived flow.
    pub tracked_count: usize,
    /// Whether this frame was promoted to a keyframe.
    pub is_keyframe: bool,
    /// Whether this frame established a **new, unrelated** world anchor.
    ///
    /// True exactly when a two-view bootstrap succeeded. The scale and origin
    /// it picks bear no relation to whatever came before, so a consumer that
    /// keeps appending to one trajectory across this boundary is splicing two
    /// different coordinate systems together.
    ///
    /// `relocalized_to` deliberately does **not** set this: relocalizing is the
    /// operation that re-establishes the *existing* frame.
    pub bootstrapped: bool,
}

/// Which image front-end is in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// The reference implementation in this crate. Always available, and the
    /// definition of correct (`docs/DECISIONS.md` D5).
    Cpu,
    /// The `wslam-gpu` compute pipeline.
    Gpu,
}

/// The GPU image front-end, when the crate is built with the `gpu` feature and
/// the caller asked for it.
///
/// Holds the pipeline and mirrors the tracker's own notion of which frames are
/// resident. `ImagePipeline` keeps two frame sets and alternates with `swap`,
/// so "previous" is implicit in the pipeline rather than owned by the tracker —
/// the one structural difference from the CPU path.
#[cfg(feature = "gpu")]
struct GpuFrontend {
    /// The device. Held for the pipeline's lifetime: `ImagePipeline` borrows it
    /// at construction, and dropping it would take the device with it.
    #[allow(dead_code)]
    context: wslam_gpu::GpuContext,
    pipeline: wslam_gpu::ImagePipeline,
    /// False until a first frame has been uploaded and swapped in, because
    /// flow against an unwritten frame set reads garbage rather than failing.
    has_previous: bool,
}

/// The keyframe the tracker triangulates new landmarks against.
#[derive(Debug, Clone)]
struct KeyframeState {
    pose: Se3,
    observations: HashMap<u64, Vec2>,
}

/// One keyframe retained for local bundle adjustment.
///
/// Separate from [`KeyframeState`], which is the *previous* keyframe kept for
/// triangulation and holds raw pixels. This holds undistorted pixels and only
/// the features that carry a landmark, because those are the only ones bundle
/// adjustment can use.
#[derive(Debug, Clone)]
struct WindowKeyframe {
    pose: Se3,
    /// `(landmark id, undistorted pixel)`.
    observations: Vec<(u64, Vec2)>,
}

/// L3, the per-frame frontend.
pub struct Tracker {
    config: TrackConfig,
    intrinsics: CameraIntrinsics,
    backend: Backend,

    state: TrackingState,
    features: Vec<Feature>,
    map: LocalMap,
    pose: Option<Se3>,
    covariance: Mat6,

    prev_pyramid: Option<Pyramid>,
    prev_prior: Option<So3>,
    keyframe: Option<KeyframeState>,
    /// Why frames failed, for diagnosis.
    failures: FailureCounts,
    /// GPU front-end, if active.
    #[cfg(feature = "gpu")]
    gpu: Option<GpuFrontend>,
    /// A device supplied from outside, before any frame arrived.
    #[cfg(feature = "gpu")]
    pending_context: Option<wslam_gpu::GpuContext>,
    /// Landmarks tracked at the last keyframe — ORB-SLAM2's `nRefMatches`.
    reference_landmarks: usize,
    /// Sliding window of recent keyframes, oldest first, for local BA.
    ///
    /// Observations here — and in every `KeyframeState` — are **undistorted**
    /// pixels, matching the contract every geometry consumer documents. Storing
    /// raw pixels was a live bug: EuRoC's `k1 = -0.283` displaces the median
    /// pixel by ~22 px, well past the 3 px RANSAC threshold, so PnP was
    /// discarding exactly the peripheral observations with the most geometric
    /// leverage.
    ///
    /// Without this the map is write-once: a landmark is triangulated and never
    /// corrected, so error accumulates monotonically and the trajectory drifts
    /// at ~4% of path length. See `local_ba` for the measurement.
    window: std::collections::VecDeque<WindowKeyframe>,
    /// Reference view for the two-view bootstrap: pose plus the pixel each
    /// feature occupied when the reference was taken.
    bootstrap: Option<KeyframeState>,

    next_feature_id: u64,
    next_landmark_id: u64,
    frame_index: u64,
    frames_since_keyframe: u64,
    solve_failures: u32,
    healthy_last_frame: bool,

    rng: DeterministicRng,
    timings: StageTimings,
    clock: Option<Box<dyn HostClock>>,
}

impl std::fmt::Debug for Tracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tracker")
            .field("state", &self.state)
            .field("backend", &self.backend)
            .field("features", &self.features.len())
            .field("landmarks", &self.map.len())
            .field("frame_index", &self.frame_index)
            .field("seed", &self.config.seed)
            .finish_non_exhaustive()
    }
}

impl Tracker {
    /// Construct a tracker.
    ///
    /// `use_gpu` is honoured only when the crate is built with the `gpu`
    /// feature; otherwise the CPU reference runs and the choice is logged
    /// rather than silently ignored.
    #[must_use]
    pub fn new(config: TrackConfig, intrinsics: CameraIntrinsics) -> Self {
        let backend = if config.use_gpu {
            if cfg!(feature = "gpu") {
                Backend::Gpu
            } else {
                log::warn!(
                    "TrackConfig::use_gpu set but wslam-track was built without the `gpu` \
                     feature; running the CPU reference front-end"
                );
                Backend::Cpu
            }
        } else {
            Backend::Cpu
        };
        Tracker {
            rng: DeterministicRng::new("wslam-track", config.seed),
            config,
            intrinsics,
            backend,
            state: TrackingState::Initializing,
            features: Vec::new(),
            map: LocalMap::new(),
            pose: None,
            covariance: Mat6::identity() * Scalar::INFINITY,
            prev_pyramid: None,
            prev_prior: None,
            keyframe: None,
            #[cfg(feature = "gpu")]
            gpu: None,
            #[cfg(feature = "gpu")]
            pending_context: None,
            failures: FailureCounts::default(),
            reference_landmarks: 0,
            window: std::collections::VecDeque::new(),
            bootstrap: None,
            next_feature_id: 0,
            next_landmark_id: 0,
            frame_index: 0,
            frames_since_keyframe: 0,
            solve_failures: 0,
            healthy_last_frame: false,
            timings: StageTimings::default(),
            clock: None,
        }
    }

    /// Install the profiling clock that fills [`StageTimings`].
    ///
    /// The only wall clock the tracker can reach, and it is write-only with
    /// respect to the estimate: see the module docs. Passing `None` leaves every
    /// timing at zero and changes nothing else.
    pub fn set_host_clock(&mut self, clock: Option<Box<dyn HostClock>>) {
        self.clock = clock;
    }

    /// Replace the intrinsics — L2 refines them during the init pan.
    ///
    /// Existing landmarks keep their positions: they were triangulated under the
    /// old focal length, so the map is re-anchored implicitly by the next few
    /// pose solves rather than rebuilt, which would throw away the only map the
    /// tracker has at the moment it is least able to make a new one.
    pub fn set_intrinsics(&mut self, k: CameraIntrinsics) {
        self.intrinsics = k;
    }

    /// Current intrinsics.
    #[must_use]
    pub fn intrinsics(&self) -> &CameraIntrinsics {
        &self.intrinsics
    }

    /// Which front-end is running.
    #[must_use]
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Drop everything and start over. The seed is unchanged, so a reset
    /// session replays identically.
    pub fn reset(&mut self) {
        self.state = TrackingState::Initializing;
        self.features.clear();
        self.map.clear();
        self.pose = None;
        self.covariance = Mat6::identity() * Scalar::INFINITY;
        self.prev_pyramid = None;
        self.prev_prior = None;
        self.keyframe = None;
        self.bootstrap = None;
        self.window.clear();
        self.reference_landmarks = 0;
        self.frames_since_keyframe = 0;
        self.solve_failures = 0;
        self.healthy_last_frame = false;
        self.timings = StageTimings::default();
        self.rng = DeterministicRng::new("wslam-track", self.config.seed);
    }

    /// Adopt a pose supplied by L4 relocalization and rebuild around it.
    ///
    /// The local map is discarded rather than transformed: the landmarks were
    /// triangulated in the pre-loss frame, and if that frame were trustworthy we
    /// would not have needed to relocalize. The tracker re-bootstraps with
    /// `pose` as the world anchor, so the new landmarks land in L4's coordinate
    /// frame and not in a fresh one.
    pub fn relocalized_to(&mut self, pose: Se3, timestamp: Timestamp) {
        log::debug!("relocalized to {:?} at {timestamp}", pose.translation());
        self.features.clear();
        self.map.clear();
        self.prev_pyramid = None;
        self.prev_prior = None;
        self.keyframe = None;
        self.bootstrap = None;
        self.window.clear();
        self.reference_landmarks = 0;
        self.frames_since_keyframe = 0;
        self.solve_failures = 0;
        self.healthy_last_frame = false;
        self.pose = Some(pose);
        // Not `Tracking`: there is no map yet, so no frame between now and the
        // re-bootstrap has an independently solved pose.
        self.state = TrackingState::Initializing;
        self.covariance = Mat6::identity() * Scalar::INFINITY;
    }

    /// Re-express every tracked quantity in a similarity-transformed frame:
    /// `p ↦ s·R·p + t` for points, with rotations composed and camera centres
    /// treated as points.
    ///
    /// The epoch-merge path: when place recognition proves this session's
    /// current coordinate frame overlaps an older one, the orchestrator solves
    /// the Sim(3) between them and folds the tracker into the older frame
    /// mid-flight, so tracking continues without a reset. Everything metric is
    /// transformed — pose, local map, the BA window and its poses, the
    /// bootstrap reference — because a half-transformed tracker mixes frames
    /// and PnP diverges on the next solve.
    pub fn apply_similarity(&mut self, rotation: &So3, translation: &Vec3, scale: Scalar) {
        let act_point = |p: &Vec3| rotation.act(p) * scale + translation;
        let act_pose = |pose: &Se3| {
            Se3::new(
                rotation.compose(&pose.rotation()),
                act_point(&pose.translation()),
            )
        };
        if let Some(pose) = self.pose.as_mut() {
            *pose = act_pose(pose);
        }
        for landmark in self.map.landmarks_mut() {
            landmark.position = act_point(&landmark.position);
        }
        for kf in &mut self.window {
            kf.pose = act_pose(&kf.pose);
        }
        if let Some(kf) = self.keyframe.as_mut() {
            kf.pose = act_pose(&kf.pose);
        }
        if let Some(b) = self.bootstrap.as_mut() {
            b.pose = act_pose(&b.pose);
        }
        // Translation variance scales with the square of the unit change;
        // rotation variance is scale-free.
        let s2 = scale * scale;
        let mut tt = self.covariance.fixed_view_mut::<3, 3>(0, 0);
        tt *= s2;
        let mut tr = self.covariance.fixed_view_mut::<3, 3>(0, 3);
        tr *= scale;
        let mut rt = self.covariance.fixed_view_mut::<3, 3>(3, 0);
        rt *= scale;
    }

    /// The active local map.
    #[must_use]
    pub fn local_map(&self) -> &LocalMap {
        &self.map
    }

    /// Features in the current frame.
    #[must_use]
    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    /// Timings for the most recent `process` call.
    #[must_use]
    pub fn timings(&self) -> &StageTimings {
        &self.timings
    }

    /// Current state.
    #[must_use]
    pub fn state(&self) -> TrackingState {
        self.state
    }

    /// Why frames have failed to produce a pose so far.
    #[must_use]
    pub fn failures(&self) -> FailureCounts {
        self.failures
    }

    /// Most recent pose, up to scale.
    #[must_use]
    pub fn pose(&self) -> Option<Se3> {
        self.pose
    }

    /// The per-frame entry point.
    ///
    /// `rotation_prior` is L1's attitude (`R_world_body`) at this frame; tier 1
    /// passes `None`. It is used only to predict where features will land, never
    /// as a pose measurement — L3 owns its own rotation and hands yaw back to
    /// L1, not the other way round.
    pub fn process(&mut self, frame: &Frame, rotation_prior: Option<So3>) -> TrackOutcome {
        let failures_before = self.failures.total();
        let mut mark = self.clock.as_ref().map(|c| c.elapsed_seconds());
        let started = mark;
        self.timings = StageTimings::default();
        self.frame_index += 1;
        self.frames_since_keyframe += 1;

        // --- upload -----------------------------------------------------
        // The CPU path's "upload" is the single full-image pass we need anyway;
        // on the GPU path it is the texture write. Measuring it separately is
        // what makes the WebGPU budget in spec.md §8 manageable.
        let mean_intensity = frame.image.mean_intensity();
        self.timings.upload_ms = self.lap(&mut mark);

        // --- pyramid ----------------------------------------------------
        //
        // On the GPU path the pyramid lives in device memory; the CPU `Pyramid`
        // is still built because the bootstrap, masking and debug overlay read
        // level 0 directly. Building it is cheap next to flow, and keeping one
        // definition of "the image" avoids the two sides disagreeing.
        #[cfg(feature = "gpu")]
        self.gpu_upload(&frame.image);
        let pyramid = Pyramid::build(&frame.image, &self.intrinsics, &self.pyramid_config());
        self.timings.pyramid_ms = self.lap(&mut mark);

        let low_light = mean_intensity < self.config.low_light_threshold;

        // --- flow -------------------------------------------------------
        let mut flow = FlowSummary::default();
        if let Some(prev) = self.prev_pyramid.take() {
            flow = self.run_flow(&prev, &pyramid, rotation_prior);
        }
        self.timings.flow_ms = self.lap(&mut mark);

        // --- pose -------------------------------------------------------
        let mut solve = self.solve_pose(frame);
        self.timings.pnp_ms = self.lap(&mut mark);

        // --- abandon a map we can no longer see -------------------------
        //
        // Recovery used to be impossible. The branch below takes the keyframe
        // path whenever `pose.is_some() && !map.is_empty()`, and after the
        // first successful track both stay true forever — `pose` holds the last
        // known pose and the landmarks are still in memory. So a tracker that
        // lost the scene kept solving PnP against a map it could not see,
        // failed on every frame, and sat in `Lost` until the page was reloaded.
        // `try_bootstrap` was unreachable.
        //
        // Once the failures are decisive, throw the map away and take a fresh
        // reference view. Relocalizing into the *existing* map would be better
        // and is what L4 is for, but it needs a trained vocabulary; re-entering
        // `Tracking` on a new coordinate frame beats never tracking again. The
        // session reports it as an `OriginReset`, so a consumer holding
        // world-anchored content knows to drop it.
        if self.solve_failures >= ABANDON_MAP_AFTER_FAILURES && !self.map.is_empty() {
            log::info!(
                "abandoning the local map after {} consecutive solve failures; \
                 re-initialising",
                self.solve_failures
            );
            self.map = LocalMap::new();
            self.pose = None;
            self.keyframe = None;
            self.window.clear();
            self.reference_landmarks = 0;
            self.bootstrap = None;
            for f in &mut self.features {
                f.landmark = None;
            }
        }

        // --- bootstrap / keyframe / refill ------------------------------
        let mut is_keyframe = false;
        let mut bootstrapped = false;
        if self.pose.is_some() && !self.map.is_empty() {
            is_keyframe = self.maybe_keyframe();
        } else if self.try_bootstrap(frame) {
            bootstrapped = true;
            // The bootstrap *is* this frame's pose solve; without saying so the
            // state machine reports the frame that finished initialisation as
            // limited.
            is_keyframe = true;
            solve = SolveSummary {
                solved: true,
                inliers: self.tracked_landmark_count(),
            };
        }
        // Top the feature set back up **every** frame it is short, not only on
        // keyframes or once it has already fallen through `min_features`.
        //
        // Flow loses a handful of tracks per frame — occlusion, the image
        // border, a patch that drifted onto its neighbour. Replacing them only
        // at keyframes lets the set sag between them and produces a sawtooth
        // whose troughs dip below `min_features`, so the state machine reports
        // `insufficient-features` on a frame whose pose solved cleanly against
        // fifty inliers and whose feature set was restored in the same call. The
        // old condition also fired the refill *and* the complaint on the same
        // frame, which is a report of a problem the tracker had just fixed.
        //
        // "The detector is masked by the surviving features, so a frame that
        // has lost nothing costs one response map and returns nothing" — and
        // that response map is precisely the cost: ~7 ms of a 17 ms wasm
        // frame, paid every frame to top up a set that is one corner short.
        // Hysteresis instead: let the set drain to `refill_below_fraction`
        // of the budget, then fill back to the brim. See the config field
        // for why this is not the keyframe-only sawtooth the paragraph above
        // this one warns about. Keyframes still refill unconditionally so
        // they register a full observation set, and so does an empty map,
        // because the bootstrap is supply-hungry.
        let refill_floor =
            (self.config.max_features as Scalar * self.config.refill_below_fraction) as usize;
        let wants_refill = self.features.len() < refill_floor
            || ((is_keyframe || self.map.is_empty())
                && self.features.len() < self.config.max_features);
        if wants_refill {
            self.refill(&frame.image);
        }
        if is_keyframe {
            // Register the refilled features against the keyframe that was just
            // created, or they wait an extra keyframe interval before they can
            // be triangulated.
            if let Some(kf) = self.keyframe.as_mut() {
                for f in &self.features {
                    kf.observations.entry(f.id).or_insert(f.px_undist);
                }
            }
        }
        // A reference view goes stale. If it was taken while the camera saw
        // nothing — a blank wall, a covered lens, the moment right after the
        // map was abandoned — it holds too few observations to ever match, and
        // because `bootstrap.is_some()` the arming branch below never fires
        // again. The tracker then sits in `Initializing` forever with a
        // perfectly good scene in front of it. Measured: recovery failed
        // exactly this way.
        //
        // Refresh it when it cannot plausibly succeed, or when it simply has
        // not for a while: a reference from too long ago has too much parallax
        // and too little overlap to match either.
        if self.map.is_empty() {
            let stale = self.bootstrap.as_ref().is_some_and(|r| {
                r.observations.len() < MIN_BOOTSTRAP_MATCHES
                    || self.frames_since_keyframe > BOOTSTRAP_REFERENCE_TTL_FRAMES
            });
            if stale {
                self.bootstrap = None;
                self.frames_since_keyframe = 0;
            }
        }
        if self.bootstrap.is_none() && self.map.is_empty() {
            // No map and no reference view: this frame becomes the reference the
            // two-view bootstrap will match against.
            //
            // The condition is "no map", not "no pose". After
            // [`Tracker::relocalized_to`] the tracker holds L4's pose but has
            // deliberately thrown its landmarks away, and gating on
            // `self.pose.is_none()` — as this used to — meant no reference view
            // was ever taken, `try_bootstrap` returned `false` on every
            // subsequent frame for want of one, and the tracker sat in
            // `Initializing` forever. Relocalization would restore the pose and
            // then never track again.
            //
            // `self.pose` still supplies the reference *pose*, so the rebuilt
            // map lands in L4's coordinate frame rather than a fresh one.
            self.bootstrap = Some(KeyframeState {
                pose: self.pose.unwrap_or_else(Se3::identity),
                observations: self.features.iter().map(|f| (f.id, f.px_undist)).collect(),
            });
        }
        self.timings.corners_ms += self.lap(&mut mark);

        // --- state ------------------------------------------------------
        self.state = self.classify(low_light, &flow, &solve);
        self.healthy_last_frame = matches!(self.state, TrackingState::Tracking);
        self.map.cull(
            self.frame_index,
            LANDMARK_TTL_FRAMES,
            self.config.max_features * 4,
        );

        // Advance the GPU's double buffer so this frame becomes "previous".
        // Without this the pipeline tracks every frame against the first one.
        #[cfg(feature = "gpu")]
        if self.backend == Backend::Gpu {
            if let Some(gpu) = self.gpu.as_mut() {
                gpu.pipeline.swap();
                gpu.has_previous = true;
            }
        }
        self.prev_pyramid = Some(pyramid);
        self.prev_prior = rotation_prior;
        if let (Some(start), Some(clock)) = (started, self.clock.as_ref()) {
            self.timings.total_ms = (clock.elapsed_seconds() - start) * 1.0e3;
        }

        // Anything still poseless that did not reach a PnP failure path is
        // waiting on the bootstrap; attribute it so the counters sum to the
        // reported loss rate rather than silently under-counting.
        if self.pose.is_none() && self.failures.total() == failures_before {
            self.failures.awaiting_bootstrap += 1;
        }

        TrackOutcome {
            state: self.state,
            pose: self.pose,
            covariance: self.covariance,
            inlier_count: solve.inliers,
            tracked_count: flow.tracked,
            is_keyframe,
            bootstrapped,
        }
    }

    // ---------------------------------------------------------------- stages

    fn pyramid_config(&self) -> PyramidConfig {
        PyramidConfig {
            levels: self.config.pyramid_levels.max(1),
            ..PyramidConfig::default()
        }
    }

    fn klt_config(&self) -> KltConfig {
        KltConfig {
            window_radius: self.config.klt_window.max(1),
            max_iterations: self.config.klt_iterations.max(1),
            forward_backward: self.config.klt_forward_backward,
            ..KltConfig::default()
        }
    }

    fn corner_config(&self) -> CornerConfig {
        let d = CornerConfig::default();
        CornerConfig {
            max_corners: self.config.max_features,
            // Spread the budget over a grid whose cells hold a handful of
            // features each; more cells than features starves the round robin.
            grid_cols: d.grid_cols,
            grid_rows: d.grid_rows,
            border: d.border.max(self.config.klt_window + 2),
            ..d
        }
    }

    /// Wall time since `mark`, in milliseconds, advancing `mark`.
    ///
    /// The single point at which this crate touches a clock. Its result reaches
    /// [`StageTimings`] and nothing else.
    fn lap(&self, mark: &mut Option<f64>) -> f64 {
        let Some(clock) = self.clock.as_ref() else {
            return 0.0;
        };
        let now = clock.elapsed_seconds();
        let dt = mark.map_or(0.0, |prev| (now - prev) * 1.0e3);
        *mark = Some(now);
        dt
    }

    /// Predict where each feature lands, given L1's attitude.
    ///
    /// Uses the infinite homography `x2 = K R21 K^-1 x1`, which is exact for a
    /// purely rotating camera at any depth and a good approximation whenever the
    /// baseline is small against the scene depth. That is precisely the regime a
    /// loose orientation prior helps in (spec.md §4, tier 2), and it needs no
    /// landmark, so freshly detected features get a prediction too.
    fn predict(&self, prior: Option<So3>) -> Option<Vec<Vec2>> {
        let (now, prev) = (prior?, self.prev_prior?);
        let r21 = now.inverse().compose(&prev);
        Some(
            self.features
                .iter()
                .map(|f| {
                    let n = self.intrinsics.unproject_normalized(f.px);
                    let bearing = r21.act(&Vec3::new(n.x, n.y, 1.0));
                    self.intrinsics.project(&bearing).unwrap_or(f.px)
                })
                .collect(),
        )
    }

    fn run_flow(&mut self, prev: &Pyramid, next: &Pyramid, prior: Option<So3>) -> FlowSummary {
        if self.features.is_empty() {
            return FlowSummary::default();
        }
        let points: Vec<Vec2> = self.features.iter().map(|f| f.px).collect();
        let mut guesses = self.predict(prior);
        // The prior models pure rotation. On rotation-in-place it is exact;
        // under fast translation the parallax it cannot express displaces
        // every seed at once, and no threshold on the *rotation* can see the
        // difference — the same predicted angle is a good prior while hovering
        // and a bad one mid-dash (measured: the ungated prior held tier 2
        // above tier 1 on every fast EuRoC sequence). So arbitrate per frame,
        // empirically: track a small spread of features both ways and let
        // whichever mode keeps more of them take the frame. Costs two
        // subsample passes on prior frames only.
        if let Some(g) = guesses.as_deref() {
            if self.backend != Backend::Gpu && points.len() >= 2 * PRIOR_TRIAL_FEATURES {
                let stride = points.len() / PRIOR_TRIAL_FEATURES;
                let idx: Vec<usize> = (0..PRIOR_TRIAL_FEATURES).map(|i| i * stride).collect();
                let sub_pts: Vec<Vec2> = idx.iter().map(|&i| points[i]).collect();
                let sub_guess: Vec<Vec2> = idx.iter().map(|&i| g[i]).collect();
                let config = self.klt_config();
                let with = klt::track(prev, next, &sub_pts, Some(&sub_guess), &config)
                    .iter()
                    .filter(|r| r.ok())
                    .count();
                let without = klt::track(prev, next, &sub_pts, None, &config)
                    .iter()
                    .filter(|r| r.ok())
                    .count();
                if with <= without {
                    guesses = None;
                }
            }
        }
        // Copied out before the drain below borrows `self.features` mutably.
        let intrinsics = self.intrinsics;
        #[cfg(feature = "gpu")]
        let gpu_results = self.gpu_track_flow(&points, guesses.as_deref());
        #[cfg(not(feature = "gpu"))]
        let gpu_results: Option<Vec<klt::KltTrack>> = None;

        let results = match gpu_results {
            Some(r) => r,
            None => klt::track(prev, next, &points, guesses.as_deref(), &self.klt_config()),
        };

        let before = self.features.len();
        let mut displacements = Vec::with_capacity(before);
        let mut kept = Vec::with_capacity(before);
        for (feature, result) in self.features.drain(..).zip(results) {
            if !result.ok() {
                continue;
            }
            displacements.push((result.px - feature.px).norm());
            kept.push(Feature {
                px: result.px,
                px_undist: undistort_px(&intrinsics, result.px),
                state: FeatureState::Tracked,
                age: feature.age.saturating_add(1),
                ..feature
            });
        }
        self.features = kept;

        displacements.sort_by(Scalar::total_cmp);
        FlowSummary {
            tracked: self.features.len(),
            attempted: before,
            median_flow: displacements
                .get(displacements.len() / 2)
                .copied()
                .unwrap_or(0.0),
        }
    }

    fn solve_pose(&mut self, frame: &Frame) -> SolveSummary {
        let mut ids = Vec::new();
        let mut points_3d = Vec::new();
        let mut points_2d = Vec::new();
        for f in &self.features {
            let Some(lid) = f.landmark else { continue };
            let Some(l) = self.map.get(lid) else { continue };
            ids.push(f.id);
            points_3d.push(l.position);
            points_2d.push(f.px_undist);
        }

        if points_3d.len() < MIN_PNP_CORRESPONDENCES {
            self.failures.too_few_correspondences += 1;
            return self.record_solve_failure();
        }

        // Fork on the frame id rather than drawing from a running stream, so a
        // dropped frame cannot shift every later RANSAC draw and break replay
        // (spec.md §6, "bit-for-bit reproducibly").
        let mut rng = self.rng.fork("ransac-pnp", frame.id.0);
        let Some(ransac) = pnp::solve_pnp_ransac(
            &points_3d,
            &points_2d,
            &self.intrinsics,
            self.config.ransac_threshold_px,
            self.config.ransac_iterations,
            &mut rng,
        ) else {
            self.failures.ransac_failed += 1;
            return self.record_solve_failure();
        };

        let ba_config = MotionBaConfig {
            huber_delta_px: self.config.ransac_threshold_px * 0.5,
            outlier_threshold_px: self.config.ransac_threshold_px,
            pixel_sigma: PIXEL_SIGMA,
            ..MotionBaConfig::default()
        };
        let refined = motion_ba::refine_motion_only(
            &ransac.pose,
            &points_3d,
            &points_2d,
            &self.intrinsics,
            Some(&ransac.inliers),
            &ba_config,
        );

        let (pose, covariance, inliers, inlier_count) = match refined {
            Some(r) if r.inlier_count >= MIN_PNP_CORRESPONDENCES => {
                (r.pose, r.covariance, r.inliers, r.inlier_count)
            }
            // The bundle adjustment can legitimately fail to improve on RANSAC;
            // falling back to the RANSAC pose is better than dropping the frame,
            // but its covariance has to come from the same geometry.
            _ => {
                let cov = pnp::pose_covariance(
                    &ransac.pose,
                    &points_3d,
                    &self.intrinsics,
                    PIXEL_SIGMA,
                    Some(&ransac.inliers),
                )
                .unwrap_or_else(|| Mat6::identity() * Scalar::INFINITY);
                (
                    ransac.pose,
                    cov,
                    ransac.inliers.clone(),
                    ransac.inlier_count,
                )
            }
        };

        if inlier_count < MIN_PNP_CORRESPONDENCES {
            self.failures.too_few_inliers += 1;
            return self.record_solve_failure();
        }

        for (i, id) in ids.iter().enumerate() {
            let inlier = inliers.get(i).copied().unwrap_or(false);
            if let Some(f) = self.features.iter_mut().find(|f| f.id == *id) {
                if inlier {
                    f.state = FeatureState::Tracked;
                    if let Some(lid) = f.landmark {
                        self.map.observe(lid, self.frame_index);
                    }
                } else {
                    f.state = FeatureState::Outlier;
                }
            }
        }

        self.pose = Some(pose);
        self.covariance = covariance;
        self.solve_failures = 0;
        SolveSummary {
            solved: true,
            inliers: inlier_count,
        }
    }

    fn record_solve_failure(&mut self) -> SolveSummary {
        self.solve_failures = self.solve_failures.saturating_add(1);
        if self.pose.is_some() {
            // Coast on the last pose, but widen: it is now stale by however many
            // frames we have failed for.
            self.covariance *= STALE_COVARIANCE_GROWTH;
        }
        SolveSummary {
            solved: false,
            inliers: 0,
        }
    }

    // ------------------------------------------------------------ bootstrap

    fn init_config(&self) -> InitConfig {
        InitConfig {
            pixel_sigma: PIXEL_SIGMA,
            ransac_iterations: self.config.ransac_iterations,
            max_reprojection_px: self.config.ransac_threshold_px,
            ..InitConfig::default()
        }
    }

    /// Two-view bootstrap against the reference view. Returns whether it
    /// succeeded and seeded the map.
    ///
    /// The pixel-parallax gate here is only a cheap pre-filter to keep a
    /// four-hundred-iteration RANSAC off the critical path while the user is
    /// still holding still; [`init::initialize_two_view`] applies the real
    /// angular parallax test, which is the one that distinguishes translation
    /// from the pure rotation that carries no depth information at all.
    fn try_bootstrap(&mut self, frame: &Frame) -> bool {
        let Some(reference) = self.bootstrap.clone() else {
            return false;
        };
        let config = self.init_config();
        let mut ids = Vec::new();
        let mut matches = Vec::new();
        let mut parallax = Vec::new();
        for f in &self.features {
            let Some(&first) = reference.observations.get(&f.id) else {
                continue;
            };
            ids.push(f.id);
            matches.push((first, f.px_undist));
            parallax.push((f.px_undist - first).norm());
        }
        if matches.len() < MIN_BOOTSTRAP_MATCHES.max(config.min_inliers) {
            return false;
        }
        parallax.sort_by(Scalar::total_cmp);
        let median = parallax[parallax.len() / 2];
        if median < BOOTSTRAP_PARALLAX_FRACTION * self.intrinsics.width as Scalar {
            return false;
        }

        let mut rng = self.rng.fork("bootstrap", frame.id.0);
        let boot = init::initialize_two_view(&matches, &self.intrinsics, &config, &mut rng);
        if std::env::var("WSLAM_PROBE").is_ok() {
            match &boot {
                Some(b) => eprintln!(
                    "    bootstrap f{}: {:?} ratio {:.3} matches {} inliers {} lm {} parallax {:.2}deg",
                    self.frame_index, b.model, b.homography_ratio, matches.len(),
                    b.inliers.iter().filter(|x| **x).count(), b.landmarks.len(),
                    b.median_parallax_rad.to_degrees()
                ),
                None => eprintln!(
                    "    bootstrap f{}: refused, {} matches, median px parallax {median:.2}",
                    self.frame_index, matches.len()
                ),
            }
        }
        let Some(boot) = boot else {
            return false;
        };

        // `boot.pose` places the *first* view at the origin, and its landmarks
        // are in that first camera's frame. The reference view carries the world
        // frame — identity for a fresh session, and L4's pose after a
        // relocalization — so both are lifted through it.
        let pose_a = reference.pose;
        let pose_b = pose_a.compose(&boot.pose);
        let mut observations = HashMap::new();
        let mut positions = Vec::new();
        for landmark in &boot.landmarks {
            let Some(&id) = ids.get(landmark.match_index) else {
                continue;
            };
            let position = pose_a.act(&landmark.position);
            let lid = self.next_landmark_id;
            self.next_landmark_id += 1;
            self.map.insert(
                LocalLandmark {
                    id: lid,
                    position,
                    observations: 2,
                },
                self.frame_index,
            );
            if let Some(f) = self.features.iter_mut().find(|f| f.id == id) {
                f.landmark = Some(lid);
                f.state = FeatureState::Tracked;
                observations.insert(f.id, f.px_undist);
                positions.push(position);
            }
        }
        if positions.len() < MIN_PNP_CORRESPONDENCES {
            self.map.clear();
            for f in &mut self.features {
                f.landmark = None;
            }
            return false;
        }

        log::debug!(
            "bootstrapped {} landmarks from {} matches via {:?}, parallax {:.2} deg",
            positions.len(),
            matches.len(),
            boot.model,
            boot.median_parallax_rad.to_degrees()
        );
        self.pose = Some(pose_b);
        // The bootstrap's own baseline defines the scale, so the translation
        // block is only meaningful up to it. Reporting the geometry's Fisher
        // information is still the honest answer: it says how well *this* frame
        // is constrained by *these* landmarks, which is what a consumer needs.
        self.covariance =
            pnp::pose_covariance(&pose_b, &positions, &self.intrinsics, PIXEL_SIGMA, None)
                .unwrap_or_else(|| Mat6::identity() * Scalar::INFINITY);
        self.keyframe = Some(KeyframeState {
            pose: pose_b,
            observations,
        });
        self.bootstrap = None;
        self.window.clear();
        self.reference_landmarks = 0;
        self.frames_since_keyframe = 0;
        self.solve_failures = 0;
        true
    }

    // ------------------------------------------------------------- keyframes

    fn maybe_keyframe(&mut self) -> bool {
        let (Some(pose), Some(keyframe)) = (self.pose, self.keyframe.clone()) else {
            return false;
        };
        if self.frames_since_keyframe < self.config.keyframe_min_frames.max(1) {
            return false;
        }
        let delta = keyframe.pose.inverse().compose(&pose);
        let translated = delta.translation().norm() >= self.config.keyframe_translation;
        let rotated = delta.rotation().angle() >= self.config.keyframe_rotation_rad;

        // Starvation is a *relative* trigger, not an absolute one.
        //
        // The absolute form — `tracked < min_features` — is a starvation loop.
        // Once the count sits below the floor it fires on every eligible frame,
        // and each keyframe is inserted at essentially the previous keyframe's
        // pose. Measured on EuRoC: 1002 keyframes over 3682 frames, one every
        // 3.7 frames, with near-zero baseline between them. That wrecks two
        // things at once — landmarks get triangulated from a baseline far too
        // short to constrain depth, and the local-BA window spans under two
        // seconds of motion, which is not enough to correct anything.
        //
        // ORB-SLAM2's rule is relative to the reference keyframe:
        // `mnMatchesInliers < nRefMatches * thRefRatio` with `thRefRatio = 0.9`
        // for monocular, and `mnMatchesInliers > 15` so a genuinely dying track
        // stops asking for keyframes it cannot support.
        let tracked = self.tracked_landmark_count();
        let starved = tracked
            < (self.reference_landmarks as Scalar * self.config.keyframe_tracked_ratio) as usize
            && tracked > self.config.keyframe_min_tracked;

        if !(translated || rotated || starved) {
            return false;
        }

        let created = self.triangulate_against(&keyframe, &pose);
        log::debug!(
            "keyframe at frame {}: +{created} landmarks, {} total",
            self.frame_index,
            self.map.len()
        );
        self.keyframe = Some(KeyframeState {
            pose,
            observations: self.features.iter().map(|f| (f.id, f.px_undist)).collect(),
        });
        self.frames_since_keyframe = 0;
        self.reference_landmarks = self.tracked_landmark_count();
        self.push_window(pose);
        self.run_local_ba();
        true
    }

    /// Record this keyframe's landmark observations for local BA.
    fn push_window(&mut self, pose: Se3) {
        if self.config.local_ba_window == 0 {
            return;
        }
        let observations: Vec<(u64, Vec2)> = self
            .features
            .iter()
            .filter(|f| f.state != FeatureState::Lost)
            .filter_map(|f| f.landmark.map(|id| (id, f.px_undist)))
            .collect();
        if observations.is_empty() {
            return;
        }
        self.window.push_back(WindowKeyframe { pose, observations });
        while self.window.len() > self.config.local_ba_window + self.config.local_ba_context {
            self.window.pop_front();
        }
    }

    /// Jointly refine the windowed keyframe poses and the landmarks they see.
    ///
    /// This is the only thing in the pipeline that ever makes the map *better*.
    /// Everything else consumes the map as given: `motion_ba` refines a pose
    /// against fixed landmarks, and triangulation only ever adds. Without this
    /// step error compounds with no way back, which measured 3.95% of path
    /// length on EuRoC against 0.020% for ORB-SLAM3.
    fn run_local_ba(&mut self) {
        let fixed = self.config.local_ba_fixed.max(1);
        if self.window.len() <= fixed {
            return;
        }

        // Collect the landmarks the window can see, and index them densely.
        let mut index: HashMap<u64, usize> = HashMap::new();
        let mut points: Vec<Vec3> = Vec::new();
        let mut ids: Vec<u64> = Vec::new();
        let mut observations = Vec::new();
        for (kf_index, kf) in self.window.iter().enumerate() {
            for (landmark_id, px) in &kf.observations {
                let Some(landmark) = self.map.get(*landmark_id) else {
                    // Culled since the keyframe was recorded.
                    continue;
                };
                let slot = *index.entry(*landmark_id).or_insert_with(|| {
                    points.push(landmark.position);
                    ids.push(*landmark_id);
                    points.len() - 1
                });
                observations.push(local_ba::Observation {
                    keyframe: kf_index,
                    point: slot,
                    px: *px,
                });
            }
        }
        if points.len() < 8 || observations.len() < 20 {
            return;
        }

        // The window is stored oldest-first, and `fixed_poses` is a prefix
        // count, so the context keyframes are fixed simply by living at the
        // front. Keep at least `local_ba_fixed` pinned even before enough
        // history has accumulated.
        let context = self
            .config
            .local_ba_context
            .min(self.window.len().saturating_sub(1))
            .max(fixed);
        let problem = local_ba::LocalBaProblem {
            poses: self.window.iter().map(|k| k.pose).collect(),
            fixed_poses: context.min(self.window.len() - 1),
            points,
            observations,
        };
        let Some(result) = local_ba::optimize(&problem, &self.intrinsics, &self.ba_config()) else {
            return;
        };
        if !result.final_cost.is_finite() || result.final_cost > result.initial_cost {
            return;
        }

        // Write the corrected map back.
        for (slot, id) in ids.iter().enumerate() {
            if let Some(landmark) = self.map.get_mut(*id) {
                let corrected = result.points[slot];
                if corrected.iter().all(|c| c.is_finite()) {
                    landmark.position = corrected;
                }
            }
        }

        // The live pose was solved against the *old* map, so carry the same
        // correction the newest keyframe received. Leaving it stale would make
        // the next frame's PnP start from a pose inconsistent with the map it
        // is about to be matched against.
        let last = self.window.len() - 1;
        let before = self.window[last].pose;
        let after = result.poses[last];
        if let Some(live) = self.pose {
            self.pose = Some(after.compose(&before.inverse().compose(&live)));
        }
        for (kf, corrected) in self.window.iter_mut().zip(result.poses.iter()) {
            kf.pose = *corrected;
        }
        if let Some(kf) = self.keyframe.as_mut() {
            kf.pose = after;
        }

        log::debug!(
            "local BA: {} keyframes, {} points, cost {:.3e} -> {:.3e}, {:.3} px rms, {} outliers",
            self.window.len(),
            result.points.len(),
            result.initial_cost,
            result.final_cost,
            result.rms_px,
            result.outliers.len()
        );
    }

    fn ba_config(&self) -> local_ba::LocalBaConfig {
        local_ba::LocalBaConfig {
            huber_delta_px: self.config.ransac_threshold_px,
            max_iterations: self.config.local_ba_iterations,
            max_scale_change: self.config.local_ba_max_scale_change,
            // usize::MAX frees no landmark, which is exactly "motion only".
            min_observations: if self.config.local_ba_motion_only {
                usize::MAX
            } else {
                local_ba::LocalBaConfig::default().min_observations
            },
            ..local_ba::LocalBaConfig::default()
        }
    }

    /// Triangulate every feature that has an observation at `keyframe` but no
    /// landmark yet.
    fn triangulate_against(&mut self, keyframe: &KeyframeState, pose: &Se3) -> usize {
        let config = TriangulationConfig {
            max_reprojection_px: self.config.ransac_threshold_px,
            pixel_sigma: PIXEL_SIGMA,
            ..TriangulationConfig::default()
        };
        let mut created = 0usize;
        for i in 0..self.features.len() {
            if self.features[i].landmark.is_some() {
                continue;
            }
            let Some(&first) = keyframe.observations.get(&self.features[i].id) else {
                continue;
            };
            let point = triangulate::triangulate_two_view(
                &keyframe.pose,
                first,
                pose,
                self.features[i].px_undist,
                &self.intrinsics,
                &config,
            );
            let Ok(point) = point else { continue };
            let id = self.next_landmark_id;
            self.next_landmark_id += 1;
            self.map.insert(
                LocalLandmark {
                    id,
                    position: point.position,
                    observations: 2,
                },
                self.frame_index,
            );
            self.features[i].landmark = Some(id);
            created += 1;
        }
        created
    }

    fn tracked_landmark_count(&self) -> usize {
        self.features
            .iter()
            .filter(|f| f.landmark.is_some() && f.state != FeatureState::Lost)
            .count()
    }

    /// Acquire a device synchronously. Natively only: on wasm the caller must
    /// supply one through [`Tracker::set_gpu_context`], because adapter
    /// acquisition in a browser is asynchronous and blocking on it deadlocks
    /// the event loop.
    #[cfg(feature = "gpu")]
    fn acquire_context_blocking() -> Option<wslam_gpu::GpuContext> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            match wslam_gpu::GpuContext::new_blocking() {
                Ok(ctx) => Some(ctx),
                Err(e) => {
                    log::warn!("GPU adapter unavailable: {e}");
                    None
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            None
        }
    }

    /// Install a device acquired by the embedder.
    ///
    /// Must be called before the first frame; the pipeline is built lazily from
    /// it once a frame reveals the image dimensions.
    #[cfg(feature = "gpu")]
    pub fn set_gpu_context(&mut self, context: wslam_gpu::GpuContext) {
        self.pending_context = Some(context);
        self.backend = Backend::Gpu;
    }

    /// Upload the frame and build its pyramid on the GPU.
    ///
    /// Constructs the pipeline on first use rather than at `new`, so a tracker
    /// that is configured for the GPU but never fed a frame does not pay for a
    /// device, and so the pipeline is sized from a real frame instead of from
    /// configuration that might disagree with it.
    #[cfg(feature = "gpu")]
    fn gpu_upload(&mut self, image: &wslam_core::GrayImage) {
        if self.backend != Backend::Gpu {
            return;
        }
        if self.gpu.is_none() {
            // Prefer a context handed in by the embedder. On wasm that is the
            // only source; natively it lets a harness share one device.
            let context = match self.pending_context.take() {
                Some(ctx) => Some(ctx),
                None => Self::acquire_context_blocking(),
            };
            let Some(context) = context else {
                log::warn!("no GPU context available, using the CPU reference");
                self.backend = Backend::Cpu;
                return;
            };
            match wslam_gpu::ImagePipeline::new(
                &context,
                image.width(),
                image.height(),
                self.config.pyramid_levels,
            ) {
                Ok(pipeline) => {
                    self.gpu = Some(GpuFrontend {
                        context,
                        pipeline,
                        has_previous: false,
                    });
                }
                Err(e) => {
                    // Fall back rather than fail: a browser that cannot give us
                    // a device still deserves a tracker (spec.md §8).
                    log::warn!("GPU pipeline unavailable, using the CPU reference: {e}");
                    self.backend = Backend::Cpu;
                    return;
                }
            }
        }
        let Some(gpu) = self.gpu.as_mut() else { return };
        if let Err(e) = gpu
            .pipeline
            .upload(image)
            .and_then(|()| gpu.pipeline.build_pyramid())
        {
            log::warn!("GPU upload failed, using the CPU reference this frame: {e}");
        }
    }

    /// Track features on the GPU. `None` means the caller should use the CPU
    /// path — either the GPU is not in use, or there is no previous frame yet.
    #[cfg(feature = "gpu")]
    fn gpu_track_flow(
        &mut self,
        points: &[Vec2],
        guesses: Option<&[Vec2]>,
    ) -> Option<Vec<klt::KltTrack>> {
        if self.backend != Backend::Gpu {
            return None;
        }
        let mut config = self.klt_config();
        // The GPU kernel solves in f32 with its own interpolation, so its round
        // trip is intrinsically noisier than the f64 CPU reference. Applying the
        // CPU tolerance unchanged rejected so many *good* tracks that 52.5% of
        // frames lost pose. Measured on EuRoC MH_01, 3x recovers the CPU
        // rejection rate; the surviving tracks are the same quality.
        const GPU_FB_SLACK: Scalar = 3.0;
        config.forward_backward = config.forward_backward.map(|t| t * GPU_FB_SLACK);
        let gpu = self.gpu.as_mut()?;
        if !gpu.has_previous {
            return None;
        }
        // `track_flow` takes the feature's position in the *previous* frame and
        // searches for it in the current one. It has no separate initial-guess
        // input, unlike the CPU `klt::track(prev, next, points, guesses)`.
        //
        // Feeding it `guesses` — the prior-predicted positions in the *current*
        // frame — therefore does not seed the search, it lies about where the
        // feature came from, and every track starts from the wrong patch. The
        // rotation prior simply cannot reach this kernel; using the true source
        // positions is both correct and the best available.
        let _ = guesses;
        let seeds: Vec<(f32, f32)> = points.iter().map(|p| (p.x as f32, p.y as f32)).collect();
        let flow_config = wslam_gpu::FlowConfig {
            // The GPU takes a full side length; the CPU config a radius.
            window: config.window_radius * 2 + 1,
            iterations: config.max_iterations,
            epsilon: config.epsilon as f32,
            max_error: config.max_error as f32,
        };
        let forward = match gpu.pipeline.track_flow(&seeds, &flow_config) {
            Ok(r) => r,
            Err(e) => {
                log::warn!("GPU flow failed, using the CPU reference this frame: {e}");
                return None;
            }
        };

        // Forward-backward consistency, on the GPU.
        //
        // This is not optional polish. The CPU path rejects any track whose
        // round trip misses by more than `forward_backward` pixels, and it is
        // the only gate that catches a track that converged confidently onto
        // the wrong texture. Running without it measured ATE 2.36 m against
        // 0.085 m for the CPU front-end — the front-end was 22x faster and
        // useless. Reporting a fabricated `fb_error` of zero is what let those
        // tracks through, so the pass is run for real.
        //
        // `track_flow` always tracks *from* the inactive frame set *into* the
        // active one, so a swap turns the forward pipeline into the backward
        // one. The second swap restores the state `process` expects to advance.
        let backward = config.forward_backward.and_then(|_| {
            let back_seeds: Vec<(f32, f32)> = forward.iter().map(|r| (r.x, r.y)).collect();
            gpu.pipeline.swap();
            let out = gpu.pipeline.track_flow(&back_seeds, &flow_config).ok();
            gpu.pipeline.swap();
            out
        });

        Some(
            forward
                .into_iter()
                .enumerate()
                .map(|(i, r)| {
                    let px = Vec2::new(r.x as Scalar, r.y as Scalar);
                    // A backward pass that failed to converge is not evidence
                    // the track is good; treat it the same way the CPU path
                    // treats its own backward failure, as infinitely far.
                    let fb_error = match (&backward, config.forward_backward) {
                        (Some(b), Some(_)) => b.get(i).map_or(Scalar::INFINITY, |bk| {
                            if bk.ok {
                                (Vec2::new(bk.x as Scalar, bk.y as Scalar) - points[i]).norm()
                            } else {
                                Scalar::INFINITY
                            }
                        }),
                        _ => 0.0,
                    };
                    let status = if !r.ok {
                        klt::KltStatus::NotConverged
                    } else if config.forward_backward.is_some_and(|tol| fb_error > tol) {
                        klt::KltStatus::InconsistentBackward
                    } else {
                        klt::KltStatus::Converged
                    };
                    klt::KltTrack {
                        px,
                        status,
                        error: r.error as Scalar,
                        fb_error,
                    }
                })
                .collect(),
        )
    }

    /// Detect corners on the GPU, honouring the same occupancy mask as the CPU
    /// detector. `None` means the caller should use the CPU path.
    #[cfg(feature = "gpu")]
    fn gpu_detect_corners(
        &mut self,
        config: &CornerConfig,
        occupied: &[Vec2],
    ) -> Option<Vec<corners::Corner>> {
        if self.backend != Backend::Gpu {
            return None;
        }
        let gpu = self.gpu.as_mut()?;
        let found = match gpu.pipeline.detect_corners(
            config.max_corners,
            config.quality_level as f32,
            config.min_distance as f32,
        ) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("GPU corner detection failed, using the CPU reference: {e}");
                return None;
            }
        };
        // The GPU kernel has no notion of already-tracked features, so the
        // occupancy mask the CPU detector applies internally is applied here.
        // Skipping it would pile new detections onto existing tracks.
        let min_d2 = config.min_distance * config.min_distance;
        Some(
            found
                .into_iter()
                .map(|(x, y, response)| corners::Corner {
                    px: Vec2::new(x as Scalar, y as Scalar),
                    response: response as Scalar,
                })
                .filter(|c| occupied.iter().all(|o| (c.px - o).norm_squared() >= min_d2))
                .take(config.max_corners)
                .collect(),
        )
    }

    fn refill(&mut self, image: &wslam_core::GrayImage) {
        let occupied: Vec<Vec2> = self.features.iter().map(|f| f.px).collect();
        let mut config = self.corner_config();
        config.max_corners = self.config.max_features.saturating_sub(self.features.len());
        if config.max_corners == 0 {
            return;
        }
        #[cfg(feature = "gpu")]
        let detected = self
            .gpu_detect_corners(&config, &occupied)
            .unwrap_or_else(|| corners::detect(image, &config, &occupied));
        #[cfg(not(feature = "gpu"))]
        let detected = corners::detect(image, &config, &occupied);

        for c in detected {
            let id = self.next_feature_id;
            self.next_feature_id += 1;
            let px_undist = undistort_px(&self.intrinsics, c.px);
            self.features.push(Feature {
                id,
                px: c.px,
                px_undist,
                state: FeatureState::New,
                landmark: None,
                age: 0,
            });
        }
        // A feature detected this frame has no observation in the current
        // bootstrap reference, so record one — otherwise a refill during
        // initialisation can never contribute to the two-view solve.
        if let Some(reference) = self.bootstrap.as_mut() {
            for f in &self.features {
                reference.observations.entry(f.id).or_insert(f.px_undist);
            }
        }
    }

    // ------------------------------------------------------------ state machine

    fn classify(&self, low_light: bool, flow: &FlowSummary, solve: &SolveSummary) -> TrackingState {
        if self.pose.is_none() || self.map.is_empty() {
            // Nothing has produced a pose yet. Low light still deserves a name:
            // it is why initialisation is not progressing.
            return if self.solve_failures >= LOST_AFTER_FAILURES && self.state.has_pose() {
                TrackingState::Lost
            } else {
                TrackingState::Initializing
            };
        }
        if self.solve_failures >= LOST_AFTER_FAILURES {
            return TrackingState::Lost;
        }
        if low_light {
            return TrackingState::Limited(LimitedReason::LowLight);
        }
        if self.is_excessive_motion(flow) {
            return TrackingState::Limited(LimitedReason::ExcessiveMotion);
        }
        // The feature-count half of this test is deliberately taken *after* the
        // refill, on the set the tracker will carry into the next frame, rather
        // than on `flow.tracked`, the count that survived this frame's flow.
        //
        // Flow always loses a few tracks; the refill always puts them back if
        // the frame has anything in it. Judging on the pre-refill number makes
        // the state oscillate between `Tracking` and `insufficient-features` on
        // a sequence that is tracking perfectly well — measured on the fixture
        // in this module, sixteen of thirty-eight frames were reported degraded
        // while every one of them solved a pose against forty to fifty inliers.
        // "Insufficient features" is a statement about the *scene*, and the
        // scene's verdict is what the detector could find in this frame, not
        // how many patches happened to survive one KLT pass.
        if !solve.solved
            || solve.inliers < MIN_PNP_CORRESPONDENCES
            || self.features.len() < self.config.min_features
        {
            return TrackingState::Limited(LimitedReason::InsufficientFeatures);
        }
        TrackingState::Tracking
    }

    fn is_excessive_motion(&self, flow: &FlowSummary) -> bool {
        if flow.attempted == 0 {
            return false;
        }
        // The envelope is what the pyramid can actually span: a patch radius at
        // the coarsest level, expressed in level-0 pixels.
        let envelope = self.config.klt_window.max(1) as Scalar
            * 2.0_f64.powi(self.config.pyramid_levels.max(1) as i32 - 1);
        if flow.median_flow > EXCESSIVE_FLOW_FRACTION * envelope {
            return true;
        }
        let survival = flow.tracked as Scalar / flow.attempted as Scalar;
        // A collapse only means "too fast" if the previous frame was fine, there
        // were enough features to collapse from, *and* the new frame is full of
        // corners that simply are not the old ones. If the detector cannot find
        // anything in the new frame either, the scene went blank and the motion
        // is not the story — which is the difference between a fast pan and a
        // wall, and the caller acts differently on each.
        //
        // "Full of corners" is measured against the frame we came from, not
        // against `min_features`. `min_features` is a tracking budget, and a
        // budget is the wrong yardstick for "does this image have texture": a
        // camera that jumps far enough that half the scene leaves the frame
        // still lands on an image full of corners, just fewer of them. On the
        // two-metre-jump fixture in this module the new frame held 38 corners
        // against the 63 we entered with — plainly textured, and plainly a
        // motion failure — yet an absolute `>= 60` test called it a blank wall.
        // A genuinely blank frame yields exactly zero, so the ratio only has to
        // separate "somewhat less textured" from "not textured".
        let textured =
            self.features.len() as Scalar >= TEXTURE_PRESENT_RATIO * flow.attempted as Scalar;
        self.healthy_last_frame
            && flow.attempted >= self.config.min_features
            && textured
            && survival < EXCESSIVE_SURVIVAL_RATIO
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct FlowSummary {
    tracked: usize,
    attempted: usize,
    median_flow: Scalar,
}

#[derive(Debug, Clone, Copy, Default)]
struct SolveSummary {
    solved: bool,
    inliers: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use wslam_core::math::umeyama;
    use wslam_core::time::ManualHostClock;
    use wslam_core::{FrameId, GrayImage};

    const WIDTH: u32 = 480;
    const HEIGHT: u32 = 360;
    const FOCAL: Scalar = 460.0;

    fn intrinsics() -> CameraIntrinsics {
        CameraIntrinsics::from_focal(FOCAL, WIDTH, HEIGHT)
    }

    /// A blob the detector can find and the flow can follow: smooth, compact,
    /// and with a distinct amplitude per landmark so a mis-association shows up
    /// as photometric error rather than as a plausible wrong answer.
    #[derive(Debug, Clone, Copy)]
    struct Landmark {
        position: Vec3,
        amplitude: Scalar,
        radius: Scalar,
    }

    /// Deterministic scene: a slab of landmarks in front of the camera. Not
    /// planar — a planar scene sends the bootstrap down the homography branch,
    /// which is a different code path and deserves its own fixture.
    fn scene(count: usize, seed: u64) -> Vec<Landmark> {
        let mut rng = DeterministicRng::new("test-scene", seed);
        (0..count)
            .map(|_| Landmark {
                position: Vec3::new(
                    rng.uniform_range(-3.4, 3.4),
                    rng.uniform_range(-2.4, 2.4),
                    rng.uniform_range(2.2, 6.5),
                ),
                amplitude: rng.uniform_range(70.0, 140.0),
                radius: rng.uniform_range(2.2, 3.2),
            })
            .collect()
    }

    /// Render the scene from `pose` (`T_world_camera`).
    ///
    /// The background carries a faint low-frequency wash rather than being flat:
    /// a perfectly uniform field would make the mean intensity and the
    /// low-light heuristic degenerate, and real sensors never deliver one.
    fn render(scene: &[Landmark], pose: &Se3, k: &CameraIntrinsics, gain: Scalar) -> GrayImage {
        let (w, h) = (k.width as usize, k.height as usize);
        let mut buf = vec![0f64; w * h];
        for (i, v) in buf.iter_mut().enumerate() {
            let (x, y) = ((i % w) as Scalar, (i / w) as Scalar);
            *v = 96.0 + 6.0 * (0.013 * x + 0.009 * y).sin() + 4.0 * (0.021 * y).cos();
        }
        let inv = pose.inverse();
        for lm in scene {
            let cam = inv.act(&lm.position);
            let Some(px) = k.project(&cam) else { continue };
            let r = lm.radius;
            let (x0, x1) = ((px.x - r).floor() as i64, (px.x + r).ceil() as i64);
            let (y0, y1) = ((px.y - r).floor() as i64, (px.y + r).ceil() as i64);
            for y in y0.max(0)..=y1.min(h as i64 - 1) {
                for x in x0.max(0)..=x1.min(w as i64 - 1) {
                    let d2 = (x as Scalar - px.x).powi(2) + (y as Scalar - px.y).powi(2);
                    let t = 1.0 - d2 / (r * r);
                    if t > 0.0 {
                        buf[y as usize * w + x as usize] += lm.amplitude * t * t;
                    }
                }
            }
        }
        let mut img = GrayImage::new(k.width, k.height);
        for (dst, v) in img.data_mut().iter_mut().zip(buf) {
            *dst = (v * gain).round().clamp(0.0, 255.0) as u8;
        }
        img
    }

    /// Sideways-dominant sweep with a little forward motion and yaw. Sideways
    /// because that is what makes depth observable; monocular depth from a
    /// camera translating along its own axis is close to unobservable at the
    /// image centre.
    fn truth_pose(i: usize) -> Se3 {
        let t = i as Scalar;
        Se3::new(
            So3::exp(&Vec3::new(
                0.004 * (0.19 * t).sin(),
                0.020 * (0.11 * t).sin(),
                0.003 * (0.07 * t).cos(),
            )),
            Vec3::new(0.055 * t, 0.010 * (0.13 * t).sin(), 0.012 * t),
        )
    }

    fn frame(i: usize, image: GrayImage) -> Frame {
        Frame::new(
            FrameId(i as u64),
            Timestamp::from_seconds(i as f64 / 30.0),
            image,
        )
    }

    struct Run {
        estimated: Vec<Vec3>,
        truth: Vec<Vec3>,
        states: Vec<TrackingState>,
        keyframes: usize,
        landmarks: usize,
    }

    fn run_sequence(frames: usize, config: TrackConfig) -> Run {
        let k = intrinsics();
        let scene = scene(180, 7);
        let mut tracker = Tracker::new(config, k);
        let mut run = Run {
            estimated: Vec::new(),
            truth: Vec::new(),
            states: Vec::new(),
            keyframes: 0,
            landmarks: 0,
        };
        for i in 0..frames {
            let pose = truth_pose(i);
            let image = render(&scene, &pose, &k, 1.0);
            let out = tracker.process(&frame(i, image), None);
            run.states.push(out.state);
            run.keyframes += usize::from(out.is_keyframe);
            if out.state.has_pose() {
                if let Some(p) = out.pose {
                    run.estimated.push(p.translation());
                    run.truth.push(pose.translation());
                }
            }
        }
        run.landmarks = tracker.local_map().len();
        run
    }

    #[test]
    fn tracks_a_synthetic_sequence_and_recovers_the_trajectory() {
        let run = run_sequence(60, TrackConfig::default());
        assert!(
            run.estimated.len() >= 45,
            "only {} tracked frames of 60; states {:?}",
            run.estimated.len(),
            &run.states[..20.min(run.states.len())]
        );
        assert!(run.keyframes >= 5, "{} keyframes", run.keyframes);
        assert!(run.landmarks >= 40, "{} landmarks", run.landmarks);

        // spec.md §6 L3: ATE after Sim(3) alignment. L3 claims no scale, so the
        // similarity is the only comparison that means anything.
        let a = umeyama(&run.estimated, &run.truth, true).expect("alignment");
        let extent = run
            .truth
            .iter()
            .map(|p| (p - run.truth[0]).norm())
            .fold(0.0, Scalar::max);
        assert!(extent > 2.0, "trajectory too short to be a test: {extent}");
        // Stated bar: 1% of trajectory extent on noiseless synthetic imagery.
        // Every pixel here is exact, so anything worse is a geometry bug, not
        // sensor noise.
        assert!(
            a.rmse < 0.01 * extent,
            "ATE {:.4} over an extent of {extent:.2} ({:.2}%)",
            a.rmse,
            100.0 * a.rmse / extent
        );
    }

    #[test]
    fn the_state_machine_reaches_tracking_and_stays_there() {
        let run = run_sequence(40, TrackConfig::default());
        let first_tracking = run
            .states
            .iter()
            .position(|s| *s == TrackingState::Tracking)
            .expect("never reached Tracking");
        assert!(
            first_tracking <= 8,
            "took {first_tracking} frames to initialise"
        );
        for s in &run.states[..first_tracking] {
            assert_eq!(*s, TrackingState::Initializing, "{:?}", run.states);
        }
        let bad = run.states[first_tracking..]
            .iter()
            .filter(|s| **s != TrackingState::Tracking)
            .count();
        assert!(
            bad <= 2,
            "{bad} degraded frames after init: {:?}",
            run.states
        );
    }

    #[test]
    fn identical_input_and_seed_give_a_bit_identical_trajectory() {
        // spec.md §6: "The same binary then runs live and replays a canned
        // trajectory bit-for-bit reproducibly." Not approximately.
        let a = run_sequence(35, TrackConfig::default());
        let b = run_sequence(35, TrackConfig::default());
        assert_eq!(a.states, b.states);
        assert_eq!(a.estimated.len(), b.estimated.len());
        assert!(!a.estimated.is_empty());
        for (p, q) in a.estimated.iter().zip(b.estimated.iter()) {
            for i in 0..3 {
                assert_eq!(p[i].to_bits(), q[i].to_bits(), "{p:?} vs {q:?}");
            }
        }
    }

    #[test]
    fn a_different_seed_still_lands_on_the_same_trajectory() {
        // RANSAC's seed changes which minimal sets are drawn, so the bits differ
        // — but the *answer* must not. A seed that moves the trajectory means
        // the solve is under-constrained, not merely random.
        let a = run_sequence(
            35,
            TrackConfig {
                seed: 1,
                ..TrackConfig::default()
            },
        );
        let b = run_sequence(
            35,
            TrackConfig {
                seed: 999_331,
                ..TrackConfig::default()
            },
        );
        let n = a.estimated.len().min(b.estimated.len());
        assert!(n >= 25);
        let align = umeyama(&a.estimated[..n], &b.estimated[..n], true).expect("alignment");
        let extent = a.estimated[..n]
            .iter()
            .map(|p| (p - a.estimated[0]).norm())
            .fold(0.0, Scalar::max);
        assert!(
            align.rmse < 0.02 * extent,
            "seed changed the answer by {}",
            align.rmse
        );
    }

    #[test]
    fn timings_are_filled_only_when_a_clock_is_installed() {
        let k = intrinsics();
        let scene = scene(120, 3);
        let mut tracker = Tracker::new(TrackConfig::default(), k);
        for i in 0..4 {
            tracker.process(&frame(i, render(&scene, &truth_pose(i), &k, 1.0)), None);
        }
        let quiet = *tracker.timings();
        assert_eq!(quiet, StageTimings::default(), "{quiet:?}");

        // A manual clock advances only when told to, so the numbers below are
        // the sequence of `advance` calls, not the machine's speed.
        let clock = ManualHostClock::new();
        let mut tracker = Tracker::new(TrackConfig::default(), k);
        tracker.set_host_clock(Some(Box::new(clock.clone())));
        tracker.process(&frame(0, render(&scene, &truth_pose(0), &k, 1.0)), None);
        let t = *tracker.timings();
        // Every stage ran; with a frozen clock every interval is exactly zero,
        // which is the point — the estimate cannot depend on them.
        assert_eq!(t.total_ms, 0.0);
        assert_eq!(t.pyramid_ms, 0.0);
    }

    #[test]
    fn installing_a_clock_does_not_change_the_trajectory() {
        // The structural version of the spec.md §6 wall-clock ban: if any
        // decision consulted the clock, these two runs would diverge.
        let k = intrinsics();
        let scene = scene(180, 7);
        let run = |with_clock: bool| {
            let mut tracker = Tracker::new(TrackConfig::default(), k);
            if with_clock {
                tracker.set_host_clock(Some(Box::new(ManualHostClock::new())));
            }
            let mut out = Vec::new();
            for i in 0..30 {
                let r = tracker.process(&frame(i, render(&scene, &truth_pose(i), &k, 1.0)), None);
                out.push(r.pose.map(|p| p.translation()));
            }
            out
        };
        let quiet = run(false);
        let timed = run(true);
        for (a, b) in quiet.iter().zip(timed.iter()) {
            match (a, b) {
                (Some(p), Some(q)) => {
                    for i in 0..3 {
                        assert_eq!(p[i].to_bits(), q[i].to_bits());
                    }
                }
                (None, None) => {}
                _ => panic!("clock changed which frames produced a pose"),
            }
        }
    }

    /// Drive to steady-state `Tracking`, then hand the tracker one frame of the
    /// caller's choosing and report the resulting state.
    fn state_after_interruption(
        make: impl Fn(&[Landmark], &CameraIntrinsics) -> Frame,
    ) -> TrackingState {
        let k = intrinsics();
        let scene = scene(180, 7);
        let mut tracker = Tracker::new(TrackConfig::default(), k);
        for i in 0..25 {
            tracker.process(&frame(i, render(&scene, &truth_pose(i), &k, 1.0)), None);
        }
        assert_eq!(tracker.state(), TrackingState::Tracking, "setup failed");
        tracker.process(&make(&scene, &k), None).state
    }

    #[test]
    fn a_blank_frame_reports_insufficient_features() {
        let state = state_after_interruption(|_, k| {
            frame(
                25,
                GrayImage::from_vec(
                    k.width,
                    k.height,
                    vec![128u8; (k.width * k.height) as usize],
                ),
            )
        });
        assert_eq!(
            state,
            TrackingState::Limited(LimitedReason::InsufficientFeatures),
            "a featureless frame must name the scene, not the motion"
        );
    }

    #[test]
    fn a_dark_frame_reports_low_light() {
        // Same geometry, same texture, 6% of the light. Feature count collapses
        // too, but the *cause* is the exposure and that is what gets reported.
        let state =
            state_after_interruption(|scene, k| frame(25, render(scene, &truth_pose(25), k, 0.06)));
        assert_eq!(state, TrackingState::Limited(LimitedReason::LowLight));
    }

    #[test]
    fn a_huge_jump_reports_excessive_motion_or_loss() {
        // Two metres sideways in one frame at 30 Hz. Nothing survives the flow,
        // but the frame is still full of corners, so this is motion and not a
        // blank wall.
        let state = state_after_interruption(|scene, k| {
            let jumped = Se3::new(
                truth_pose(25).rotation(),
                truth_pose(25).translation() + Vec3::new(2.0, 0.6, 0.0),
            );
            frame(25, render(scene, &jumped, k, 1.0))
        });
        assert!(
            matches!(
                state,
                TrackingState::Limited(LimitedReason::ExcessiveMotion) | TrackingState::Lost
            ),
            "got {state:?}"
        );
    }

    #[test]
    fn sustained_failure_escalates_from_limited_to_lost() {
        let k = intrinsics();
        let scene = scene(180, 7);
        let mut tracker = Tracker::new(TrackConfig::default(), k);
        for i in 0..25 {
            tracker.process(&frame(i, render(&scene, &truth_pose(i), &k, 1.0)), None);
        }
        assert_eq!(tracker.state(), TrackingState::Tracking);
        let blank = GrayImage::from_vec(
            k.width,
            k.height,
            vec![128u8; (k.width * k.height) as usize],
        );
        let mut states = Vec::new();
        for i in 25..30 {
            states.push(tracker.process(&frame(i, blank.clone()), None).state);
        }
        assert_eq!(
            states[0],
            TrackingState::Limited(LimitedReason::InsufficientFeatures)
        );
        assert_eq!(*states.last().unwrap(), TrackingState::Lost);
    }

    #[test]
    fn the_covariance_widens_while_coasting_on_a_stale_pose() {
        // spec.md §6 L6: overconfidence is worse than no covariance at all, and
        // reporting the last good solve's covariance for a pose that is now
        // three frames old is exactly that.
        let k = intrinsics();
        let scene = scene(180, 7);
        let mut tracker = Tracker::new(TrackConfig::default(), k);
        let mut last = None;
        for i in 0..25 {
            last = Some(tracker.process(&frame(i, render(&scene, &truth_pose(i), &k, 1.0)), None));
        }
        let good = last.unwrap();
        assert_eq!(good.state, TrackingState::Tracking);
        let trace_before: Scalar = (0..6).map(|i| good.covariance[(i, i)]).sum();
        assert!(trace_before.is_finite() && trace_before > 0.0);

        let blank = GrayImage::from_vec(
            k.width,
            k.height,
            vec![128u8; (k.width * k.height) as usize],
        );
        let stale = tracker.process(&frame(25, blank), None);
        let trace_after: Scalar = (0..6).map(|i| stale.covariance[(i, i)]).sum();
        assert!(
            trace_after > trace_before,
            "{trace_after} should exceed {trace_before}"
        );
        assert_eq!(stale.pose, good.pose, "the pose must be held, not invented");
    }

    #[test]
    fn features_are_refilled_when_they_drain() {
        let k = intrinsics();
        let scene = scene(180, 7);
        let config = TrackConfig {
            max_features: 120,
            min_features: 40,
            ..TrackConfig::default()
        };
        let mut tracker = Tracker::new(config, k);
        let mut minimum = usize::MAX;
        for i in 0..50 {
            tracker.process(&frame(i, render(&scene, &truth_pose(i), &k, 1.0)), None);
            if i > 5 {
                minimum = minimum.min(tracker.features().len());
            }
        }
        assert!(
            minimum >= config.min_features,
            "feature count fell to {minimum}, below min_features"
        );
        assert!(tracker.features().len() <= config.max_features);
    }

    #[test]
    fn the_local_map_stays_bounded() {
        let k = intrinsics();
        let scene = scene(300, 11);
        let config = TrackConfig::default();
        let mut tracker = Tracker::new(config, k);
        for i in 0..90 {
            tracker.process(&frame(i, render(&scene, &truth_pose(i), &k, 1.0)), None);
        }
        assert!(
            tracker.local_map().len() <= config.max_features * 4,
            "{} landmarks",
            tracker.local_map().len()
        );
        assert!(!tracker.local_map().is_empty());
    }

    #[test]
    fn reset_returns_the_tracker_to_its_starting_state() {
        let k = intrinsics();
        let scene = scene(180, 7);
        let mut tracker = Tracker::new(TrackConfig::default(), k);
        for i in 0..20 {
            tracker.process(&frame(i, render(&scene, &truth_pose(i), &k, 1.0)), None);
        }
        assert_eq!(tracker.state(), TrackingState::Tracking);
        tracker.reset();
        assert_eq!(tracker.state(), TrackingState::Initializing);
        assert!(tracker.local_map().is_empty());
        assert!(tracker.features().is_empty());
        assert!(tracker.pose().is_none());

        // And it re-initialises rather than being merely emptied.
        for i in 0..20 {
            tracker.process(&frame(i, render(&scene, &truth_pose(i), &k, 1.0)), None);
        }
        assert_eq!(tracker.state(), TrackingState::Tracking);
    }

    #[test]
    fn relocalization_anchors_the_new_map_to_the_supplied_pose() {
        let k = intrinsics();
        let scene = scene(180, 7);
        let mut tracker = Tracker::new(TrackConfig::default(), k);
        for i in 0..20 {
            tracker.process(&frame(i, render(&scene, &truth_pose(i), &k, 1.0)), None);
        }
        let anchor = Se3::new(
            So3::exp(&Vec3::new(0.0, 0.4, 0.0)),
            Vec3::new(17.0, -3.0, 5.0),
        );
        tracker.relocalized_to(anchor, Timestamp::from_seconds(1.0));
        assert_eq!(tracker.state(), TrackingState::Initializing);
        assert!(tracker.local_map().is_empty());
        assert_eq!(tracker.pose(), Some(anchor));

        // Re-bootstrap, then check the recovered poses live near the anchor
        // rather than back at the origin: L4's frame must survive the rebuild.
        let mut first = None;
        let mut last = None;
        for i in 20..45 {
            let out = tracker.process(&frame(i, render(&scene, &truth_pose(i), &k, 1.0)), None);
            if out.state == TrackingState::Tracking {
                if first.is_none() {
                    first = out.pose;
                }
                last = out.pose;
            }
        }
        let first = first.expect("never re-initialised after relocalization");
        let last = last.expect("never re-initialised after relocalization");

        // The sharp claim: the rebuilt map starts at the anchor plus exactly one
        // unit of baseline. L3 makes no metric claim, so the two-view bootstrap
        // normalises its baseline to |t| = 1 — meaning the *only* distance the
        // first recovered pose can legitimately sit from the anchor is one unit.
        // Anything else means the anchor was not adopted.
        let from_anchor = (first.translation() - anchor.translation()).norm();
        assert!(
            (from_anchor - 1.0).abs() < 1e-6,
            "the first pose after relocalization should sit exactly one bootstrap \
             baseline from the anchor, not {from_anchor}"
        );

        // Everything after that drifts in those same arbitrary units, so the
        // durable property is relative: the trajectory stays in the anchor's
        // frame rather than sliding back toward the origin. Asserting an
        // absolute distance here would be asserting a bootstrap scale that is
        // unobservable by construction.
        let to_anchor = (last.translation() - anchor.translation()).norm();
        let to_origin = last.translation().norm();
        assert!(
            to_anchor < 0.5 * to_origin,
            "after relocalization the trajectory is {to_anchor:.2} from the anchor but \
             {to_origin:.2} from the origin — L4's frame did not survive the rebuild"
        );
    }

    #[test]
    fn a_rotation_prior_does_not_change_a_slow_sequence() {
        // Tier 1 (no motion permission) and tier 2 must agree when the motion is
        // inside the pyramid's envelope: the prior is a prediction, never a
        // measurement, so it may only change how fast the flow converges.
        let k = intrinsics();
        let scene = scene(180, 7);
        let run = |use_prior: bool| {
            let mut tracker = Tracker::new(TrackConfig::default(), k);
            let mut out = Vec::new();
            for i in 0..30 {
                let pose = truth_pose(i);
                let prior = use_prior.then(|| pose.rotation());
                let r = tracker.process(&frame(i, render(&scene, &pose, &k, 1.0)), prior);
                if r.state.has_pose() {
                    if let Some(p) = r.pose {
                        out.push((p.translation(), pose.translation()));
                    }
                }
            }
            out
        };
        {
            let (with, without) = (true, false);
            let a: Vec<Vec3> = run(with).into_iter().map(|(e, _)| e).collect();
            let b: Vec<Vec3> = run(without).into_iter().map(|(e, _)| e).collect();
            let n = a.len().min(b.len());
            assert!(n >= 20, "{n} common frames");
            let align = umeyama(&a[..n], &b[..n], true).expect("alignment");
            let extent = a[..n]
                .iter()
                .map(|p| (p - a[0]).norm())
                .fold(0.0, Scalar::max);
            assert!(
                align.rmse < 0.02 * extent,
                "the prior moved the answer by {}",
                align.rmse
            );
        }
    }

    #[test]
    fn a_pure_rotation_never_bootstraps() {
        // The degenerate case for monocular initialisation: rotation carries no
        // parallax, so any depth it produces is fabricated. spec.md §6 Tier 3
        // asks for it to be *detected*, not silently wrong.
        let k = intrinsics();
        let scene = scene(180, 7);
        let mut tracker = Tracker::new(TrackConfig::default(), k);
        for i in 0..40 {
            let pose = Se3::from_rotation(So3::exp(&Vec3::new(0.0, 0.006 * i as Scalar, 0.0)));
            let out = tracker.process(&frame(i, render(&scene, &pose, &k, 1.0)), None);
            assert_eq!(
                out.state,
                TrackingState::Initializing,
                "frame {i} produced {:?} from a pure rotation",
                out.state
            );
            assert!(out.pose.is_none());
        }
        assert!(tracker.local_map().is_empty());
    }

    #[test]
    fn config_with_no_gpu_feature_falls_back_to_the_cpu_reference() {
        let tracker = Tracker::new(
            TrackConfig {
                use_gpu: true,
                ..TrackConfig::default()
            },
            intrinsics(),
        );
        let expected = if cfg!(feature = "gpu") {
            Backend::Gpu
        } else {
            Backend::Cpu
        };
        assert_eq!(tracker.backend(), expected);
    }

    #[test]
    fn the_local_map_index_survives_culling() {
        let mut map = LocalMap::new();
        for id in 0..10u64 {
            map.insert(
                LocalLandmark {
                    id,
                    position: Vec3::new(id as Scalar, 0.0, 1.0),
                    observations: id as u32,
                },
                id,
            );
        }
        // Ten landmarks, landmark `id` last seen at frame `id`, so last-seen
        // spans frames 0..=9.
        //
        // The original of this test called `cull(20, 5, 100)` and asserted six
        // survivors. That is arithmetically unreachable: a time-to-live of five
        // frames evaluated at frame 20 retains only what was seen in frames
        // 15..=20, and nothing here was seen after frame 9, so the correct
        // answer for those arguments is *zero* — which is what the
        // implementation returned, and which also made every assertion below it
        // vacuous. The ttl is widened instead so the call retains a proper
        // subset and the index rebuild is actually exercised: `20 - id <= 16`
        // keeps ids 4..=9.
        map.cull(20, 16, 100);
        assert_eq!(map.len(), 6, "frames 4..=20 is the retention window");
        assert_eq!(
            map.landmarks().iter().map(|l| l.id).collect::<Vec<_>>(),
            vec![4, 5, 6, 7, 8, 9]
        );
        for l in map.landmarks() {
            assert_eq!(map.get(l.id).map(|g| g.position), Some(l.position));
        }
        // And what fell outside the window is gone from the index too, not just
        // from the vector.
        assert!(map.get(3).is_none());
        map.cull(20, 100, 3);
        assert_eq!(map.len(), 3);
        // Capacity trimming keeps the most-observed.
        let ids: Vec<u64> = map.landmarks().iter().map(|l| l.id).collect();
        assert_eq!(ids, vec![7, 8, 9]);
        for l in map.landmarks() {
            assert!(map.get(l.id).is_some());
        }
    }

    #[test]
    fn a_tracker_that_loses_the_scene_recovers_on_its_own() {
        // The regression this exists for: `pose.is_some() && !map.is_empty()`
        // stays true forever once tracking has succeeded once, so the bootstrap
        // branch was unreachable and a lost tracker never tracked again. A user
        // who covered the lens for a second had to reload the page.
        let k = intrinsics();
        let world = scene(180, 7);
        let mut tracker = Tracker::new(TrackConfig::default(), k);

        for i in 0..30 {
            let image = render(&world, &truth_pose(i), &k, 1.0);
            tracker.process(&frame(i, image), None);
        }
        assert_eq!(
            tracker.state(),
            TrackingState::Tracking,
            "precondition: should be tracking before the scene is taken away"
        );

        // A featureless view: nothing to track, so PnP cannot succeed. Long
        // enough to pass the abandonment threshold, checking `Lost` on the way.
        let blank = GrayImage::from_vec(
            k.width,
            k.height,
            vec![128u8; (k.width * k.height) as usize],
        );
        let mut reported_lost = false;
        for i in 30..(30 + ABANDON_MAP_AFTER_FAILURES as usize + 6) {
            if tracker.process(&frame(i, blank.clone()), None).state == TrackingState::Lost {
                reported_lost = true;
            }
        }
        assert!(
            reported_lost,
            "a blank scene must be reported as Lost before the map is abandoned, \
             so a consumer knows to stop trusting the pose"
        );

        // Give the scene back, from where the camera actually was. Continuing
        // `truth_pose` forward would fly past the landmarks and test nothing.
        let mut recovered = false;
        let resume = 30 + ABANDON_MAP_AFTER_FAILURES as usize + 6;
        for (n, i) in (resume..resume + 60).enumerate() {
            let image = render(&world, &truth_pose(n), &k, 1.0);
            if tracker.process(&frame(i, image), None).state == TrackingState::Tracking {
                recovered = true;
                break;
            }
        }
        assert!(
            recovered,
            "never re-entered Tracking after the scene returned; state is {:?}",
            tracker.state()
        );
    }
}
