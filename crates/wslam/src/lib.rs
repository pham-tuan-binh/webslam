//! # wslam — orchestration
//!
//! The only crate aware of every layer. It owns the state machine, the
//! frontend/backend split, and the composition of uncertainty from L3 through
//! L5 into the [`Pose`] that L6 emits.
//!
//! Nothing here implements an algorithm. If you find estimation logic in this
//! crate it belongs in a layer, and the layering exists so the compiler can
//! tell you (spec.md §7).
//!
//! ## One frame
//!
//! ```text
//! push_motion ──► TimeBase.map_motion ──► L1 integrate ──► StateWindow
//!                                              │
//! push_frame ──► TimeBase.map_camera           │ rotation prior
//!      │                                       ▼
//!      └────────────────────────────────► L3 tracker.process
//!                                              │ pose (up to scale) + covariance
//!                          ┌───────────────────┼───────────────────┐
//!                          ▼                   ▼                   ▼
//!                 L2 focal (during init)   L4 keyframe/reloc   L5 scale.estimate
//!                                                                  │
//!                                                                  ▼
//!                                                        L6 Pose::with_scale
//! ```
//!
//! ## The frontend never blocks on the backend
//!
//! The default build is single-threaded so it can be embedded anywhere
//! (docs/DECISIONS.md D2), which makes the split a *budget* rather than a
//! thread boundary: [`MapConfig::backend_budget_ms`] bounds backend work per
//! frame and the remainder resumes next frame. Relocalization is exempt — it is
//! one BoW query plus one PnP, and the user is standing there waiting.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod config;
mod debug;
mod events;

pub use config::{IntrinsicsConfig, MapConfig, SensorTier, SlamConfig};
pub use debug::{DebugFeature, DebugKeyframe, DebugPoseGraphEdge, DebugSnapshot};
pub use events::SlamEvent;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use wslam_core::{
    CameraIntrinsics, DeterministicRng, Error, Frame, ImuSample, MotionEvent, PassthroughTimeBase,
    Pose, Result, Scalar, ScaleEstimate, ScaleKind, Se3, So3, StateWindow, TimeBase, Timestamp,
    TrackingState, Vec2, WindowSample,
};
use wslam_map::{
    describe, fast_keypoints, serialize_map, BinaryDescriptor, Keyframe, KeyframeId, MapDb,
    PoseGraph, RelocConfig, Relocalizer, SolverConfig, Vocabulary,
};
use wslam_orientation::OrientationFilter;
use wslam_scale::{NoneScale, ScaleSource};
use wslam_track::{FeatureState, Tracker};

/// Re-exported so a caller can name a scale source without depending on L5
/// directly.
pub mod scale {
    pub use wslam_scale::*;
}

/// Re-exported layer types a caller is likely to configure.
pub mod layers {
    pub use wslam_calib as calib;
    #[cfg(feature = "tight-vi")]
    pub use wslam_clock as clock;
    pub use wslam_map as map;
    pub use wslam_orientation as orientation;
    pub use wslam_track as track;
}

/// A running session.
pub struct WebSlam {
    config: SlamConfig,

    // L0 — the timebase. Boxed because tier 3 swaps in a fitted model and the
    // rest of the pipeline must not care which it got.
    timebase: Box<dyn TimeBase>,

    // L1
    orientation: OrientationFilter,
    motion_available: bool,
    effective_tier: SensorTier,

    // L2
    focal: Option<wslam_calib::FocalEstimator>,
    intrinsics: CameraIntrinsics,
    intrinsics_locked: bool,
    previous_features: HashMap<u64, Vec2>,
    previous_frame_time: Option<Timestamp>,
    /// `R_world_camera` at the previous frame, for L2's relative rotation.
    previous_camera_attitude: Option<So3>,
    /// Frames for which L1 supplied a prior. See `frames_with_rotation_prior`.
    frames_with_prior: u64,
    /// Frames where L1 had an attitude at all, gated or not.
    frames_with_attitude: u64,
    /// Current coordinate-frame epoch; see `Pose::frame_epoch`.
    frame_epoch: u32,
    /// Set by a verified relocalization, so the bootstrap it enables does not
    /// count as a new frame.
    relocalized_since_loss: bool,

    // L3
    tracker: Tracker,

    // L4
    map: Option<MapDb>,
    relocalizer: Relocalizer,
    pose_graph: PoseGraph,
    keyframe_times: Vec<(KeyframeId, Timestamp)>,
    /// Tracker landmark id -> stable map landmark id.
    ///
    /// The tracker's ids are local to its current map and restart on reset; the
    /// map's must be stable for covisibility to mean anything. Without this
    /// indirection the same physical point becomes a fresh map landmark at
    /// every keyframe and no two keyframes ever share one.
    landmark_ids: std::collections::HashMap<u64, wslam_map::LandmarkId>,
    lost_since: Option<u64>,
    backend_queue: VecDeque<BackendJob>,

    // L5
    scale_source: Box<dyn ScaleSource>,
    scale: ScaleEstimate,
    window: StateWindow,

    // Session state
    rng: DeterministicRng,
    frames: VecDeque<Frame>,
    frame_counter: u64,
    latest: Option<Pose>,
    state: TrackingState,
    session_start: Option<Timestamp>,
    events: Vec<SlamEvent>,
    trajectory: Vec<wslam_core::Vec3>,
    /// Each emitted pose stored *relative to the keyframe it was tracked
    /// against*, with that keyframe's id and the frame's timestamp.
    ///
    /// This is ORB-SLAM2's `mlRelativeFramePoses`, and it exists for one
    /// reason: a loop closure corrects **keyframe** poses, and without a
    /// relative representation there is no way for that correction to reach the
    /// per-frame trajectory that was already emitted. Measured: 30 loop
    /// closures were accepted and the reported ATE did not change by a single
    /// digit, because the corrections landed in the pose graph and nowhere
    /// else.
    relative_poses: Vec<(Timestamp, KeyframeId, Se3)>,
    rejected_loops: Vec<DebugPoseGraphEdge>,
}

/// Deferred backend work, executed within the per-frame budget.
enum BackendJob {
    /// Cull the map back under its keyframe ceiling.
    Cull,
    /// Look for a loop closure from this keyframe.
    LoopSearch(KeyframeId),
    /// Re-optimise the pose graph.
    Optimize,
}

impl WebSlam {
    /// Start a session.
    ///
    /// # Errors
    /// - [`Error::SensorTier`] when the configuration asks for tier 3 without
    ///   the `tight-vi` feature compiled in. spec.md §3 specifies that
    ///   `ScaleSource.inertial()` *"throws if L0 unavailable"* rather than
    ///   silently degrading, and the same applies to the tier itself.
    /// - [`Error::Config`] for an impossible image size.
    pub fn new(config: SlamConfig) -> Result<Self> {
        Self::with_scale_source(config, Box::new(NoneScale::new()))
    }

    /// Start a session with an explicit scale source.
    ///
    /// The default is [`NoneScale`] — up to scale, and honest about it. Passing
    /// anything else is the caller declaring which ruler they accept
    /// (spec.md §1).
    pub fn with_scale_source(
        config: SlamConfig,
        scale_source: Box<dyn ScaleSource>,
    ) -> Result<Self> {
        if config.width == 0 || config.height == 0 {
            return Err(Error::Config(format!(
                "image size must be non-zero, got {}x{}",
                config.width, config.height
            )));
        }

        #[cfg(not(feature = "tight-vi"))]
        if config.tier == SensorTier::TightVisualInertial {
            return Err(Error::SensorTier {
                required: 3,
                reason: "this build was compiled without the `tight-vi` feature, so L0 \
                         (the clock layer) is absent. Rebuild with --features tight-vi, \
                         or use tier 2 and a non-inertial ScaleSource."
                    .into(),
            });
        }

        let config = config.normalized();
        let intrinsics = config.initial_intrinsics();

        let timebase: Box<dyn TimeBase> = {
            #[cfg(feature = "tight-vi")]
            {
                if config.tier == SensorTier::TightVisualInertial {
                    Box::new(wslam_clock::FittedTimeBase::new(
                        wslam_clock::ClockConfig::default(),
                    ))
                } else {
                    Box::new(PassthroughTimeBase::new())
                }
            }
            #[cfg(not(feature = "tight-vi"))]
            {
                Box::new(PassthroughTimeBase::new())
            }
        };

        let focal = config.needs_intrinsics_estimation().then(|| {
            wslam_calib::FocalEstimator::new(
                config.intrinsics.calib.clone(),
                config.width,
                config.height,
            )
        });

        let map = config.map.enabled.then(|| {
            // NOTE: an *empty* vocabulary supports nothing at all.
            // `Vocabulary::transform` early-returns `BowVector::empty()` when
            // it has no words, so every keyframe gets an empty bag, every
            // place-recognition query returns nothing, and loop closure is
            // structurally dead — 0 candidates proposed over a 3682-frame
            // sequence with real revisits. Use `set_vocabulary` with a trained
            // artifact to turn L4 on.
            MapDb::new(Arc::new(Vocabulary::empty()))
        });

        Ok(WebSlam {
            timebase,
            orientation: OrientationFilter::new(config.orientation),
            motion_available: false,
            effective_tier: config.tier,
            focal,
            intrinsics,
            intrinsics_locked: config.intrinsics.known.is_some(),
            previous_features: HashMap::new(),
            previous_frame_time: None,
            previous_camera_attitude: None,
            frames_with_prior: 0,
            frames_with_attitude: 0,
            frame_epoch: 0,
            relocalized_since_loss: false,
            tracker: Tracker::new(config.track, intrinsics),
            map,
            relocalizer: Relocalizer::new(RelocConfig::default()),
            pose_graph: PoseGraph::new(),
            keyframe_times: Vec::new(),
            landmark_ids: std::collections::HashMap::new(),
            lost_since: None,
            backend_queue: VecDeque::new(),
            scale_source,
            scale: ScaleEstimate::unscaled(),
            window: StateWindow::with_default_capacity(),
            rng: DeterministicRng::new("wslam-session", config.seed),
            frames: VecDeque::new(),
            frame_counter: 0,
            latest: None,
            state: TrackingState::Initializing,
            session_start: None,
            events: Vec::new(),
            trajectory: Vec::new(),
            relative_poses: Vec::new(),
            rejected_loops: Vec::new(),
            config,
        })
    }

    /// Load a previously saved map and adopt its scale anchor.
    ///
    /// The session will relocalize into it, and once it does, poses are metric
    /// in the anchor's units. The reported variance is the anchor's **plus**
    /// relocalization error — spec.md §4 L5.
    pub fn load_map(&mut self, bytes: &[u8]) -> Result<()> {
        let (db, _vocab) = wslam_map::deserialize_map(bytes)?;
        let anchor = db.scale_anchor();
        log::info!(
            "loaded map: {} keyframes, {} landmarks, anchor {} (var {:.3e})",
            db.keyframe_count(),
            db.landmark_count(),
            anchor.source.as_str(),
            anchor.variance
        );
        if anchor.source.is_metric() {
            self.scale_source = Box::new(wslam_scale::MapScale::new(
                anchor,
                self.config.map.reloc_scale_variance,
            ));
        }
        self.map = Some(db);
        Ok(())
    }

    /// Feed one camera frame, exactly as the shim delivered it.
    ///
    /// `media_time` is `VideoFrameCallbackMetadata.mediaTime` in seconds.
    pub fn push_frame_raw(&mut self, media_time: f64, image: wslam_core::GrayImage) {
        let index = self.frame_counter;
        self.frame_counter += 1;
        let timestamp = self.timebase.map_camera(media_time, index);
        self.frames
            .push_back(Frame::new(wslam_core::FrameId(index), timestamp, image));
    }

    /// Feed a frame that already carries a timestamp — the replay path.
    pub fn push_frame(&mut self, frame: Frame) {
        self.frame_counter = self.frame_counter.max(frame.id.0 + 1);
        self.frames.push_back(frame);
    }

    /// Feed one motion event, exactly as the shim delivered it.
    ///
    /// Ignored at tier 1. Never buffered or smoothed here: L0 exists to measure
    /// the delivery jitter, and cleaning it up would destroy the signal
    /// (spec.md §7).
    pub fn push_motion(&mut self, event: MotionEvent) {
        if !self.config.tier.uses_motion() {
            return;
        }
        self.motion_available = true;
        let timestamp = self.timebase.map_motion(event.index, event.arrival_millis);
        let sample = ImuSample::new(timestamp, event.gyro, event.accel_with_gravity);
        self.orientation.integrate(&sample);
        self.window.push_imu(sample);
    }

    /// Feed a motion sample already in the unified timebase — the replay path.
    pub fn push_imu(&mut self, sample: ImuSample) {
        if !self.config.tier.uses_motion() {
            return;
        }
        self.motion_available = true;
        self.orientation.integrate(&sample);
        self.window.push_imu(sample);
    }

    /// Advance one frame. Returns the pose if one was produced.
    ///
    /// Returns `None` when no frame is queued, or when the tracker produced no
    /// usable pose this frame — which is a normal outcome during
    /// initialisation, loss and relocalization, not an error.
    pub fn step(&mut self) -> Option<Pose> {
        let frame = self.frames.pop_front()?;
        self.session_start.get_or_insert(frame.timestamp);

        // Tier demotion. spec.md §4 lists tier 1 as the fallback when motion
        // permission is denied, so this is a documented degradation rather than
        // a failure.
        if self.config.tier.uses_motion() && !self.motion_available {
            // Only complain once we have had a real chance to receive one. On
            // the first frame there has been no interval to sample over, so a
            // warning there is noise that trains people to ignore the real one.
            if self.effective_tier != SensorTier::VisionOnly && self.frame_counter > 1 {
                log::warn!("no motion samples received; running at sensor tier 1 (vision only)");
                self.effective_tier = SensorTier::VisionOnly;
            }
        } else {
            self.effective_tier = self.config.tier;
        }

        // L3 wants the absolute attitude; L2 wants the rotation *between* the
        // two frames it is given correspondences for. Both in the camera frame.
        let attitude = self.camera_attitude(frame.timestamp);
        if attitude.is_some() {
            self.frames_with_attitude += 1;
        }
        let inter_frame = match (attitude, self.previous_camera_attitude) {
            // R_cam_now_from_cam_prev, matching the (previous, current) order
            // of the correspondences handed to L2.
            (Some(now), Some(prev)) => Some(now.inverse().compose(&prev)),
            _ => None,
        };

        // Hand L3 the prior only when it predicts more motion than it is itself
        // wrong by. See `min_prior_rotation_rad`: below that threshold the
        // prediction is noise and measurably costs more frames than it saves.
        let useful = inter_frame
            .map(|d| d.angle() >= self.config.min_prior_rotation_rad)
            .unwrap_or(false);
        let prior = if useful { attitude } else { None };
        if prior.is_some() {
            self.frames_with_prior += 1;
        }
        let outcome = self.tracker.process(&frame, prior);

        self.update_intrinsics(inter_frame);
        self.previous_camera_attitude = attitude;
        // Advance the prior's anchor. Forgetting this left `rotation_prior`
        // returning `None` for the whole session, which silently disabled two
        // layers: L3 lost its motion prediction, and L2 never received a single
        // matched pair because `update_intrinsics` gates on the prior. Neither
        // showed up as an error — the tracker simply tracked worse and the focal
        // estimator simply never answered.
        self.previous_frame_time = Some(frame.timestamp);
        self.transition_to(outcome.state);

        // A bootstrap that no relocalization vouched for adopts a brand-new
        // scale and origin. Bumping the epoch is what stops the trajectory
        // being spliced across the discontinuity, and what lets the harness
        // report per-segment error instead of measuring the seams.
        if outcome.bootstrapped {
            if self.relocalized_since_loss {
                self.relocalized_since_loss = false;
            } else if self.frame_epoch > 0 || !self.trajectory.is_empty() {
                self.frame_epoch += 1;
                log::info!(
                    "new coordinate frame (epoch {}) — bootstrapped without relocalizing",
                    self.frame_epoch
                );
                self.events.push(SlamEvent::OriginReset {
                    at: frame.timestamp,
                    epoch: self.frame_epoch,
                });
            }
        }

        if outcome.is_keyframe {
            self.add_keyframe(&frame, outcome.pose);
        }
        if matches!(
            outcome.state,
            TrackingState::Lost | TrackingState::Relocalizing
        ) {
            self.try_relocalize(&frame);
        } else {
            self.lost_since = None;
        }

        self.run_backend();

        let pose_up_to_scale = outcome.pose?;
        self.window.push_pose(WindowSample {
            timestamp: frame.timestamp,
            pose: pose_up_to_scale,
            landmark_count: outcome.inlier_count,
        });
        self.trajectory.push(pose_up_to_scale.translation());
        // Anchor this frame to the most recent keyframe so a later correction
        // to that keyframe carries the frame with it.
        if let Some(&(reference, _)) = self.keyframe_times.last() {
            if let Some(anchor) = self.pose_graph.pose(reference) {
                self.relative_poses.push((
                    frame.timestamp,
                    reference,
                    anchor.inverse().compose(&pose_up_to_scale),
                ));
            }
        }

        // L5. A source that declines leaves the previous estimate in place
        // rather than reverting to unscaled — losing a fiducial from view does
        // not un-know its measurement.
        if let Some(estimate) = self.scale_source.estimate(&self.window) {
            let first = !self.scale.source.is_metric();
            self.scale = estimate;
            if first && estimate.source.is_metric() {
                self.events.push(SlamEvent::ScaleAcquired { estimate });
            }
        }

        let init_age_ms = self
            .session_start
            .map(|t0| frame.timestamp.since(t0) * 1000.0)
            .unwrap_or(0.0);

        let base = Pose {
            timestamp: frame.timestamp,
            transform: pose_up_to_scale,
            covariance: outcome.covariance,
            scale: ScaleEstimate::unscaled(),
            state: outcome.state,
            init_age_ms,
            frame_epoch: self.frame_epoch,
        };
        // `with_scale` is where the scale's own variance enters the position
        // covariance. Skipping that term is the standard way an estimator ends
        // up overconfident (spec.md §6 L6).
        let pose = base.with_scale(self.scale);
        self.latest = Some(pose);
        Some(pose)
    }

    /// Run until the frame queue drains, collecting every pose.
    ///
    /// The replay entry point. The push path in the browser calls
    /// [`WebSlam::step`] once per delivered frame instead.
    pub fn drain(&mut self) -> Vec<Pose> {
        let mut out = Vec::new();
        while !self.frames.is_empty() {
            if let Some(pose) = self.step() {
                out.push(pose);
            }
        }
        out
    }

    /// The freshest pose. Pull path, for renderers (spec.md §3).
    #[must_use]
    pub fn current_pose(&self) -> Option<Pose> {
        self.latest
    }

    /// Current tracking state.
    #[must_use]
    pub fn state(&self) -> TrackingState {
        self.state
    }

    /// The tier actually in force, which may be below the configured ceiling.
    #[must_use]
    pub fn effective_tier(&self) -> SensorTier {
        self.effective_tier
    }

    /// Frames on which L1's prediction was actually handed to L3.
    ///
    /// Lower than [`WebSlam::frames_with_attitude`] whenever the gate in
    /// `min_prior_rotation_rad` judged the predicted motion too small to be
    /// worth trusting. The two are deliberately separate: "L1 is not wired up"
    /// and "L1 is wired up and chose to stay out of the way" are different
    /// conditions and only the first is a bug.
    /// Per-stage timings for the most recent frame.
    #[must_use]
    pub fn stage_timings(&self) -> wslam_track::StageTimings {
        *self.tracker.timings()
    }

    /// The trajectory rebuilt from the *current* keyframe poses.
    ///
    /// **Known defective — do not use for evaluation yet.** With loop closure
    /// disabled entirely this must reduce to the identity transformation of the
    /// emitted trajectory, because `anchor.compose(anchor.inverse() * pose)` is
    /// exactly `pose` when the anchor has not moved. It does not: measured over
    /// 3682 frames with zero accepted loops, absolute error is unchanged
    /// (0.2914 m vs 0.2906 m) but inter-frame RPE degrades from 0.0147 m /
    /// 0.130 deg to 0.0313 m / 0.834 deg. Something perturbs a frame's anchor
    /// between recording and readout, and until that is found this path adds
    /// per-frame noise rather than removing drift.
    ///
    /// Every frame is re-expressed against its reference keyframe's pose as the
    /// pose graph holds it now, so loop closures and pose-graph optimisation
    /// are reflected in the whole history rather than only in frames that came
    /// after them. Frames whose reference keyframe has since been culled are
    /// dropped: there is nothing left to anchor them to, and emitting them at a
    /// stale pose would silently mix corrected and uncorrected segments.
    #[must_use]
    pub fn refined_trajectory(&self) -> Vec<(Timestamp, Se3)> {
        self.relative_poses
            .iter()
            .filter_map(|(t, reference, relative)| {
                let anchor = self.pose_graph.pose(*reference)?;
                Some((*t, anchor.compose(relative)))
            })
            .collect()
    }

    /// Install a GPU device acquired by the embedder.
    ///
    /// The browser only offers adapters asynchronously, so wasm callers must
    /// `await` a context and hand it in here before the first frame rather than
    /// letting the tracker acquire one lazily.
    #[cfg(feature = "gpu")]
    pub fn set_gpu_context(&mut self, context: wslam_track::GpuContext) {
        self.tracker.set_gpu_context(context);
    }

    /// Install a trained vocabulary, replacing the empty default.
    ///
    /// Must be called before the first frame: the bag-of-words vectors already
    /// stored in the map were computed against the old vocabulary and are not
    /// comparable with ones computed against a new one.
    ///
    /// # Errors
    /// If the session has already processed a frame.
    pub fn set_vocabulary(&mut self, vocabulary: Arc<Vocabulary>) -> Result<()> {
        if self.frame_counter > 0 {
            return Err(wslam_core::Error::Config(
                "vocabulary must be installed before the first frame".into(),
            ));
        }
        self.map = Some(MapDb::new(vocabulary));
        Ok(())
    }

    /// Why frames failed to produce a pose. See [`FailureCounts`].
    #[must_use]
    pub fn failures(&self) -> wslam_track::FailureCounts {
        self.tracker.failures()
    }

    /// Frames on which a rotation prior was available.
    #[must_use]
    pub fn frames_with_rotation_prior(&self) -> u64 {
        self.frames_with_prior
    }

    /// The current coordinate-frame epoch. See [`Pose::frame_epoch`].
    #[must_use]
    pub fn frame_epoch(&self) -> u32 {
        self.frame_epoch
    }

    /// Frames on which L1 had an attitude available at all.
    ///
    /// This is the wiring signal. Zero at tier 2 over a long session means L1
    /// is not reaching the orchestrator, which also disables L2.
    #[must_use]
    pub fn frames_with_attitude(&self) -> u64 {
        self.frames_with_attitude
    }

    /// Matched pairs L2 has accepted so far. Zero after a long tier-2 session
    /// with unknown intrinsics means L2 is not running.
    #[must_use]
    pub fn calibration_pairs(&self) -> usize {
        self.focal.as_ref().map_or(0, |f| f.pair_count())
    }

    /// L1's attitude in the **body** frame, `R_world_body`, if initialised.
    ///
    /// Exposed for scoring against a dataset's ground-truth orientation. This
    /// is the raw filter output, before the camera extrinsic is applied — which
    /// is the frame EuRoC's `state_groundtruth_estimate0` publishes in.
    #[must_use]
    pub fn body_attitude(&self) -> Option<So3> {
        self.orientation
            .is_initialized()
            .then(|| self.orientation.attitude())
    }

    /// Whether L2 is estimating intrinsics at all. `false` when the caller
    /// supplied a known calibration, which is the replay harness's case.
    #[must_use]
    pub fn intrinsics_are_estimated(&self) -> bool {
        self.focal.is_some()
    }

    /// Current intrinsics, refined by L2 if it ran.
    #[must_use]
    pub fn intrinsics(&self) -> CameraIntrinsics {
        self.intrinsics
    }

    /// Current scale estimate and its provenance.
    #[must_use]
    pub fn scale(&self) -> ScaleEstimate {
        self.scale
    }

    /// Take the events accumulated since the last call.
    pub fn take_events(&mut self) -> Vec<SlamEvent> {
        std::mem::take(&mut self.events)
    }

    /// Serialise the map.
    ///
    /// # Errors
    /// [`Error::Config`] when mapping is disabled — there is nothing to save,
    /// and returning empty bytes would let a caller persist a map that will
    /// never relocalize.
    pub fn save_map(&self) -> Result<Vec<u8>> {
        let db = self
            .map
            .as_ref()
            .ok_or_else(|| Error::Config("mapping is disabled; nothing to save".into()))?;
        let mut snapshot = db.clone();
        // The anchor is what makes a saved map a ScaleSource (spec.md §4 L4c),
        // so it travels with the bytes rather than being reconstructed.
        snapshot.set_scale_anchor(self.scale);
        Ok(serialize_map(&snapshot))
    }

    /// The unstable debug surface.
    #[must_use]
    pub fn debug(&self) -> DebugSnapshot<'_> {
        DebugSnapshot::new(self)
    }

    // -- internals ----------------------------------------------------------

    /// L1's attitude, expressed in the **camera** frame: `R_world_camera`.
    ///
    /// Two things this has to get right, both of which it originally got wrong:
    ///
    /// 1. **Absolute, not relative.** `Tracker::process` documents its argument
    ///    as "L1's attitude (`R_world_body`) at this frame" and internally
    ///    differences consecutive values to build the inter-frame rotation.
    ///    Handing it a delta made that difference meaningless.
    /// 2. **Camera frame, not body frame.** L1 works in the frame the inertial
    ///    sensor reports in. `R_world_camera = R_world_body * R_body_camera`.
    fn camera_attitude(&self, at: Timestamp) -> Option<So3> {
        if self.effective_tier == SensorTier::VisionOnly || !self.orientation.is_initialized() {
            return None;
        }
        // Interpolate to the frame's own timestamp, and return nothing when the
        // history does not reach it.
        //
        // Falling back to `attitude()` here — the state after the most recent
        // sample — looks harmless and is not. When the history has evicted the
        // span a frame sits in, every frame gets handed the *same* latest
        // attitude, consecutive differences collapse to identity, and the
        // caller sees a confident prior that encodes no motion at all. An
        // absent prior is honest and the gate handles it; a stale one is a
        // silent lie about the device having held still.
        let body = self.orientation.attitude_at(at)?;
        Some(body.compose(&self.config.body_from_camera))
    }

    /// Feed L2 from the tracker's own correspondences during initialisation.
    ///
    /// The focal estimator needs matched pairs plus the rotation between them,
    /// and the tracker already produces both — so L2 rides the init pan for
    /// free instead of demanding a separate calibration step from the user.
    fn update_intrinsics(&mut self, inter_frame: Option<So3>) {
        let features = self.tracker.features();
        let current: HashMap<u64, Vec2> = features
            .iter()
            .filter(|f| !matches!(f.state, FeatureState::Lost))
            .map(|f| (f.id, f.px))
            .collect();

        if !self.intrinsics_locked {
            if let (Some(estimator), Some(rotation)) = (self.focal.as_mut(), inter_frame) {
                let matches: Vec<(Vec2, Vec2)> = current
                    .iter()
                    .filter_map(|(id, px)| self.previous_features.get(id).map(|prev| (*prev, *px)))
                    .collect();
                if matches.len() >= 8 {
                    // Errors here are transient by construction — too little
                    // rotation, too few inliers — and the right response is to
                    // keep panning, not to surface anything to the caller.
                    let _ = estimator.push_pair(&matches, &rotation);
                }
                if let Some(estimate) = estimator.estimate() {
                    let refined = estimate.intrinsics(self.config.width, self.config.height);
                    if (refined.fx - self.intrinsics.fx).abs() > 0.5 {
                        log::info!(
                            "L2 focal {:.1} px (±{:.2}%), {} pairs",
                            estimate.focal_px,
                            estimate.relative_stddev_percent(),
                            estimate.pairs_used
                        );
                        self.intrinsics = refined;
                        self.tracker.set_intrinsics(refined);
                    }
                }
            }
        }

        self.previous_features = current;
    }

    fn transition_to(&mut self, next: TrackingState) {
        if next != self.state {
            self.events.push(SlamEvent::State {
                from: self.state,
                to: next,
            });
            self.state = next;
        }
    }

    /// Build a map keyframe from the current frame.
    fn add_keyframe(&mut self, frame: &Frame, pose: Option<Se3>) {
        let (Some(db), Some(pose)) = (self.map.as_mut(), pose) else {
            return;
        };
        let detections = fast_keypoints(&frame.image, 20, 400);
        if detections.len() < 16 {
            // Too few features to be recognisable later. Storing it would grow
            // the map without improving recall.
            return;
        }
        let (keypoints, angles): (Vec<Vec2>, Vec<Scalar>) = detections.into_iter().unzip();
        let descriptors: Vec<BinaryDescriptor> = describe(&frame.image, &keypoints, &angles);
        let bow = db.transform(&descriptors);

        // Associate each stored keypoint with the tracker landmark it sits on.
        //
        // Without this the keyframe carries descriptors that match nothing in
        // 3D, and every downstream consumer that needs point correspondences —
        // relocalization, loop closure, the Sim(3) solve — has nothing to work
        // with. `vec![None; n]` here is why L4 could recognise a place and then
        // do nothing about it.
        //
        // The tracker's landmark ids are local and are recycled across resets,
        // so `landmark_ids` maps them to stable map ids and keeps the mapping
        // for the life of the session; the same tracker landmark seen from two
        // keyframes must land on one map landmark or covisibility is empty.
        let mut landmarks = vec![None; keypoints.len()];
        let mut landmark_ids_seen: Vec<wslam_map::LandmarkId> = Vec::new();
        {
            // Squared radius for "this FAST corner is the same point as that
            // tracked feature". The detectors differ, so an exact match is not
            // available; 3 px is tight enough not to cross-associate at the
            // feature densities we run at.
            const ASSOC_RADIUS_SQ: Scalar = 9.0;
            let features = self.tracker.features();
            let local = self.tracker.local_map();
            for (i, kp) in keypoints.iter().enumerate() {
                let Some(feature) = features
                    .iter()
                    .filter(|f| f.landmark.is_some())
                    .filter(|f| (f.px - kp).norm_squared() <= ASSOC_RADIUS_SQ)
                    .min_by(|a, b| {
                        (a.px - kp)
                            .norm_squared()
                            .partial_cmp(&(b.px - kp).norm_squared())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                else {
                    continue;
                };
                let tracker_id = feature.landmark.expect("filtered to Some above");
                let Some(l) = local.get(tracker_id) else {
                    continue;
                };
                // The tracker works in the camera-anchored world frame; the map
                // stores world positions, which is the same frame here because
                // `pose` is `T_world_camera` from that tracker.
                let position = l.position;
                match self.landmark_ids.get(&tracker_id).copied() {
                    Some(existing) => {
                        if let Some(lm) = db.landmark_mut(existing) {
                            lm.position = position;
                        }
                        landmarks[i] = Some(existing);
                        landmark_ids_seen.push(existing);
                    }
                    None => {
                        let new_id = db.insert_landmark(wslam_map::Landmark {
                            id: wslam_map::LandmarkId::UNSET,
                            position,
                            descriptor: descriptors[i],
                            observations: Vec::new(),
                        });
                        self.landmark_ids.insert(tracker_id, new_id);
                        landmarks[i] = Some(new_id);
                        landmark_ids_seen.push(new_id);
                    }
                }
            }
        }

        let id = db.insert_keyframe(Keyframe {
            // UNSET is the sentinel that asks the map to assign a fresh id.
            // Passing KeyframeId(0) here is an explicit request for id 0, which
            // silently overwrote the same keyframe every time and left L4 dead:
            // one keyframe, self-edges in the pose graph, no loop closure, no
            // relocalization. Found on the first real EuRoC run.
            id: KeyframeId::UNSET,
            timestamp: frame.timestamp,
            pose,
            keypoints,
            descriptors,
            landmarks,
            bow,
            intrinsics: self.intrinsics,
        });
        // Record the observation edges now that the keyframe has an id. This
        // is what makes a landmark's `observations` list a covisibility graph
        // rather than a set of orphans.
        for lid in &landmark_ids_seen {
            if let Some(lm) = db.landmark_mut(*lid) {
                lm.observations.push(id);
                lm.observations.sort_unstable();
                lm.observations.dedup();
            }
        }
        self.keyframe_times.push((id, frame.timestamp));

        self.pose_graph
            .add_node(id, pose, self.keyframe_times.len() == 1);
        if self.keyframe_times.len() >= 2 {
            let previous = self.keyframe_times[self.keyframe_times.len() - 2].0;
            if let Some(from) = self.pose_graph.pose(previous) {
                let measurement = from.inverse().compose(&pose);
                self.pose_graph.add_edge(
                    previous,
                    id,
                    measurement,
                    wslam_core::Mat6::identity() * 100.0,
                );
            }
        }

        self.backend_queue.push_back(BackendJob::LoopSearch(id));
        if db.keyframe_count() > self.config.map.max_keyframes {
            self.backend_queue.push_back(BackendJob::Cull);
        }
    }

    /// Query place recognition and, if geometry agrees, recover.
    fn try_relocalize(&mut self, frame: &Frame) {
        let Some(db) = self.map.as_ref() else { return };
        if db.is_empty() {
            return;
        }
        let since = *self.lost_since.get_or_insert(frame.id.0);
        if frame.id.0 - since < self.config.map.reloc_delay_frames {
            return;
        }

        let detections = fast_keypoints(&frame.image, 20, 400);
        if detections.len() < 16 {
            return;
        }
        let (keypoints, angles): (Vec<Vec2>, Vec<Scalar>) = detections.into_iter().unzip();
        let descriptors = describe(&frame.image, &keypoints, &angles);
        let bow = db.transform(&descriptors);

        // `relocalize` is the only accept path and it verifies geometry on
        // every candidate. There is deliberately no unverified variant
        // (spec.md §5).
        if let Some(verified) = self.relocalizer.relocalize(
            db,
            &bow,
            &keypoints,
            &descriptors,
            &self.intrinsics,
            0,
            &mut self.rng,
        ) {
            log::info!(
                "relocalized into keyframe {} with {} inliers",
                verified.keyframe.0,
                verified.inliers
            );
            self.tracker.relocalized_to(verified.pose, frame.timestamp);
            self.lost_since = None;
            self.relocalized_since_loss = true;
            self.events.push(SlamEvent::Relocalized {
                at: frame.timestamp,
                keyframe: verified.keyframe.0,
                inliers: verified.inliers,
            });
        }
    }

    /// Execute deferred backend work within this frame's budget.
    ///
    /// One job per frame. The budget is a *scheduling* decision, not an
    /// estimate, so it does not violate the no-wall-clock rule — but counting
    /// jobs rather than milliseconds keeps it deterministic, which replay
    /// requires (spec.md §6). `backend_budget_ms` sizes the per-job work rather
    /// than gating on measured time.
    fn run_backend(&mut self) {
        let Some(job) = self.backend_queue.pop_front() else {
            return;
        };
        let Some(db) = self.map.as_mut() else { return };

        match job {
            BackendJob::Cull => {
                let removed = db.cull(&wslam_map::CullPolicy::default());
                if removed > 0 {
                    log::debug!("culled {removed} keyframes, {} remain", db.keyframe_count());
                }
            }
            BackendJob::LoopSearch(from) => {
                let Some(kf) = db.keyframe(from) else { return };
                let (bow, keypoints, descriptors) =
                    (kf.bow.clone(), kf.keypoints.clone(), kf.descriptors.clone());
                let k = kf.intrinsics;
                let candidates =
                    self.relocalizer
                        .query(db, &bow, self.config.map.loop_exclude_recent);

                for candidate in candidates.into_iter().take(3) {
                    let verified = self.relocalizer.verify(
                        db,
                        &candidate,
                        &keypoints,
                        &descriptors,
                        &k,
                        &mut self.rng,
                    );
                    let accepted = verified.is_some();
                    self.events.push(SlamEvent::LoopClosure {
                        accepted,
                        candidate: candidate.keyframe.0,
                        score: candidate.score,
                    });
                    match verified {
                        Some(v) => {
                            if let Some(anchor) = self.pose_graph.pose(candidate.keyframe) {
                                let measurement = anchor.inverse().compose(&v.pose);
                                self.pose_graph.add_edge(
                                    candidate.keyframe,
                                    from,
                                    measurement,
                                    wslam_core::Mat6::identity() * 50.0,
                                );
                                self.backend_queue.push_back(BackendJob::Optimize);
                            }
                            break;
                        }
                        None => {
                            // Rejected candidates are retained for the debug
                            // surface. spec.md §8: drawing them is how the
                            // verification threshold gets tuned by eye rather
                            // than by guesswork.
                            self.rejected_loops.push(DebugPoseGraphEdge {
                                from: from.0,
                                to: candidate.keyframe.0,
                                is_loop: true,
                                accepted: false,
                                score: candidate.score,
                            });
                        }
                    }
                }
            }
            BackendJob::Optimize => {
                let report = self.pose_graph.optimize(&SolverConfig::default());
                log::debug!(
                    "pose graph: {} iterations, cost {:.3e} -> {:.3e}",
                    report.iterations,
                    report.initial_cost,
                    report.final_cost
                );
                let updates: Vec<(KeyframeId, Se3)> = self
                    .keyframe_times
                    .iter()
                    .filter_map(|(id, _)| self.pose_graph.pose(*id).map(|p| (*id, p)))
                    .collect();
                for (id, pose) in updates {
                    db.set_keyframe_pose(id, pose);
                }
            }
        }
    }

    pub(crate) fn config_ref(&self) -> &SlamConfig {
        &self.config
    }
    pub(crate) fn tracker_ref(&self) -> &Tracker {
        &self.tracker
    }
    pub(crate) fn map_ref(&self) -> Option<&MapDb> {
        self.map.as_ref()
    }
    pub(crate) fn graph_ref(&self) -> &PoseGraph {
        &self.pose_graph
    }
    pub(crate) fn trajectory_ref(&self) -> &[wslam_core::Vec3] {
        &self.trajectory
    }
    pub(crate) fn rejected_loops_ref(&self) -> &[DebugPoseGraphEdge] {
        &self.rejected_loops
    }
}

impl std::fmt::Debug for WebSlam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebSlam")
            .field("state", &self.state)
            .field("tier", &self.effective_tier)
            .field("scale", &self.scale.source.as_str())
            .field("frames_queued", &self.frames.len())
            .field(
                "keyframes",
                &self.map.as_ref().map_or(0, MapDb::keyframe_count),
            )
            .finish()
    }
}

/// Whether this build can supply tier 3.
///
/// spec.md §3: `ScaleSource.inertial()` *"requires tier 3; throws if L0
/// unavailable"*. A caller can check first rather than catching.
#[must_use]
pub fn tight_vi_available() -> bool {
    cfg!(feature = "tight-vi")
}

/// The scale kinds this build can actually provide.
#[must_use]
pub fn available_scale_kinds() -> Vec<ScaleKind> {
    let mut kinds = vec![
        ScaleKind::None,
        ScaleKind::Declared,
        ScaleKind::Fiducial,
        ScaleKind::Map,
    ];
    if cfg!(feature = "learned-scale") {
        kinds.push(ScaleKind::Learned);
    }
    if cfg!(feature = "tight-vi") {
        kinds.push(ScaleKind::Inertial);
    }
    kinds
}

#[cfg(test)]
mod tests {
    use super::*;
    use wslam_core::{GrayImage, Vec3};

    fn config() -> SlamConfig {
        SlamConfig::new(320, 240)
    }

    /// A textured synthetic frame that moves, so the tracker has something to
    /// find. Deterministic — no RNG, no wall clock.
    fn textured_frame(index: u64, shift: f64) -> Frame {
        let (w, h) = (320u32, 240u32);
        let mut image = GrayImage::new(w, h);
        let data = image.data_mut();
        for y in 0..h as usize {
            for x in 0..w as usize {
                let fx = x as f64 + shift;
                let fy = y as f64;
                let v = 128.0
                    + 90.0 * (fx * 0.21).sin() * (fy * 0.17).cos()
                    + 40.0 * (fx * 0.07 + fy * 0.05).sin();
                data[y * w as usize + x] = v.clamp(0.0, 255.0) as u8;
            }
        }
        Frame::new(
            wslam_core::FrameId(index),
            Timestamp::from_seconds(index as f64 / 30.0),
            image,
        )
    }

    #[test]
    fn a_session_starts_up_to_scale_and_says_so() {
        // spec.md §1: "The library never silently guesses scale."
        let slam = WebSlam::new(config()).unwrap();
        assert_eq!(slam.scale().source, ScaleKind::None);
        assert!(slam.scale().variance.is_infinite());
        assert_eq!(slam.state(), TrackingState::Initializing);
        assert!(slam.current_pose().is_none());
    }

    #[test]
    fn rejects_an_impossible_image_size() {
        let bad = SlamConfig::new(0, 240);
        assert!(matches!(WebSlam::new(bad), Err(Error::Config(_))));
    }

    #[cfg(not(feature = "tight-vi"))]
    #[test]
    fn tier_three_is_refused_when_l0_is_not_compiled_in() {
        // spec.md §3: throws if L0 unavailable, rather than silently degrading.
        let cfg = SlamConfig {
            tier: SensorTier::TightVisualInertial,
            ..config()
        };
        match WebSlam::new(cfg) {
            Err(Error::SensorTier { required, reason }) => {
                assert_eq!(required, 3);
                assert!(reason.contains("tight-vi"), "{reason}");
            }
            other => panic!("expected a SensorTier error, got {other:?}"),
        }
        assert!(!tight_vi_available());
        assert!(!available_scale_kinds().contains(&ScaleKind::Inertial));
    }

    #[test]
    fn a_denied_motion_permission_degrades_to_tier_one_instead_of_failing() {
        // spec.md §4 lists tier 1 as the "fallback when motion permission is
        // denied", so this is a documented degradation, not an error.
        let mut slam = WebSlam::new(config()).unwrap();
        for i in 0..3 {
            slam.push_frame(textured_frame(i, i as f64));
            slam.step();
        }
        assert_eq!(slam.effective_tier(), SensorTier::VisionOnly);
    }

    #[test]
    fn motion_keeps_the_session_at_tier_two() {
        let mut slam = WebSlam::new(config()).unwrap();
        for i in 0..40 {
            slam.push_imu(ImuSample::new(
                Timestamp::from_seconds(i as f64 * 0.01),
                Vec3::new(0.001, 0.0, 0.0),
                Vec3::new(0.0, 0.0, wslam_core::imu::GRAVITY),
            ));
        }
        slam.push_frame(textured_frame(0, 0.0));
        slam.step();
        assert_eq!(slam.effective_tier(), SensorTier::VisionOrientation);
    }

    /// L1's prior must actually reach L3.
    ///
    /// This is the test that was missing when `previous_frame_time` was declared,
    /// initialised, read once and never assigned — leaving `rotation_prior()`
    /// returning `None` for an entire session. Nothing errored. L3 just lost its
    /// motion prediction and L2 never received a matched pair, because
    /// `update_intrinsics` gates on the prior being present.
    ///
    /// A layer that silently never runs is the worst kind of integration bug,
    /// so the wiring is asserted rather than assumed.
    #[test]
    fn the_orientation_prior_reaches_the_tracker() {
        let mut slam = WebSlam::new(config()).unwrap();
        // Enough motion, at a high enough rate, to initialise L1 and cover the
        // frame intervals.
        for i in 0..200 {
            slam.push_imu(ImuSample::new(
                Timestamp::from_seconds(i as f64 * 0.005),
                Vec3::new(0.02, -0.01, 0.015),
                Vec3::new(0.0, 0.0, wslam_core::imu::GRAVITY),
            ));
        }
        for i in 0..12 {
            slam.push_frame(textured_frame(i, i as f64 * 1.5));
            slam.step();
        }
        assert_eq!(slam.effective_tier(), SensorTier::VisionOrientation);
        assert!(
            slam.frames_with_attitude() >= 10,
            "L1 produced an attitude on only {} of 12 frames — the prior is not \
             wired through, which also disables L2",
            slam.frames_with_attitude()
        );
    }

    /// The prior is used only when it predicts more motion than it is wrong by.
    ///
    /// Measured on EuRoC: feeding L1's prediction during the near-stationary
    /// opening took frame loss from 19% to 77%, because the prediction's own
    /// error (0.88 deg p95, 7.1 px at the image edge) exceeded the motion it
    /// was predicting. Gating on magnitude is what makes "loose orientation"
    /// mean a prior used where it helps.
    #[test]
    fn the_prior_stands_aside_when_it_predicts_less_than_its_own_error() {
        // IMU is interleaved with frames rather than pushed up front, because
        // that is what a live session and the replay harness both do — and
        // because the attitude history is bounded, so a front-loaded burst
        // leaves the frames outside the retained window.
        fn run(rate: Vec3) -> WebSlam {
            let mut slam = WebSlam::new(config()).unwrap();
            // The accelerometer must agree with the gyro. Holding it constant
            // in the body frame while the gyro reports rotation describes a
            // device that is turning and not turning at once, and the filter
            // resolves that contradiction toward gravity — pinning the attitude
            // and making the test measure the rig rather than the gate.
            let world_down = Vec3::new(0.0, 0.0, wslam_core::imu::GRAVITY);
            for frame in 0..12u64 {
                let until = (frame + 1) as f64 / 30.0;
                let mut t = frame as f64 / 30.0;
                while t < until {
                    let truth = So3::exp(&(rate * t));
                    slam.push_imu(ImuSample::new(
                        Timestamp::from_seconds(t),
                        rate,
                        truth.inverse().act(&world_down),
                    ));
                    t += 0.005;
                }
                slam.push_frame(textured_frame(frame, frame as f64 * 1.5));
                slam.step();
            }
            slam
        }

        // Barely moving: 0.001 rad/s over 1/30 s frames is far under the gate.
        let still = run(Vec3::new(0.001, 0.0, 0.0));
        assert!(still.frames_with_attitude() >= 10, "L1 should be running");
        assert_eq!(
            still.frames_with_rotation_prior(),
            0,
            "a prior predicting less than its own error must stand aside"
        );

        // Turning hard: 2 rad/s is 0.067 rad per frame, well over the gate.
        let turning = run(Vec3::new(0.0, 2.0, 0.0));
        assert!(
            turning.frames_with_rotation_prior() >= 8,
            "under real rotation the prior must engage, got {} of 12",
            turning.frames_with_rotation_prior()
        );
    }

    /// The first frame has had no interval over which motion could arrive, so it
    /// must not trigger the tier-1 demotion warning.
    #[test]
    fn a_single_frame_does_not_prematurely_demote_to_tier_one() {
        let mut slam = WebSlam::new(config()).unwrap();
        slam.push_frame(textured_frame(0, 0.0));
        slam.step();
        assert_eq!(slam.effective_tier(), SensorTier::VisionOrientation);
    }

    #[test]
    fn state_transitions_are_reported_exactly_once() {
        let mut slam = WebSlam::new(config()).unwrap();
        for i in 0..12 {
            slam.push_frame(textured_frame(i, i as f64 * 1.5));
            slam.step();
        }
        let events = slam.take_events();
        let states: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                SlamEvent::State { from, to } => Some((*from, *to)),
                _ => None,
            })
            .collect();
        // No self-transitions: an event that reports the same state twice makes
        // a consumer's `onState` fire for nothing.
        for (from, to) in &states {
            assert_ne!(from, to);
        }
        // And taking them clears them.
        assert!(slam.take_events().is_empty());
    }

    #[test]
    fn every_emitted_pose_carries_covariance_and_provenance() {
        // spec.md §3: they travel *with* the pose, because separate queries get
        // skipped.
        let mut slam = WebSlam::new(config()).unwrap();
        let mut emitted = 0;
        for i in 0..40 {
            slam.push_frame(textured_frame(i, i as f64 * 1.2));
            if let Some(pose) = slam.step() {
                emitted += 1;
                assert_eq!(pose.covariance.nrows(), 6);
                assert!(wslam_core::covariance::is_valid_covariance(
                    &pose.covariance,
                    1e-6
                ));
                assert!(pose.scale.source == ScaleKind::None || pose.scale.variance.is_finite());
                assert!(pose.init_age_ms >= 0.0);
            }
        }
        // The synthetic scene is a moving texture, so whether the tracker
        // bootstraps is not what this test is about; if it never emitted, the
        // assertions above simply did not run and the test is vacuous.
        // Guard against that silently happening.
        assert!(
            emitted > 0 || slam.state() == TrackingState::Initializing,
            "no poses emitted and not initialising — the pipeline is stuck"
        );
    }

    #[test]
    fn saving_a_map_is_refused_when_mapping_is_disabled() {
        let cfg = SlamConfig {
            map: MapConfig {
                enabled: false,
                ..MapConfig::default()
            },
            ..config()
        };
        let slam = WebSlam::new(cfg).unwrap();
        assert!(matches!(slam.save_map(), Err(Error::Config(_))));
    }

    #[test]
    fn a_saved_map_round_trips_and_keeps_its_anchor() {
        let slam = WebSlam::new(config()).unwrap();
        let bytes = slam.save_map().expect("empty maps are still saveable");
        let (db, _) = wslam_map::deserialize_map(&bytes).expect("round trip");
        assert_eq!(db.scale_anchor().source, ScaleKind::None);
    }

    #[test]
    fn replay_is_deterministic() {
        // spec.md §6: the same binary replays a canned trajectory bit-for-bit.
        let run = || {
            let mut slam = WebSlam::new(config()).unwrap();
            for i in 0..30 {
                slam.push_frame(textured_frame(i, i as f64 * 1.3));
            }
            slam.drain()
                .into_iter()
                .map(|p| {
                    let t = p.position();
                    (t.x.to_bits(), t.y.to_bits(), t.z.to_bits())
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn the_debug_surface_is_available_from_the_first_frame() {
        let slam = WebSlam::new(config()).unwrap();
        let debug = slam.debug();
        assert!(debug.landmarks().is_empty());
        assert!(debug.trajectory().is_empty());
        assert_eq!(debug.timings().total_ms, 0.0);
        // The pose graph must always be queryable, even empty — a viewer that
        // has to special-case "before the first keyframe" is a worse viewer.
        assert!(debug.pose_graph().is_empty());
    }

    #[test]
    fn available_scale_kinds_reflects_the_build() {
        let kinds = available_scale_kinds();
        assert!(kinds.contains(&ScaleKind::None));
        assert!(kinds.contains(&ScaleKind::Declared));
        assert!(kinds.contains(&ScaleKind::Fiducial));
        assert!(kinds.contains(&ScaleKind::Map));
        assert_eq!(
            kinds.contains(&ScaleKind::Inertial),
            cfg!(feature = "tight-vi")
        );
    }

    #[test]
    fn an_explicit_scale_source_is_reported_once_acquired() {
        let declared = wslam_scale::DeclaredScale::from_distance(2.0, 1.0).unwrap();
        let mut slam = WebSlam::with_scale_source(config(), Box::new(declared)).unwrap();
        for i in 0..20 {
            slam.push_frame(textured_frame(i, i as f64 * 1.4));
            slam.step();
        }
        // Either it acquired scale and said so, or it has not yet — but it must
        // never report a metric kind without having emitted the event.
        let acquired = slam
            .take_events()
            .iter()
            .any(|e| matches!(e, SlamEvent::ScaleAcquired { .. }));
        if slam.scale().source.is_metric() {
            assert!(acquired, "metric scale without a ScaleAcquired event");
        }
    }
}
