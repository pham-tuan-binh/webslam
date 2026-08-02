# Crate contracts

Frozen interfaces between the layer crates. `wslam` (orchestration) is written
against exactly these signatures, so a crate that deviates breaks the build.

Read `crates/wslam-core/src/` first — it is the shared vocabulary and it is
complete. Do not modify it.

## Conventions that apply everywhere

- Poses are `T_world_camera` (`Se3`). `translation()` is the camera centre in
  world coordinates.
- Twists and 6x6 covariances are `[translation; rotation]`, right-perturbation
  (`T ⊞ δ = T · exp(δ)`).
- Geometry is `f64`, image data is `u8`/`f32`.
- **No wall clock.** Time enters via `wslam_core::TimeBase`. `HostClock` is for
  profiling only and may not appear on an estimation path.
- **No unseeded RNG.** `wslam_core::DeterministicRng` only. RANSAC included.
- Every public item carries a doc comment (`#![warn(missing_docs)]`).
- Errors are `wslam_core::Error` / `Result`.

---

## `wslam-gpu` — GPU compute

```rust
pub struct GpuContext;
impl GpuContext {
    pub async fn new() -> Result<Self>;
    pub fn limits(&self) -> GpuLimits;
}
pub struct GpuLimits { pub max_workgroup_size: u32, pub max_buffer_bytes: u64 }

/// Full L3 image front-end on the GPU: upload -> pyramid -> corners -> flow.
pub struct ImagePipeline;
impl ImagePipeline {
    pub fn new(ctx: &GpuContext, width: u32, height: u32, levels: u32) -> Result<Self>;
    pub fn upload(&mut self, image: &GrayImage) -> Result<()>;
    pub fn build_pyramid(&mut self) -> Result<()>;
    /// Shi-Tomasi response + non-max suppression. Reads back a few hundred
    /// points, never a full image (spec.md §4 L3).
    pub fn detect_corners(&mut self, max_corners: usize, quality: f32, min_distance: f32)
        -> Result<Vec<(f32, f32, f32)>>;   // (x, y, response)
    /// Pyramidal Lucas-Kanade. `prev` must have been uploaded to the other
    /// buffer set via `swap()`.
    pub fn track_flow(&mut self, points: &[(f32, f32)], config: &FlowConfig)
        -> Result<Vec<FlowResult>>;
    pub fn swap(&mut self);
}
pub struct FlowConfig { pub window: u32, pub iterations: u32, pub epsilon: f32, pub max_error: f32 }
pub struct FlowResult { pub x: f32, pub y: f32, pub ok: bool, pub error: f32 }
```

WGSL kernels live in `src/shaders/*.wgsl` and are `include_str!`-ed. Every
kernel must have a **CPU reference** in `src/reference.rs` and an equivalence
test — that is what makes "any divergence is a port bug" (spec.md §6 L3)
checkable. If no adapter is available the tests must `skip`, not fail.

## `wslam-clock` — L0

```rust
/// Linear model t = slope * index + intercept, fit robustly over event INDEX,
/// not over the jittery delivery stamp (spec.md §4 L0).
pub struct CadenceModel;
impl CadenceModel {
    pub fn new(config: CadenceConfig) -> Self;
    pub fn push(&mut self, index: u64, observed_seconds: f64);
    pub fn predict(&self, index: u64) -> Option<f64>;
    pub fn slope(&self) -> Option<f64>;          // seconds per event
    pub fn residual_variance(&self) -> f64;      // s^2 — the reported number
    pub fn sample_count(&self) -> usize;
    pub fn is_converged(&self) -> bool;
}
pub struct CadenceConfig { pub min_samples: usize, pub window: usize, pub huber_k: f64 }

/// Camera-IMU offset td as an online filter state (Li & Mourikis 2014).
pub struct OffsetFilter;
impl OffsetFilter {
    pub fn new(initial_variance: f64, process_noise: f64) -> Self;
    pub fn update(&mut self, measured_offset: f64, measurement_variance: f64);
    pub fn offset(&self) -> f64;
    pub fn variance(&self) -> f64;
    /// Suspend estimation under degenerate motion, per Li & Mourikis §V.
    pub fn set_degenerate(&mut self, degenerate: bool);
}

pub struct FittedTimeBase;   // impl wslam_core::TimeBase
impl FittedTimeBase { pub fn new(config: ClockConfig) -> Self; }

/// Offline: cross-correlate two rate signals, peak lag IS the offset
/// (spec.md §6 L0). Used by the turntable + strobe rig.
pub fn cross_correlate_lag(a: &[(f64, f64)], b: &[(f64, f64)], max_lag: f64, step: f64)
    -> Option<LagEstimate>;
pub struct LagEstimate { pub lag_seconds: f64, pub correlation: f64, pub variance: f64 }
```

## `wslam-orientation` — L1

Error-state Kalman filter on SO(3) with gyro bias. Drift-free in roll/pitch,
yaw drifts and is arrested by L3.

```rust
pub struct OrientationFilter;
impl OrientationFilter {
    pub fn new(config: OrientationConfig) -> Self;
    pub fn integrate(&mut self, sample: &ImuSample);
    pub fn attitude(&self) -> So3;                // R_world_body
    pub fn gravity_body(&self) -> Vec3;           // gravity direction in body frame
    pub fn gyro_bias(&self) -> Vec3;
    pub fn covariance(&self) -> Mat3;             // attitude error covariance
    pub fn is_initialized(&self) -> bool;
    /// L3 hands back an absolute yaw observation to arrest yaw drift.
    pub fn correct_yaw(&mut self, yaw_world: f64, variance: f64);
    /// Predicted relative rotation between two times, for tracking prediction.
    pub fn delta_rotation(&self, from: Timestamp, to: Timestamp) -> Option<So3>;
}
pub struct OrientationConfig {
    pub gyro_noise: f64, pub gyro_bias_walk: f64, pub accel_noise: f64,
    pub gravity_gate: f64,   // reject accel updates when |a| deviates from g
    pub static_threshold: f64,
}
```

## `wslam-calib` — L2

```rust
pub struct FocalEstimator;
impl FocalEstimator {
    pub fn new(config: CalibConfig, width: u32, height: u32) -> Self;
    /// Feed a matched pair with the gyro-known relative rotation.
    pub fn push_pair(&mut self, matches: &[(Vec2, Vec2)], relative_rotation: &So3) -> Result<()>;
    pub fn estimate(&self) -> Option<FocalEstimate>;
    pub fn pair_count(&self) -> usize;
}
pub struct FocalEstimate {
    pub focal_px: f64, pub variance: f64,
    pub distortion: RadialTangential, pub pairs_used: usize,
}
/// The two ablations spec.md §6 L2 makes a gate, not a nice-to-have.
pub struct CalibConfig {
    pub model_distortion: bool,
    pub model_lever_arm: bool,
    /// Camera-IMU translation. Handheld rotation is about the wrist, ~20 cm
    /// from the optical centre, which injects translation (Ji et al.).
    pub lever_arm_m: Vec3,
    pub prior_hfov_degrees: f64,
    pub min_pairs: usize,
    pub ransac_iterations: usize,
    pub seed: u64,
}
/// Infinite-homography constraint (de Agapito et al. 2001).
pub fn focal_from_rotation_homography(h: &Mat3, r: &So3) -> Option<f64>;
pub fn estimate_homography(matches: &[(Vec2, Vec2)]) -> Option<Mat3>;
pub fn estimate_homography_ransac(matches: &[(Vec2, Vec2)], threshold: f64,
                                  iterations: usize, rng: &mut DeterministicRng)
    -> Option<(Mat3, Vec<bool>)>;
```

## `wslam-track` — L3

```rust
pub struct Tracker;
impl Tracker {
    pub fn new(config: TrackConfig, intrinsics: CameraIntrinsics) -> Self;
    /// The per-frame entry point. `rotation_prior` is L1's prediction (tier 2+).
    pub fn process(&mut self, frame: &Frame, rotation_prior: Option<So3>) -> TrackOutcome;
    pub fn set_intrinsics(&mut self, k: CameraIntrinsics);
    pub fn reset(&mut self);
    /// Force the pose after relocalization; the local map is rebuilt around it.
    pub fn relocalized_to(&mut self, pose: Se3, timestamp: Timestamp);
    pub fn local_map(&self) -> &LocalMap;
    pub fn features(&self) -> &[Feature];
    pub fn timings(&self) -> &StageTimings;
}
pub struct TrackOutcome {
    pub state: TrackingState,
    pub pose: Option<Se3>,          // up to scale
    pub covariance: Mat6,
    pub inlier_count: usize,
    pub tracked_count: usize,
    pub is_keyframe: bool,
}
pub struct TrackConfig {
    pub max_features: usize, pub min_features: usize, pub pyramid_levels: u32,
    pub klt_window: u32, pub klt_iterations: u32,
    pub ransac_threshold_px: f64, pub ransac_iterations: usize,
    pub keyframe_translation: f64, pub keyframe_rotation_rad: f64,
    pub keyframe_min_frames: u64, pub low_light_threshold: f64, pub seed: u64,
    pub use_gpu: bool,
}
pub struct Feature { pub id: u64, pub px: Vec2, pub state: FeatureState, pub landmark: Option<u64>, pub age: u32 }
pub enum FeatureState { New, Tracked, Outlier, Lost }
pub struct LocalMap { /* landmarks with positions + observation counts */ }
impl LocalMap {
    pub fn landmarks(&self) -> &[LocalLandmark];
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}
pub struct LocalLandmark { pub id: u64, pub position: Vec3, pub observations: u32 }
pub struct StageTimings { pub upload_ms: f64, pub pyramid_ms: f64, pub corners_ms: f64,
                          pub flow_ms: f64, pub pnp_ms: f64, pub total_ms: f64 }
```

Submodules to build: `pyramid`, `corners` (Shi-Tomasi + grid-bucketed NMS),
`klt` (pyramidal Lucas-Kanade, inverse compositional), `pnp` (P3P/EPnP +
RANSAC over `DeterministicRng`), `triangulate` (DLT + cheirality),
`motion_ba` (motion-only bundle adjustment, Huber), `init` (two-view
homography/essential bootstrap).

## `wslam-map` — L4

```rust
pub struct BinaryDescriptor(pub [u8; 32]);          // 256-bit, ORB-style
impl BinaryDescriptor { pub fn hamming(&self, other: &Self) -> u32; }
pub fn describe(image: &GrayImage, keypoints: &[Vec2], orientations: &[f64]) -> Vec<BinaryDescriptor>;
pub fn fast_keypoints(image: &GrayImage, threshold: u8, max: usize) -> Vec<(Vec2, f64)>;

/// DBoW2-style vocabulary tree (Gálvez-López & Tardós 2012), reimplemented —
/// the vocabulary file is reusable data, the code is a tree search.
pub struct Vocabulary;
impl Vocabulary {
    pub fn train(descriptors: &[BinaryDescriptor], branching: usize, depth: usize,
                 rng: &mut DeterministicRng) -> Self;
    pub fn transform(&self, descriptors: &[BinaryDescriptor]) -> BowVector;
    pub fn serialize(&self) -> Vec<u8>;
    pub fn deserialize(bytes: &[u8]) -> Result<Self>;
    pub fn word_count(&self) -> usize;
}
pub struct BowVector;      // sparse word_id -> tf-idf weight
impl BowVector { pub fn score(&self, other: &Self) -> f64; }   // L1 score, in [0,1]

pub struct KeyframeId(pub u64);
pub struct LandmarkId(pub u64);
pub struct Keyframe { pub id: KeyframeId, pub timestamp: Timestamp, pub pose: Se3,
                      pub keypoints: Vec<Vec2>, pub descriptors: Vec<BinaryDescriptor>,
                      pub landmarks: Vec<Option<LandmarkId>>, pub bow: BowVector,
                      pub intrinsics: CameraIntrinsics }
pub struct Landmark { pub id: LandmarkId, pub position: Vec3,
                      pub descriptor: BinaryDescriptor, pub observations: Vec<KeyframeId> }

pub struct MapDb;
impl MapDb {
    pub fn new(vocabulary: Arc<Vocabulary>) -> Self;
    pub fn insert_keyframe(&mut self, kf: Keyframe) -> KeyframeId;
    pub fn insert_landmark(&mut self, lm: Landmark) -> LandmarkId;
    pub fn keyframe(&self, id: KeyframeId) -> Option<&Keyframe>;
    pub fn keyframes(&self) -> impl Iterator<Item = &Keyframe>;
    pub fn landmarks(&self) -> impl Iterator<Item = &Landmark>;
    /// Bounded memory: cull redundant keyframes (spec.md §9, "tab killed").
    pub fn cull(&mut self, policy: &CullPolicy) -> usize;
    pub fn memory_bytes(&self) -> usize;
    pub fn scale_anchor(&self) -> ScaleEstimate;
    pub fn set_scale_anchor(&mut self, s: ScaleEstimate);
}

/// Place recognition + MANDATORY geometric verification. A false positive
/// corrupts the map irrecoverably and is worse than no loop closure at all
/// (spec.md §5) — `verify` is not optional and its threshold is a release gate.
pub struct Relocalizer;
impl Relocalizer {
    pub fn new(config: RelocConfig) -> Self;
    pub fn query(&self, db: &MapDb, bow: &BowVector, exclude_recent: usize) -> Vec<Candidate>;
    pub fn verify(&self, db: &MapDb, candidate: &Candidate,
                  keypoints: &[Vec2], descriptors: &[BinaryDescriptor],
                  k: &CameraIntrinsics, rng: &mut DeterministicRng) -> Option<Verified>;
}
pub struct Candidate { pub keyframe: KeyframeId, pub score: f64 }
pub struct Verified { pub keyframe: KeyframeId, pub pose: Se3, pub inliers: usize, pub covariance: Mat6 }
pub struct RelocConfig { pub min_bow_score: f64, pub max_candidates: usize,
                         pub min_inliers: usize, pub ransac_threshold_px: f64,
                         pub ransac_iterations: usize }

/// Pose graph, Gauss-Newton over SE(3). We need an optimizer, not a solver
/// framework (spec.md §7).
pub struct PoseGraph;
impl PoseGraph {
    pub fn new() -> Self;
    pub fn add_node(&mut self, id: KeyframeId, pose: Se3, fixed: bool);
    pub fn add_edge(&mut self, from: KeyframeId, to: KeyframeId, measurement: Se3, information: Mat6);
    pub fn optimize(&mut self, config: &SolverConfig) -> SolverReport;
    pub fn pose(&self, id: KeyframeId) -> Option<Se3>;
    pub fn edges(&self) -> &[Edge];
}
pub struct SolverConfig { pub max_iterations: usize, pub tolerance: f64,
                          pub lambda: f64, pub huber_delta: f64 }
pub struct SolverReport { pub iterations: usize, pub initial_cost: f64,
                          pub final_cost: f64, pub converged: bool }

/// Versioned binary map format. Round-trip is a Tier-1 property test.
pub fn serialize_map(db: &MapDb) -> Vec<u8>;
pub fn deserialize_map(bytes: &[u8]) -> Result<(MapDb, Arc<Vocabulary>)>;
```

## `wslam-scale` — L5

```rust
pub trait ScaleSource: Send {
    fn kind(&self) -> ScaleKind;
    fn estimate(&mut self, window: &StateWindow) -> Option<ScaleEstimate>;
    fn reset(&mut self) {}
}
pub struct NoneScale;                                     // honest default
pub struct DeclaredScale;  // ::new(metres_between: (Vec3, Vec3), observed_units: f64)
pub struct FiducialScale;  // ::new(family, size_meters); consumes detections
pub struct MapScale;       // ::new(anchor: ScaleEstimate, reloc_variance: f64)
pub struct LearnedScale;   // ::new(model) — opt-in, feature `learned-scale`
pub struct InertialScale;  // ::new(config) — tier 3; Err if L0 absent
pub mod fiducial { /* AprilTag 36h11 quad detection + decode + IPPE pose */ }
```

`MapScale` **must** inflate: `anchor.inflated_by(reloc_variance)`.
`InertialScale` **must** return `None` when `window.mean_excitation()` is below
the observability threshold rather than a confident wrong answer.

## `wslam` — orchestration

The only crate aware of every layer. Owns the state machine, the
frontend/backend split, and the debug snapshot.

```rust
pub struct WebSlam;
impl WebSlam {
    pub fn new(config: SlamConfig) -> Result<Self>;
    pub fn push_frame(&mut self, frame: Frame);
    pub fn push_motion(&mut self, event: MotionEvent);
    /// Advance one step. Returns the pose if one was produced.
    pub fn step(&mut self) -> Option<Pose>;
    pub fn current_pose(&self) -> Option<Pose>;
    pub fn state(&self) -> TrackingState;
    pub fn take_events(&mut self) -> Vec<SlamEvent>;
    pub fn save_map(&self) -> Result<Vec<u8>>;
    pub fn debug(&self) -> DebugSnapshot<'_>;
}
pub enum SlamEvent {
    State { from: TrackingState, to: TrackingState },
    Relocalized { at: Timestamp, keyframe: u64 },
    LoopClosure { accepted: bool, candidate: u64, score: f64 },
    ScaleAcquired { estimate: ScaleEstimate },
}
pub struct SlamConfig { pub tier: SensorTier, pub intrinsics: Option<CameraIntrinsics>,
                        pub scale: ScaleConfig, pub map: MapConfig,
                        pub track: TrackConfig, pub seed: u64 }
pub enum SensorTier { VisionOnly, VisionOrientation, TightVisualInertial }
pub struct DebugSnapshot<'a> { /* landmarks, keyframes, trajectory, features,
                                  pose_graph (incl. REJECTED loop candidates),
                                  timings */ }
```

---

## Rules for implementers

1. `cargo test -p <your-crate>` must pass, and `cargo clippy -p <your-crate>
   -- -D warnings` must be clean.
2. Do **not** edit `crates/wslam-core/`, the root `Cargo.toml`, or another
   crate's files. Add dependencies to your own `Cargo.toml` using
   `foo.workspace = true` where the workspace already declares `foo`.
3. Tier-1 tests are the deliverable, not an afterthought: closed-form answers on
   synthetic input, property tests for round-trips, and a named test for every
   degenerate case the spec calls out.
4. Set `CARGO_TARGET_DIR` to your own scratch path so parallel builds do not
   serialise on the workspace target lock.
5. Comment *why*, not *what*, and cite the spec section when a choice is
   traceable to one.
