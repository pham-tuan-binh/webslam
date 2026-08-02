//! # wslam-wasm — the JavaScript boundary
//!
//! spec.md §7: *"wasm-bindgen boundary. Deliberately thin — no logic, so the
//! core stays natively testable."*
//!
//! Read that as a review rule. This crate may pack, unpack, and convert units.
//! It may not decide anything. Every line here should be traceable to "a
//! JavaScript value had to become a Rust value, or vice versa" — if you find
//! yourself writing a threshold, a heuristic, or a state transition in this
//! file, it belongs in `wslam` or a layer.
//!
//! ## Wire format
//!
//! Poses cross as one packed `Float64Array`, [`POSE_STRIDE`] doubles per pose,
//! because a per-frame JSON round-trip at 60 Hz is a measurable fraction of the
//! frame budget. Events cross as JSON, because they are rare and their shapes
//! vary. The layout is mirrored in `packages/web-slam/src/wasm.ts` and pinned
//! by tests on both sides.
//!
//! | offset | count | meaning |
//! |--------|-------|---------|
//! | 0  | 1  | timestamp, ms |
//! | 1  | 3  | position xyz |
//! | 4  | 4  | rotation xyzw |
//! | 8  | 36 | covariance, row-major `[translation, rotation]` |
//! | 44 | 1  | scale kind tag |
//! | 45 | 1  | scale variance |
//! | 46 | 1  | tracking state tag |
//! | 47 | 1  | limited reason tag, or -1 |
//! | 48 | 1  | init age, ms |

#![deny(unsafe_code)]
#![warn(missing_docs)]

use wasm_bindgen::prelude::*;

/// A device acquired by [`acquire_gpu`], waiting for a session to claim it.
///
/// A thread-local rather than a field because the acquisition has to be a free
/// async function; wasm is single-threaded so there is no contention.
#[cfg(feature = "gpu")]
thread_local! {
    static PENDING_GPU: std::cell::RefCell<Option<wslam_track::GpuContext>> =
        const { std::cell::RefCell::new(None) };
}

/// Acquire a WebGPU device, to be claimed by [`WasmSlam::use_gpu`].
///
/// Asynchronous because that is what the platform requires: a browser hands out
/// adapters through a promise, and the blocking constructor the native harness
/// uses deliberately does not exist on wasm — awaiting it on the main thread
/// deadlocks the event loop.
///
/// Resolves `false` rather than rejecting when no adapter is available: a
/// device without WebGPU should still get a tracker, on the CPU reference.
#[cfg(feature = "gpu")]
#[wasm_bindgen]
pub async fn acquire_gpu() -> bool {
    match wslam_track::GpuContext::new().await {
        Ok(context) => {
            log::info!("WebGPU adapter: {}", context.adapter_name());
            PENDING_GPU.with(|c| *c.borrow_mut() = Some(context));
            true
        }
        Err(e) => {
            log::warn!("no WebGPU adapter, staying on the CPU reference: {e}");
            false
        }
    }
}

use wslam::{SensorTier, SlamConfig, WebSlam};
use wslam_core::{GrayImage, LimitedReason, MotionEvent, Pose, ScaleKind, TrackingState, Vec3};

/// Doubles per packed pose. Must match `POSE_STRIDE` in `wasm.ts`.
pub const POSE_STRIDE: usize = 49;

/// Tracking-state tags. Must match the `STATES` array in `wasm.ts`.
fn state_tag(state: TrackingState) -> f64 {
    match state {
        TrackingState::Initializing => 0.0,
        TrackingState::Tracking => 1.0,
        TrackingState::Limited(_) => 2.0,
        TrackingState::Relocalizing => 3.0,
        TrackingState::Lost => 4.0,
    }
}

/// Limited-reason tags, or `-1`. Must match `LIMITED_REASONS` in `wasm.ts`.
fn limited_tag(state: TrackingState) -> f64 {
    match state.limited_reason() {
        Some(LimitedReason::ExcessiveMotion) => 0.0,
        Some(LimitedReason::InsufficientFeatures) => 1.0,
        Some(LimitedReason::LowLight) => 2.0,
        None => -1.0,
    }
}

/// Feature-state tags. Must match `FEATURE_STATES` in `wasm.ts`.
fn feature_tag(state: wslam::layers::track::FeatureState) -> f32 {
    use wslam::layers::track::FeatureState as F;
    match state {
        F::New => 0.0,
        F::Tracked => 1.0,
        F::Outlier => 2.0,
        F::Lost => 3.0,
    }
}

/// Append one pose to the packed buffer.
fn pack_pose(out: &mut Vec<f64>, pose: &Pose) {
    let p = pose.position();
    let q = pose.rotation().quaternion();
    out.push(pose.timestamp.millis());
    out.extend_from_slice(&[p.x, p.y, p.z]);
    out.extend_from_slice(&[q.i, q.j, q.k, q.w]);
    for r in 0..6 {
        for c in 0..6 {
            out.push(pose.covariance[(r, c)]);
        }
    }
    out.push(pose.scale.source.tag() as f64);
    out.push(pose.scale.variance);
    out.push(state_tag(pose.state));
    out.push(limited_tag(pose.state));
    out.push(pose.init_age_ms);
}

/// Configuration as it arrives from JavaScript.
///
/// Parsed by hand rather than with serde: the shape is small and fixed, and
/// keeping serde out of the wasm binary matters for the size budget spec.md §7
/// gives as one of three reasons for choosing Rust.
struct WireConfig {
    width: u32,
    height: u32,
    tier: u8,
    map: bool,
    seed: u64,
    focal_prior: Option<f64>,
    scale_kind: String,
    scale_size_meters: Option<f64>,
    scale_distance_meters: Option<f64>,
    motion_available: bool,
}

impl WireConfig {
    /// Extremely small hand-rolled JSON reader for the flat config object.
    ///
    /// It does not implement JSON. It looks up known keys in a flat object
    /// written by our own shim, and every field falls back to a default, so a
    /// malformed value degrades rather than panicking across the FFI boundary —
    /// where a panic would abort the whole wasm instance.
    fn parse(json: &str) -> Self {
        WireConfig {
            width: field_f64(json, "width").unwrap_or(1280.0) as u32,
            height: field_f64(json, "height").unwrap_or(720.0) as u32,
            tier: field_f64(json, "tier").unwrap_or(2.0) as u8,
            map: field_bool(json, "map").unwrap_or(true),
            seed: field_f64(json, "seed").unwrap_or(24301.0) as u64,
            focal_prior: field_f64(json, "focalPrior"),
            scale_kind: field_str(json, "kind").unwrap_or_else(|| "none".into()),
            scale_size_meters: field_f64(json, "sizeMeters"),
            scale_distance_meters: field_f64(json, "distanceMeters"),
            motion_available: field_bool(json, "motionAvailable").unwrap_or(false),
        }
    }
}

fn field_slice<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let start = json.find(&needle)? + needle.len();
    let rest = json[start..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    Some(rest)
}

fn field_f64(json: &str, key: &str) -> Option<f64> {
    let rest = field_slice(json, key)?;
    let end = rest
        .find(|c: char| {
            !(c.is_ascii_digit() || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E')
        })
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn field_bool(json: &str, key: &str) -> Option<bool> {
    let rest = field_slice(json, key)?;
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn field_str(json: &str, key: &str) -> Option<String> {
    let rest = field_slice(json, key)?;
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// A session, as seen from JavaScript.
#[wasm_bindgen]
pub struct WasmSlam {
    inner: WebSlam,
    packed: Vec<f64>,
}

#[wasm_bindgen]
impl WasmSlam {
    /// Construct from the shim's JSON configuration.
    ///
    /// # Errors
    /// Returns the error message as a JS exception. The most likely one is
    /// asking for tier 3 from a build without the `tight-vi` feature, which
    /// spec.md §3 requires be a throw rather than a silent downgrade.
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<WasmSlam, JsValue> {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();

        let wire = WireConfig::parse(config_json);
        let mut config = SlamConfig::new(wire.width.max(1), wire.height.max(1));
        config.seed = wire.seed;
        config.map.enabled = wire.map;
        config.intrinsics.focal_prior_px = wire.focal_prior;
        config.tier = match wire.tier {
            1 => SensorTier::VisionOnly,
            3 => SensorTier::TightVisualInertial,
            _ => SensorTier::VisionOrientation,
        };
        // The shim already knows whether the platform granted motion access.
        // Honouring it here means a denied permission starts at tier 1 rather
        // than being discovered a frame later (spec.md §4).
        if !wire.motion_available && config.tier == SensorTier::VisionOrientation {
            config.tier = SensorTier::VisionOnly;
        }

        let source = build_scale_source(&wire).map_err(|e| JsValue::from_str(&e))?;
        let inner = WebSlam::with_scale_source(config, source)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(WasmSlam {
            inner,
            packed: Vec::new(),
        })
    }

    /// Install a device previously acquired by [`acquire_gpu`].
    ///
    /// Split from the acquisition because `wasm-bindgen` cannot generate a
    /// binding for an `async fn` taking `&mut self` — the borrow would have to
    /// live across the await. So acquisition is a free async function that
    /// parks the device, and this takes it. Returns whether one was waiting.
    ///
    /// Must be called before the first frame: the tracker builds its GPU
    /// pipeline lazily from whatever context it holds when a frame arrives.
    #[cfg(feature = "gpu")]
    pub fn use_gpu(&mut self) -> bool {
        match PENDING_GPU.with(|c| c.borrow_mut().take()) {
            Some(context) => {
                self.inner.set_gpu_context(context);
                true
            }
            None => false,
        }
    }


    /// Feed one camera frame as tightly-packed RGBA.
    ///
    /// The luma conversion happens here rather than in JavaScript because it is
    /// a per-pixel loop and wasm is much better at it — and because doing it in
    /// one place keeps native replay and browser runs bit-identical
    /// (spec.md §6 L3).
    pub fn push_frame(
        &mut self,
        _index: f64,
        media_time: f64,
        _arrival_ms: f64,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) {
        let expected = (width as usize) * (height as usize) * 4;
        if rgba.len() != expected || width == 0 || height == 0 {
            // A malformed frame is dropped rather than panicking: a panic here
            // poisons the wasm instance and kills the session.
            log::warn!(
                "dropping malformed frame: {} bytes for {width}x{height} (expected {expected})",
                rgba.len()
            );
            return;
        }
        self.inner
            .push_frame_raw(media_time, GrayImage::from_rgba(width, height, rgba));
    }

    /// Feed one motion event, in browser-native units (degrees per second).
    pub fn push_motion(
        &mut self,
        index: f64,
        arrival_ms: f64,
        gyro: &[f64],
        accel: &[f64],
        interval_s: f64,
    ) {
        if gyro.len() < 3 || accel.len() < 3 {
            return;
        }
        self.inner.push_motion(MotionEvent::from_browser(
            index as u64,
            arrival_ms,
            Vec3::new(gyro[0], gyro[1], gyro[2]),
            Vec3::new(accel[0], accel[1], accel[2]),
            None,
            interval_s,
        ));
    }

    /// Advance the pipeline and return every pose produced since the last call,
    /// packed.
    ///
    /// Returns a list rather than one pose so the push path cannot silently
    /// drop samples when the host calls at a slower rate than frames arrive
    /// (spec.md §3, pull vs push).
    pub fn step_poses(&mut self) -> Vec<f64> {
        self.packed.clear();
        while let Some(pose) = self.inner.step() {
            pack_pose(&mut self.packed, &pose);
        }
        self.packed.clone()
    }

    /// Take the events accumulated since the last call, as a JSON array.
    pub fn take_events(&mut self) -> String {
        let events = self.inner.take_events();
        if events.is_empty() {
            return "[]".to_string();
        }
        let mut out = String::from("[");
        for (i, event) in events.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&event.to_json());
        }
        out.push(']');
        out
    }

    /// Serialise the map.
    pub fn save_map(&self) -> Result<Vec<u8>, JsValue> {
        self.inner
            .save_map()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Load a previously saved map.
    pub fn load_map(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        self.inner
            .load_map(bytes)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Landmark positions, packed `xyz`.
    pub fn debug_landmarks(&self) -> Vec<f32> {
        self.inner.debug().landmarks_packed()
    }

    /// Trajectory, packed `xyz`.
    pub fn debug_trajectory(&self) -> Vec<f32> {
        self.inner.debug().trajectory_packed()
    }

    /// Features, packed `[id, x, y, stateTag, age]`.
    pub fn debug_features(&self) -> Vec<f32> {
        let features = self.inner.debug().features();
        let mut out = Vec::with_capacity(features.len() * 5);
        for f in features {
            out.push(f.id as f32);
            out.push(f.px.x as f32);
            out.push(f.px.y as f32);
            out.push(feature_tag(f.state));
            out.push(f.age as f32);
        }
        out
    }

    /// Keyframes, packed `[id, timestamp, px, py, pz, qx, qy, qz, qw]`.
    pub fn debug_keyframes(&self) -> Vec<f64> {
        let keyframes = self.inner.debug().keyframes();
        let mut out = Vec::with_capacity(keyframes.len() * 9);
        for kf in keyframes {
            let p = kf.pose.translation();
            let q = kf.pose.rotation().quaternion();
            out.extend_from_slice(&[
                kf.id as f64,
                kf.timestamp_ms,
                p.x,
                p.y,
                p.z,
                q.i,
                q.j,
                q.k,
                q.w,
            ]);
        }
        out
    }

    /// Pose-graph edges as JSON, including rejected loop candidates.
    pub fn debug_pose_graph(&self) -> String {
        let edges = self.inner.debug().pose_graph();
        let mut out = String::from("[");
        for (i, e) in edges.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                r#"{{"from":{},"to":{},"kind":"{}","accepted":{},"score":{:.6}}}"#,
                e.from,
                e.to,
                if e.is_loop { "loop" } else { "odometry" },
                e.accepted,
                e.score
            ));
        }
        out.push(']');
        out
    }

    /// Per-stage timings: `[upload, pyramid, corners, flow, pnp, total]`, ms.
    pub fn debug_timings(&self) -> Vec<f64> {
        let t = self.inner.debug().timings();
        vec![
            t.upload_ms,
            t.pyramid_ms,
            t.corners_ms,
            t.flow_ms,
            t.pnp_ms,
            t.total_ms,
        ]
    }

    /// Current estimated focal length in pixels, for the renderer's projection.
    pub fn focal_px(&self) -> f64 {
        self.inner.intrinsics().fx
    }

    /// The sensor tier actually in force, which may be below the configured one.
    pub fn effective_tier(&self) -> u8 {
        self.inner.effective_tier().number()
    }

    /// Version of the stable public API.
    pub fn version(&self) -> String {
        wslam_core::PUBLIC_API_VERSION.to_string()
    }
}

/// Build the declared scale source.
///
/// Unknown kinds are an error rather than a fallback to `none`: silently
/// downgrading a caller who asked for `fiducial` would be exactly the silent
/// guess spec.md §1 forbids.
fn build_scale_source(wire: &WireConfig) -> Result<Box<dyn wslam::scale::ScaleSource>, String> {
    use wslam::scale::{DeclaredScale, FiducialScale, NoneScale, TagFamily};
    match wire.scale_kind.as_str() {
        "none" => Ok(Box::new(NoneScale::new())),
        "declared" => {
            let metres = wire.scale_distance_meters.ok_or_else(|| {
                "ScaleSource.declared() needs distanceMeters before it can anchor".to_string()
            })?;
            // The observed up-to-scale distance is supplied later by the UI; a
            // unit baseline means "not yet calibrated", and the source reports
            // no estimate until it is.
            DeclaredScale::from_distance(metres, 1.0)
                .map(|s| Box::new(s) as Box<dyn wslam::scale::ScaleSource>)
                .map_err(|e| e.to_string())
        }
        "fiducial" => {
            let size = wire
                .scale_size_meters
                .ok_or_else(|| "ScaleSource.fiducial() needs sizeMeters".to_string())?;
            FiducialScale::new(TagFamily::AprilTag36h11, size)
                .map(|s| Box::new(s) as Box<dyn wslam::scale::ScaleSource>)
                .map_err(|e| e.to_string())
        }
        // `map` is configured by `load_map`, which reads the anchor out of the
        // bytes; there is nothing to build from the config object.
        "map" => Ok(Box::new(NoneScale::new())),
        "inertial" => Err(
            "ScaleSource.inertial() requires sensor tier 3 and a build with the \
             `tight-vi` feature. See spec.md §2 for the alternatives."
                .to_string(),
        ),
        "learned" => Err(
            "ScaleSource.learned() requires a build with the `learned-scale` feature.".to_string(),
        ),
        other => Err(format!("unknown ScaleSource kind {other:?}")),
    }
}

/// Install a `console.log`-backed logger. Call once, from JS, before `new`.
#[wasm_bindgen]
pub fn init_logging(level: &str) {
    let filter = match level {
        "trace" => log::LevelFilter::Trace,
        "debug" => log::LevelFilter::Debug,
        "warn" => log::LevelFilter::Warn,
        "error" => log::LevelFilter::Error,
        _ => log::LevelFilter::Info,
    };
    log::set_max_level(filter);
}

/// The packed pose stride, exported so the JS side can assert agreement rather
/// than hard-code a number in two places.
#[wasm_bindgen]
pub fn pose_stride() -> usize {
    POSE_STRIDE
}

/// Scale kinds this build can provide, comma-separated.
#[wasm_bindgen]
pub fn available_scale_kinds() -> String {
    wslam::available_scale_kinds()
        .into_iter()
        .map(ScaleKind::as_str)
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;
    use wslam_core::{Mat6, ScaleEstimate, Se3, So3, Timestamp};

    #[test]
    fn the_packed_layout_matches_the_documented_offsets() {
        let mut pose = Pose::identity_at(Timestamp::from_millis_f64(1234.5));
        pose.transform = Se3::new(
            So3::exp(&Vec3::new(0.0, 0.0, 0.0)),
            Vec3::new(1.0, 2.0, 3.0),
        );
        pose.covariance = Mat6::identity() * 0.25;
        pose.scale = ScaleEstimate::metric(ScaleKind::Fiducial, 1.0, 1e-5);
        pose.state = TrackingState::Tracking;
        pose.init_age_ms = 900.0;

        let mut packed = Vec::new();
        pack_pose(&mut packed, &pose);

        assert_eq!(packed.len(), POSE_STRIDE);
        assert_eq!(packed[0], 1234.5);
        assert_eq!(&packed[1..4], &[1.0, 2.0, 3.0]);
        assert_eq!(packed[7], 1.0); // identity quaternion w
        assert_eq!(packed[8], 0.25); // covariance[0][0]
        assert_eq!(packed[44], ScaleKind::Fiducial.tag() as f64);
        assert_eq!(packed[46], 1.0); // tracking
        assert_eq!(packed[47], -1.0); // not limited
        assert_eq!(packed[48], 900.0);
    }

    #[test]
    fn limited_state_packs_its_reason() {
        let mut pose = Pose::identity_at(Timestamp::ZERO);
        pose.state = TrackingState::Limited(LimitedReason::LowLight);
        let mut packed = Vec::new();
        pack_pose(&mut packed, &pose);
        assert_eq!(packed[46], 2.0);
        assert_eq!(packed[47], 2.0);
    }

    #[test]
    fn state_tags_are_distinct_and_stable() {
        // These numbers are a wire contract with `wasm.ts`. Reordering them
        // silently mislabels every state in the browser.
        assert_eq!(state_tag(TrackingState::Initializing), 0.0);
        assert_eq!(state_tag(TrackingState::Tracking), 1.0);
        assert_eq!(
            state_tag(TrackingState::Limited(LimitedReason::LowLight)),
            2.0
        );
        assert_eq!(state_tag(TrackingState::Relocalizing), 3.0);
        assert_eq!(state_tag(TrackingState::Lost), 4.0);
    }

    #[test]
    fn config_parsing_reads_the_shim_shape() {
        let json = r#"{"scale":{"kind":"fiducial","config":{"family":"apriltag36h11","sizeMeters":0.1}},"tier":2,"map":true,"seed":1234,"focalPrior":null,"width":1280,"height":720,"motionAvailable":true,"mapBytesLength":0}"#;
        let wire = WireConfig::parse(json);
        assert_eq!(wire.width, 1280);
        assert_eq!(wire.height, 720);
        assert_eq!(wire.tier, 2);
        assert_eq!(wire.seed, 1234);
        assert!(wire.map);
        assert!(wire.motion_available);
        assert_eq!(wire.scale_kind, "fiducial");
        assert_eq!(wire.scale_size_meters, Some(0.1));
        // `null` must not parse as a number.
        assert_eq!(wire.focal_prior, None);
    }

    #[test]
    fn malformed_config_falls_back_instead_of_panicking() {
        // A panic across the FFI boundary poisons the wasm instance.
        let wire = WireConfig::parse("not json at all");
        assert_eq!(wire.width, 1280);
        assert_eq!(wire.scale_kind, "none");
        assert!(WireConfig::parse("").focal_prior.is_none());
    }

    #[test]
    fn an_unknown_scale_kind_is_an_error_not_a_silent_downgrade() {
        // spec.md §1: the library never silently guesses scale. Quietly
        // substituting `none` for a misspelled `fiducial` would do exactly that.
        let wire = WireConfig {
            scale_kind: "fiducal".into(),
            ..WireConfig::parse("{}")
        };
        assert!(build_scale_source(&wire).is_err());
    }

    #[test]
    fn fiducial_without_a_size_is_refused() {
        let wire = WireConfig {
            scale_kind: "fiducial".into(),
            scale_size_meters: None,
            ..WireConfig::parse("{}")
        };
        let err = match build_scale_source(&wire) {
            Err(e) => e,
            Ok(_) => panic!("fiducial without a size must be refused"),
        };
        assert!(err.contains("sizeMeters"), "{err}");
    }

    #[test]
    fn inertial_is_refused_with_an_actionable_message() {
        let wire = WireConfig {
            scale_kind: "inertial".into(),
            ..WireConfig::parse("{}")
        };
        let err = match build_scale_source(&wire) {
            Err(e) => e,
            Ok(_) => panic!("inertial must be refused without tier 3"),
        };
        assert!(err.contains("tier 3"), "{err}");
        assert!(err.contains("tight-vi"), "{err}");
    }

    #[test]
    fn available_kinds_are_reported_as_the_shim_expects() {
        let kinds = available_scale_kinds();
        assert!(kinds.contains("none"));
        assert!(kinds.contains("fiducial"));
        assert!(!kinds.ends_with(','));
    }

    #[test]
    fn pose_stride_is_exported_so_both_sides_can_agree() {
        assert_eq!(pose_stride(), 49);
    }
}
