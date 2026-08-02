//! Session configuration.
//!
//! Everything the orchestrator needs, declared up front. spec.md §4 applies the
//! same discipline to sensors that §1 applies to scale: *"Sensor use is declared
//! configuration, not an assumption."*

use wslam_core::{CameraIntrinsics, Scalar, So3};
use wslam_track::TrackConfig;

/// Which sensors the session is allowed to use.
///
/// From the table in spec.md §4. The tier is a *ceiling*: a session configured
/// for tier 2 drops to tier 1 automatically when motion permission is denied,
/// because §4 lists tier 1 as the "fallback when motion permission is denied"
/// rather than an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum SensorTier {
    /// Vision only. No gravity, no orientation prior.
    VisionOnly = 1,
    /// Vision + loose orientation. **The baseline, and what we ship.**
    #[default]
    VisionOrientation = 2,
    /// Tight visual-inertial. Requires L0; adds inertial metric scale and
    /// nothing else.
    TightVisualInertial = 3,
}

impl SensorTier {
    /// Numeric tier, matching the spec's table.
    #[must_use]
    pub fn number(self) -> u8 {
        self as u8
    }

    /// Whether this tier consumes inertial data at all.
    #[must_use]
    pub fn uses_motion(self) -> bool {
        self >= SensorTier::VisionOrientation
    }
}

/// L4 configuration.
#[derive(Debug, Clone)]
pub struct MapConfig {
    /// Build keyframes and relocalize at all.
    pub enabled: bool,
    /// Keyframes above which culling runs.
    pub max_keyframes: usize,
    /// Frames to wait after loss before querying place recognition.
    ///
    /// Non-zero because the frames immediately after loss are usually the
    /// motion-blurred ones that caused it, and a query against blur is a query
    /// that will fail.
    pub reloc_delay_frames: u64,
    /// Recent keyframes excluded from loop-closure queries. Without this every
    /// frame "recognises" its immediate predecessor.
    pub loop_exclude_recent: usize,
    /// Per-frame budget for backend work, in milliseconds.
    ///
    /// The default build is single-threaded and embeddable anywhere
    /// (docs/DECISIONS.md D2), so the frontend/backend split is enforced by
    /// this budget rather than by a thread boundary. spec.md §4 is
    /// unambiguous: *"The frontend must never block on the backend."*
    pub backend_budget_ms: Scalar,
    /// Extra scale variance attributed to relocalizing into a stored map.
    ///
    /// spec.md §4 L5: a map-derived scale *"must not report itself as more
    /// certain than its origin"*, so this is added to the anchor's own
    /// variance, never substituted for it.
    pub reloc_scale_variance: Scalar,
}

impl Default for MapConfig {
    fn default() -> Self {
        MapConfig {
            enabled: true,
            max_keyframes: 300,
            reloc_delay_frames: 3,
            loop_exclude_recent: 12,
            backend_budget_ms: 4.0,
            reloc_scale_variance: 4e-4,
        }
    }
}

/// L2 behaviour for this session.
#[derive(Debug, Clone)]
pub struct IntrinsicsConfig {
    /// Known intrinsics. When supplied, L2 does not run at all.
    ///
    /// A caller who has calibrated the device before — or read a stored
    /// estimate from a previous session — should pass it. The init pan is a
    /// user-visible cost.
    pub known: Option<CameraIntrinsics>,
    /// Focal-length prior in pixels, if the caller has a guess but not a
    /// measurement.
    pub focal_prior_px: Option<Scalar>,
    /// Run L2's estimator during initialisation.
    pub estimate_online: bool,
    /// Underlying L2 configuration, including the two required ablation knobs.
    pub calib: wslam_calib::CalibConfig,
}

impl Default for IntrinsicsConfig {
    fn default() -> Self {
        IntrinsicsConfig {
            known: None,
            focal_prior_px: None,
            estimate_online: true,
            calib: wslam_calib::CalibConfig::default(),
        }
    }
}

/// Everything a session needs.
#[derive(Debug, Clone)]
pub struct SlamConfig {
    /// Sensor ceiling.
    pub tier: SensorTier,
    /// Image size in pixels. Required, because L2's prior depends on it.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Intrinsics handling.
    pub intrinsics: IntrinsicsConfig,
    /// L3 configuration.
    pub track: TrackConfig,
    /// L4 configuration.
    pub map: MapConfig,
    /// L1 configuration.
    pub orientation: wslam_orientation::OrientationConfig,
    /// Smallest inter-frame rotation, in radians, for which L1's prediction is
    /// handed to L3.
    ///
    /// A prior is only worth using when the motion it predicts is larger than
    /// its own error. Below that the prediction displaces a feature further
    /// from truth than leaving it where it was, and KLT does better with no
    /// guess at all.
    ///
    /// Measured on EuRoC MH_01: L1's inter-frame rotation error is 0.88 deg at
    /// p95, which is 7.1 px at the image edge for that camera. During the
    /// sequence's hovering opening the true inter-frame motion is smaller than
    /// that, and feeding the prediction anyway took frame loss from 19% to 77%.
    /// The default is set just above the measured p95 so the prior engages for
    /// real motion and stands aside for noise.
    ///
    /// Set to zero to always use the prior, or to infinity to never use it —
    /// both are useful for ablation and neither is a good default.
    pub min_prior_rotation_rad: Scalar,
    /// Rotation taking **camera** axes into **IMU body** axes, `R_body_camera`.
    ///
    /// L1 estimates attitude in the body frame the inertial sensor reports in;
    /// L2 and L3 need it in the camera frame. Those differ by a real rotation —
    /// EuRoC's `T_BS` for cam0 is about 90 degrees, and a phone's camera is
    /// mounted at its own angle to the axes `DeviceMotion` reports in.
    ///
    /// Leaving this at identity is what produced a **-53.7% focal error** on
    /// real EuRoC imagery (212 px estimated against 458.7 px published): the
    /// infinite-homography solve was handed a rotation from the wrong frame.
    /// It degrades L3's flow prediction the same way, silently.
    pub body_from_camera: So3,
    /// Seed for every RNG in the session (spec.md §6). Layer configs inherit
    /// it through [`SlamConfig::normalized`] unless they were set explicitly.
    pub seed: u64,
}

impl SlamConfig {
    /// A session at the shipped baseline: tier 2, mapping on, up to scale.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        SlamConfig {
            tier: SensorTier::default(),
            width,
            height,
            intrinsics: IntrinsicsConfig::default(),
            track: TrackConfig::default(),
            map: MapConfig::default(),
            orientation: wslam_orientation::OrientationConfig::default(),
            // Identity is the honest default: with no extrinsic supplied we
            // assume the camera and the inertial frame coincide, which is right
            // for a synthetic rig and wrong for every real device.
            // ~1.7 deg, just above the 0.88 deg p95 error measured on EuRoC.
            min_prior_rotation_rad: 0.03,
            body_from_camera: So3::identity(),
            seed: 0x5eed,
        }
    }

    /// Propagate the session seed into every layer, so one seed reproduces the
    /// whole run.
    #[must_use]
    pub fn normalized(mut self) -> Self {
        self.track.seed = self.seed;
        self.intrinsics.calib.seed = self.seed ^ 0x9E37_79B9;
        self
    }

    /// The intrinsics to start with: the caller's measurement, else their
    /// prior, else a field-of-view guess that L2 will refine.
    #[must_use]
    pub fn initial_intrinsics(&self) -> CameraIntrinsics {
        if let Some(k) = self.intrinsics.known {
            return k;
        }
        match self.intrinsics.focal_prior_px {
            Some(f) => CameraIntrinsics::from_focal(f, self.width, self.height),
            None => CameraIntrinsics::from_hfov_degrees(
                self.intrinsics.calib.prior_hfov_degrees,
                self.width,
                self.height,
            ),
        }
    }

    /// Whether L2 has work to do.
    #[must_use]
    pub fn needs_intrinsics_estimation(&self) -> bool {
        self.intrinsics.known.is_none() && self.intrinsics.estimate_online
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_ordering_matches_the_spec_table() {
        assert!(SensorTier::VisionOnly < SensorTier::VisionOrientation);
        assert!(SensorTier::VisionOrientation < SensorTier::TightVisualInertial);
        assert_eq!(SensorTier::VisionOrientation.number(), 2);
        assert!(!SensorTier::VisionOnly.uses_motion());
        assert!(SensorTier::VisionOrientation.uses_motion());
    }

    #[test]
    fn the_default_is_the_shipped_baseline() {
        // spec.md §3: "Defaults: sensor tier 2 (vision + loose orientation),
        // scale source `none` (up-to-scale), map enabled with relocalization."
        let c = SlamConfig::new(1280, 720);
        assert_eq!(c.tier, SensorTier::VisionOrientation);
        assert!(c.map.enabled);
    }

    #[test]
    fn one_seed_reproduces_the_whole_run() {
        let c = SlamConfig {
            seed: 12345,
            ..SlamConfig::new(640, 480)
        }
        .normalized();
        assert_eq!(c.track.seed, 12345);
        // Layers get distinct streams, not the same one — otherwise two layers
        // drawing the same number would correlate their decisions.
        assert_ne!(c.intrinsics.calib.seed, c.track.seed);
    }

    #[test]
    fn known_intrinsics_short_circuit_l2() {
        let k = CameraIntrinsics::from_focal(700.0, 1280, 720);
        let c = SlamConfig {
            intrinsics: IntrinsicsConfig {
                known: Some(k),
                ..IntrinsicsConfig::default()
            },
            ..SlamConfig::new(1280, 720)
        };
        assert!(!c.needs_intrinsics_estimation());
        assert_eq!(c.initial_intrinsics().fx, 700.0);
    }

    #[test]
    fn the_fallback_prior_is_a_plausible_phone_camera() {
        let k = SlamConfig::new(1280, 720).initial_intrinsics();
        let hfov = k.hfov_degrees();
        assert!((60.0..80.0).contains(&hfov), "hfov {hfov}");
    }
}
