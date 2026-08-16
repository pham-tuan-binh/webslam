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
    /// Inliers a loop closure needs on top of relocalization's `min_inliers`.
    ///
    /// A wrong relocalization strands one session; a wrong loop edge bends the
    /// whole graph through pose-graph optimisation, so loops are held to a
    /// stricter bar. ORB-SLAM uses 40 here against 15–20 for relocalization,
    /// for the same asymmetry-of-damage reason.
    pub loop_min_inliers: usize,
}

impl Default for MapConfig {
    fn default() -> Self {
        MapConfig {
            enabled: true,
            max_keyframes: 300,
            reloc_delay_frames: 3,
            // 48, not 12: with keyframes every few frames, 12 leaves loop
            // closure free to "recognise" places seen seconds ago, where the
            // accumulated drift is smaller than the closure's own PnP noise —
            // edges that can only make the graph worse. Measured on MH_01.
            loop_exclude_recent: 48,
            backend_budget_ms: 4.0,
            reloc_scale_variance: 4e-4,
            loop_min_inliers: 40,
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
    /// Largest predicted inter-frame rotation, in radians, the prior is
    /// trusted for. Gyro integration error grows with angular rate, so the
    /// prediction is withheld during the most violent rotations rather than
    /// trusted most there.
    pub max_prior_rotation_rad: Scalar,
    /// Recent prior error (EWMA, radians) above which the orientation prior
    /// is withheld from L3.
    ///
    /// The prior's failure mode is a heavy tail, not a bias: its inter-frame
    /// error sits well under 0.5° most of the time and then spikes for
    /// stretches, and during a spike every KLT seed is misdirected at once —
    /// measured on EuRoC MH_01 CPU, the ungated prior turned a 0.66 m
    /// trajectory into a 2.30 m one. 0.5° ≈ 4 px at EuRoC's focal length,
    /// about the largest displacement the pyramid absorbs without help.
    pub max_prior_error_rad: Scalar,
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
            max_prior_rotation_rad: 0.05,
            max_prior_error_rad: 0.5_f64.to_radians(),
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

/// `R_body_camera` for a phone's **rear** camera in a browser.
///
/// The two frames this bridges:
///
/// - **Body**: what `DeviceMotion` reports in (W3C) — `x` right of the screen
///   in the natural portrait orientation, `y` up the screen, `z` out of the
///   screen toward the viewer. Fixed to the hardware; never rotates with the
///   viewport.
/// - **Camera**: the frame the tracker projects in — `x` image-right, `y`
///   image-down, `z` along the optical axis into the scene. The browser
///   delivers frames upright in the *current viewport*, so this frame moves
///   when the viewport rotates.
///
/// `screen_angle_deg` is `screen.orientation.angle`: degrees the screen is
/// rotated **counter-clockwise** from its natural orientation.
///
/// Derivation. Viewport-right in body coordinates is `(cos θ, −sin θ, 0)` —
/// rotate the device CCW by θ and the viewer's "right" lands on the body axis
/// that rotated *into* it. Image-down is minus viewport-up:
/// `(−sin θ, −cos θ, 0)`. The rear optical axis points out the back:
/// `(0, 0, −1)`. Those three columns are right-handed for every θ
/// (`x̂ × ŷ = ẑ` identically), and factor as `Rz(−θ) · Rx(π)`. In portrait
/// this is the familiar `diag(1, −1, −1)`.
///
/// Leaving `body_from_camera` at identity instead of this fed L2 rotations
/// from the wrong frame on every phone — the same class of error that
/// produced the −53.7% focal failure documented on
/// [`SlamConfig::body_from_camera`], now from hardware rather than from a
/// dataset extrinsic.
#[must_use]
pub fn browser_rear_camera_extrinsic(screen_angle_deg: Scalar) -> So3 {
    let theta = screen_angle_deg.to_radians();
    let flip = So3::exp(&wslam_core::Vec3::new(std::f64::consts::PI, 0.0, 0.0));
    So3::exp(&wslam_core::Vec3::new(0.0, 0.0, -theta)).compose(&flip)
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

    fn assert_vec_close(got: wslam_core::Vec3, want: (f64, f64, f64)) {
        let want = wslam_core::Vec3::new(want.0, want.1, want.2);
        assert!((got - want).norm() < 1e-12, "got {got:?}, want {want:?}");
    }

    #[test]
    fn portrait_rear_camera_is_a_half_turn_about_x() {
        // Portrait: image-right is body-x, image-down is body−y (down the
        // screen), and the optical axis points out the back of the device.
        let r = browser_rear_camera_extrinsic(0.0);
        assert_vec_close(
            r.act(&wslam_core::Vec3::new(1.0, 0.0, 0.0)),
            (1.0, 0.0, 0.0),
        );
        assert_vec_close(
            r.act(&wslam_core::Vec3::new(0.0, 1.0, 0.0)),
            (0.0, -1.0, 0.0),
        );
        assert_vec_close(
            r.act(&wslam_core::Vec3::new(0.0, 0.0, 1.0)),
            (0.0, 0.0, -1.0),
        );
    }

    #[test]
    fn landscape_rear_camera_swaps_the_image_axes_onto_the_body() {
        // Device rotated 90° CCW (angle 90): the viewer's "right" is the edge
        // that rotated into it — body −y. Image-down lands on body −x.
        let r = browser_rear_camera_extrinsic(90.0);
        assert_vec_close(
            r.act(&wslam_core::Vec3::new(1.0, 0.0, 0.0)),
            (0.0, -1.0, 0.0),
        );
        assert_vec_close(
            r.act(&wslam_core::Vec3::new(0.0, 1.0, 0.0)),
            (-1.0, 0.0, 0.0),
        );
        assert_vec_close(
            r.act(&wslam_core::Vec3::new(0.0, 0.0, 1.0)),
            (0.0, 0.0, -1.0),
        );
    }

    #[test]
    fn the_extrinsic_is_a_proper_rotation_at_every_angle() {
        for deg in [0.0, 90.0, 180.0, 270.0, 37.5] {
            let r = browser_rear_camera_extrinsic(deg);
            let x = r.act(&wslam_core::Vec3::new(1.0, 0.0, 0.0));
            let y = r.act(&wslam_core::Vec3::new(0.0, 1.0, 0.0));
            let z = r.act(&wslam_core::Vec3::new(0.0, 0.0, 1.0));
            // Orthonormal and right-handed: x̂ × ŷ = ẑ.
            assert!((x.cross(&y) - z).norm() < 1e-12, "angle {deg}");
            assert!((x.norm() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn a_full_turn_is_portrait_again() {
        let a = browser_rear_camera_extrinsic(360.0);
        let b = browser_rear_camera_extrinsic(0.0);
        assert!(a.compose(&b.inverse()).angle() < 1e-9);
    }
}
