//! # wslam-calib — L2, focal length from rotation
//!
//! Estimates the focal length during a short init pan, using the rotation L1
//! already knows from the gyro. For a purely rotating camera
//! `x2 ~ K R K^-1 x1`, and with `R` known the infinite-homography constraint
//! (de Agapito, Hayman & Reid, IJCV 45(2), 2001) gives a linear solve for `f`
//! under the square-pixel assumption.
//!
//! ## Why this layer is mostly about two failure modes
//!
//! The linear method is settled work from 1994–2001. What makes it hard on a
//! phone is that both of its assumptions are false, and spec.md §6 L2 makes
//! measuring that a **gate**, not a nice-to-have:
//!
//! 1. **The lens is not a pinhole.** Hayman & Murray (CVIU 2004) found barrel
//!    distortion wrecks rotation-based focal estimation, and phone wide cameras
//!    are strongly barrel-distorted. [`CalibConfig::model_distortion`] switches
//!    joint `k1`/`k2` estimation on and off so the ablation runs as an
//!    experiment rather than an argument.
//!
//!    Measured here, the unmodelled arm is 5-8% wrong and **under**estimates,
//!    where Hayman & Murray report an overestimate. The difference is that they
//!    solve for the rotations too; we get them from L1, so the error cannot
//!    hide in the rotation and lands entirely on `f`. See
//!    `barrel_distortion_biases_focal_badly_without_the_model` for the full
//!    argument — the citation stands for the failure mode, not for its sign.
//!
//! 2. **The rotation is not pure.** Ji et al. observe that handheld rotation is
//!    about the wrist, ~20 cm from the optical centre, which injects a
//!    translation `t = (I - R) l` and therefore depth-dependent parallax.
//!    spec.md §5 gives the mitigation — *"the lever arm is precisely the
//!    camera-IMU extrinsic translation that VI systems already estimate. Fold
//!    it in rather than treating it as a separate hack"* — so
//!    [`CalibConfig::model_lever_arm`] adds it to the model, with the per-pair
//!    scene depth solved alongside because parallax and depth are only
//!    separable together.
//!
//! Both switches default to *on*. Turning them off is how you reproduce the
//! published failure, which the test suite does by name.
//!
//! ## Pipeline
//!
//! ```text
//!   matched pairs + gyro rotation
//!        │
//!        ├─ homography::estimate_homography_ransac   robust H per pair
//!        ├─ focal::focal_from_rotation_homography    linear f per pair
//!        ├─ focal::aggregate_focals                  robust median + variance
//!        └─ refine::refine                           joint LM over all pairs
//!                                                    (f, k1, k2, depths)
//! ```
//!
//! The linear stage exists to give the nonlinear stage a start that is in the
//! right basin. The reported number always comes from the refinement when it
//! converges, and from the linear aggregate when it does not.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod focal;
pub mod homography;
pub mod refine;
pub mod synthetic;

pub use focal::{
    aggregate_focals, axis_tilt, focal_from_rotation_homography, rotation_is_observable,
    FocalAggregate, MIN_AXIS_TILT,
};
pub use homography::{
    apply_homography, estimate_homography, estimate_homography_ransac, normalize_homography,
    symmetric_transfer_error, MIN_HOMOGRAPHY_MATCHES,
};
pub use refine::{
    refine, refine_from, refine_multistart, PairObservation, RefineOptions, RefineReport,
};
pub use synthetic::SyntheticRig;

use wslam_core::{
    CameraIntrinsics, DeterministicRng, Error, RadialTangential, Result, Scalar, So3, Vec2, Vec3,
};

/// Widest plausible horizontal field of view for a phone rear camera, degrees.
///
/// Ultra-wide modules reach ~120 deg; nothing sane exceeds this.
const MAX_PLAUSIBLE_HFOV_DEG: Scalar = 130.0;

/// Narrowest plausible horizontal field of view, degrees.
///
/// `getUserMedia` with `facingMode: environment` gives the *main* rear camera,
/// which across shipping phones spans roughly 60-80 deg; ultra-wide modules go
/// wider, never narrower. 35 deg leaves generous headroom below anything a
/// browser will hand us while still excluding the solver escapes, which land
/// near 26 deg. A dedicated telephoto module would fall outside this band and
/// is out of scope — the library does not select one.
const MIN_PLAUSIBLE_HFOV_DEG: Scalar = 35.0;

/// Configuration for [`FocalEstimator`].
///
/// The two `model_*` switches are the ablation arms spec.md §6 L2 requires.
/// They default to the modelled arm; the unmodelled arm exists to demonstrate
/// the failure, not to ship.
#[derive(Debug, Clone)]
pub struct CalibConfig {
    /// Estimate `k1`/`k2` jointly with the focal length.
    ///
    /// Setting this to `false` reproduces Hayman & Murray's overestimate on a
    /// barrel-distorted lens. That is its only legitimate use.
    pub model_distortion: bool,
    /// Model the wrist lever arm and per-pair scene depth.
    pub model_lever_arm: bool,
    /// Camera-IMU translation in camera coordinates, metres.
    ///
    /// Only consulted when `model_lever_arm` is set. The default is a 20 cm
    /// offset along the camera's `-Z`, which is roughly where a wrist sits
    /// behind a phone held at reading distance.
    pub lever_arm_m: Vec3,
    /// Field-of-view prior used to seed the solver when the linear stage has
    /// not produced a usable estimate yet. Phone rear cameras cluster near 66°.
    pub prior_hfov_degrees: Scalar,
    /// Pairs required before [`FocalEstimator::estimate`] returns anything.
    pub min_pairs: usize,
    /// RANSAC iteration cap for the per-pair homography.
    pub ransac_iterations: usize,
    /// RANSAC inlier threshold, pixels of transfer error.
    pub ransac_threshold_px: Scalar,
    /// Seed for every RNG in this layer (spec.md §6).
    pub seed: u64,
    /// Largest number of pairs retained. Bounded so a long init pan cannot grow
    /// the refinement problem without limit.
    pub max_pairs: usize,
}

impl Default for CalibConfig {
    fn default() -> Self {
        CalibConfig {
            model_distortion: true,
            model_lever_arm: true,
            lever_arm_m: Vec3::new(0.0, 0.0, -0.20),
            prior_hfov_degrees: 66.0,
            min_pairs: 6,
            ransac_iterations: 512,
            ransac_threshold_px: 2.0,
            seed: 0x5eed,
            max_pairs: 40,
        }
    }
}

impl CalibConfig {
    /// The ablation arm with neither correction — the textbook method, and the
    /// one the literature says fails on a phone.
    #[must_use]
    pub fn ablation_naive() -> Self {
        CalibConfig {
            model_distortion: false,
            model_lever_arm: false,
            ..Self::default()
        }
    }

    /// Distortion modelled, lever arm not.
    #[must_use]
    pub fn ablation_distortion_only() -> Self {
        CalibConfig {
            model_distortion: true,
            model_lever_arm: false,
            ..Self::default()
        }
    }

    /// Lever arm modelled, distortion not.
    #[must_use]
    pub fn ablation_lever_arm_only() -> Self {
        CalibConfig {
            model_distortion: false,
            model_lever_arm: true,
            ..Self::default()
        }
    }

    /// Short label for reporting an ablation cell.
    #[must_use]
    pub fn ablation_label(&self) -> &'static str {
        match (self.model_distortion, self.model_lever_arm) {
            (false, false) => "naive",
            (true, false) => "distortion",
            (false, true) => "lever-arm",
            (true, true) => "full",
        }
    }
}

/// The estimated focal length, with everything needed to judge it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FocalEstimate {
    /// Focal length in pixels.
    pub focal_px: Scalar,
    /// Variance of `focal_px`, in px². Never zero — a single pair carries no
    /// information about its own dispersion, and reporting confidence there
    /// would be exactly the overconfidence spec.md §6 L6 warns against.
    pub variance: Scalar,
    /// Estimated distortion. Identity when `model_distortion` was off.
    pub distortion: RadialTangential,
    /// Pairs that survived into the final estimate.
    pub pairs_used: usize,
    /// Whether the nonlinear refinement converged. `false` means the number
    /// came from the linear aggregate alone and deserves more suspicion.
    pub refined: bool,
}

impl FocalEstimate {
    /// Relative standard deviation as a percentage — the unit spec.md §6 L2
    /// reports focal error in.
    #[must_use]
    pub fn relative_stddev_percent(&self) -> Scalar {
        if self.focal_px.abs() < 1e-9 {
            Scalar::INFINITY
        } else {
            100.0 * self.variance.sqrt() / self.focal_px
        }
    }

    /// Full intrinsics for an image of the given size, assuming square pixels
    /// and a centred principal point.
    #[must_use]
    pub fn intrinsics(&self, width: u32, height: u32) -> CameraIntrinsics {
        let mut k = CameraIntrinsics::from_focal(self.focal_px, width, height);
        k.distortion = self.distortion;
        k
    }
}

/// Accumulates rotation-compensated frame pairs during the init pan and solves
/// for focal length.
///
/// Feed it matched correspondences plus the gyro-known relative rotation with
/// [`FocalEstimator::push_pair`], then call [`FocalEstimator::estimate`].
#[derive(Debug)]
pub struct FocalEstimator {
    config: CalibConfig,
    width: u32,
    height: u32,
    /// Correspondences recentred on the principal point, with their rotation.
    pairs: Vec<PairObservation>,
    /// Linear per-pair focal estimates, index-aligned with `pairs`.
    linear_focals: Vec<Scalar>,
    rng: DeterministicRng,
    /// Pairs rejected before entering the problem, for reporting.
    rejected: usize,
}

impl FocalEstimator {
    /// Build an estimator for a given image size.
    #[must_use]
    pub fn new(config: CalibConfig, width: u32, height: u32) -> Self {
        let rng = DeterministicRng::new("calib-homography-ransac", config.seed);
        FocalEstimator {
            config,
            width,
            height,
            pairs: Vec::new(),
            linear_focals: Vec::new(),
            rng,
            rejected: 0,
        }
    }

    /// Principal point assumed by the estimator.
    #[must_use]
    fn principal(&self) -> Vec2 {
        Vec2::new(self.width as Scalar * 0.5, self.height as Scalar * 0.5)
    }

    /// Focal length implied by the field-of-view prior.
    #[must_use]
    pub fn prior_focal(&self) -> Scalar {
        let half: Scalar = (self.config.prior_hfov_degrees.to_radians() * 0.5).tan();
        self.width as Scalar * 0.5 / half.max(1e-6)
    }

    /// Feed one matched pair and the rotation L1 measured across it.
    ///
    /// `matches` are raw pixel correspondences `(frame1, frame2)`; `rotation`
    /// is `R_cam2_cam1`.
    ///
    /// # Errors
    /// - [`Error::Insufficient`] when there are too few correspondences, when
    ///   the rotation is too small to constrain the focal length, or when
    ///   RANSAC finds no consensus. All three are transient: the caller should
    ///   keep panning.
    pub fn push_pair(&mut self, matches: &[(Vec2, Vec2)], rotation: &So3) -> Result<()> {
        if matches.len() < MIN_HOMOGRAPHY_MATCHES {
            self.rejected += 1;
            return Err(Error::insufficient(format!(
                "need {MIN_HOMOGRAPHY_MATCHES} correspondences, got {}",
                matches.len()
            )));
        }
        // A near-zero rotation makes the infinite-homography constraint
        // degenerate: H tends to the identity and the focal length drops out.
        // Refusing here is what stops a stationary phone from producing a
        // confident garbage focal.
        if !rotation_is_observable(rotation) {
            self.rejected += 1;
            return Err(Error::insufficient(format!(
                "rotation too small or about the optical axis (tilt {:.4} rad < {MIN_AXIS_TILT})",
                axis_tilt(rotation)
            )));
        }
        if self.pairs.len() >= self.config.max_pairs {
            // Bounded memory: a 30-second pan must not grow the refinement
            // problem without limit.
            return Ok(());
        }

        let (h, inliers) = estimate_homography_ransac(
            matches,
            self.config.ransac_threshold_px,
            self.config.ransac_iterations,
            &mut self.rng,
        )
        .ok_or_else(|| Error::insufficient("homography RANSAC found no consensus"))?;

        let f = focal_from_rotation_homography(&h, rotation);

        // Recentre on the principal point: both the linear solve and the
        // refinement work in centred pixels, and doing it once here keeps the
        // convention in one place.
        let c = self.principal();
        let kept: Vec<(Vec2, Vec2)> = matches
            .iter()
            .zip(inliers.iter())
            .filter(|(_, keep)| **keep)
            .map(|((a, b), _)| (a - c, b - c))
            .collect();

        if kept.len() < MIN_HOMOGRAPHY_MATCHES {
            self.rejected += 1;
            return Err(Error::insufficient("too few RANSAC inliers survived"));
        }

        self.pairs.push(PairObservation {
            matches: kept,
            rotation: *rotation,
        });
        // Push even when the linear solve failed: the pair still constrains the
        // refinement, and `aggregate_focals` skips non-finite entries.
        self.linear_focals.push(f.unwrap_or(Scalar::NAN));
        Ok(())
    }

    /// Pairs accumulated so far.
    #[must_use]
    pub fn pair_count(&self) -> usize {
        self.pairs.len()
    }

    /// Pairs rejected as unusable.
    #[must_use]
    pub fn rejected_count(&self) -> usize {
        self.rejected
    }

    /// Whether enough pairs have accumulated to attempt an estimate.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.pairs.len() >= self.config.min_pairs
    }

    /// Solve for the focal length.
    ///
    /// Returns `None` until [`FocalEstimator::is_ready`], and if every pair
    /// turns out to be unusable.
    #[must_use]
    pub fn estimate(&self) -> Option<FocalEstimate> {
        if !self.is_ready() {
            return None;
        }

        // Linear stage: robust median over the per-pair closed-form solutions.
        let aggregate = aggregate_focals(&self.linear_focals);
        let (seed_focal, linear_variance, linear_pairs) = match &aggregate {
            Some(a) => (a.focal_px, a.variance, a.kept.len()),
            // Every linear solve failed. The refinement can still recover from
            // the field-of-view prior, so this is not fatal — but the variance
            // must reflect that we are starting from a guess.
            None => {
                let f = self.prior_focal();
                (f, (0.25 * f).powi(2), 0)
            }
        };

        let options = RefineOptions {
            model_distortion: self.config.model_distortion,
            lever_arm: self
                .config
                .model_lever_arm
                .then_some(self.config.lever_arm_m),
            ..RefineOptions::default()
        };

        // Multi-start, not single-start: the (f, k1) landscape is multi-modal
        // and LM reports `converged` in the wrong basin just as confidently as
        // in the right one. See `refine_multistart` for the measurements.
        match refine_multistart(&self.pairs, seed_focal, &options) {
            Some(report) if report.converged && self.is_plausible(report.focal_px) => {
                Some(FocalEstimate {
                    focal_px: report.focal_px,
                    // Take the larger of the two variance estimates. The
                    // normal-equations variance describes fit quality at the
                    // solution and says nothing about a systematic bias shared
                    // by every pair — which is exactly what an unmodelled
                    // distortion or lever arm produces. Trusting it alone would
                    // report high confidence in a biased answer.
                    variance: report.focal_variance.max(linear_variance).max(1e-6),
                    distortion: report.distortion,
                    pairs_used: self.pairs.len(),
                    refined: true,
                })
            }
            // The refinement failed, did not converge, or escaped the
            // plausible field-of-view band. Fall back to the linear aggregate,
            // and only if that is itself plausible — returning nothing is a
            // valid answer for this layer and a wrong focal is not.
            _ => aggregate
                .filter(|a| self.is_plausible(a.focal_px))
                .map(|a| FocalEstimate {
                    focal_px: a.focal_px,
                    variance: a.variance.max(1e-6),
                    distortion: RadialTangential::NONE,
                    pairs_used: linear_pairs,
                    refined: false,
                }),
        }
    }

    /// Whether a focal length is physically plausible for this image size.
    ///
    /// The refinement can converge, report success, and still return a focal
    /// implying a 26-degree field of view on a phone wide-angle lens — measured
    /// at `k1 = -0.35`, where the `(f, k1)` valley has a far basin the
    /// multi-start seeds can still land in. A wrong number that announces
    /// itself as wrong is recoverable; a wrong number that claims success is
    /// not, so the band is checked rather than assumed.
    #[must_use]
    fn is_plausible(&self, focal_px: Scalar) -> bool {
        if !(focal_px > 0.0) || !focal_px.is_finite() {
            return false;
        }
        let hfov = 2.0 * (self.width as Scalar * 0.5 / focal_px).atan().to_degrees();
        (MIN_PLAUSIBLE_HFOV_DEG..=MAX_PLAUSIBLE_HFOV_DEG).contains(&hfov)
    }

    /// Drop all accumulated pairs. Called on re-initialisation, so a stale pan
    /// cannot calibrate a fresh session.
    pub fn reset(&mut self) {
        self.pairs.clear();
        self.linear_focals.clear();
        self.rejected = 0;
        self.rng = DeterministicRng::new("calib-homography-ransac", self.config.seed);
    }

    /// The configuration in force.
    #[must_use]
    pub fn config(&self) -> &CalibConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use wslam_core::Vec3;

    const TRUE_FOCAL: Scalar = 720.0;
    const WIDTH: u32 = 1280;
    const HEIGHT: u32 = 720;

    /// A pan: a sequence of relative rotations large enough to be observable.
    fn pan_rotations(n: usize) -> Vec<So3> {
        (0..n)
            .map(|i| {
                let t = i as Scalar;
                So3::exp(&Vec3::new(
                    0.03 + 0.004 * (t * 0.7).sin(),
                    0.05 + 0.006 * (t * 0.4).cos(),
                    0.008 * (t * 1.1).sin(),
                ))
            })
            .collect()
    }

    fn run(rig: &SyntheticRig, config: CalibConfig, pairs: usize) -> Option<FocalEstimate> {
        let mut rng = DeterministicRng::new("calib-test-scene", 4242);
        let mut estimator = FocalEstimator::new(config, rig.width, rig.height);
        for rotation in pan_rotations(pairs) {
            let matches = rig.raw_pair(&rotation, 160, &mut rng);
            let _ = estimator.push_pair(&matches, &rotation);
        }
        estimator.estimate()
    }

    fn relative_error(estimate: &FocalEstimate) -> Scalar {
        (estimate.focal_px - TRUE_FOCAL).abs() / TRUE_FOCAL
    }

    #[test]
    fn recovers_focal_from_a_noise_free_pinhole_pan() {
        let rig = SyntheticRig::pinhole(TRUE_FOCAL, WIDTH, HEIGHT);
        let estimate = run(&rig, CalibConfig::default(), 10).expect("estimate");
        assert!(
            relative_error(&estimate) < 0.01,
            "focal {} vs {TRUE_FOCAL} ({:.2}%)",
            estimate.focal_px,
            100.0 * relative_error(&estimate)
        );
    }

    #[test]
    fn reports_a_variance_that_is_never_zero() {
        // A zero variance is a claim of perfect knowledge. spec.md §6 L6 calls
        // overconfidence worse than no covariance at all.
        let rig = SyntheticRig::pinhole(TRUE_FOCAL, WIDTH, HEIGHT);
        let estimate = run(&rig, CalibConfig::default(), 10).expect("estimate");
        assert!(estimate.variance > 0.0);
        assert!(estimate.variance.is_finite());
        assert!(estimate.relative_stddev_percent() > 0.0);
    }

    /// The headline ablation: unmodelled barrel distortion badly biases the
    /// focal length, and modelling it removes the bias.
    ///
    /// **On the direction of the bias.** Hayman & Murray (CVIU 2004) report an
    /// *overestimate*. Measured here, with the rotation known from the gyro,
    /// the naive arm consistently *underestimates* — by 4.8% at `k1 = -0.10`
    /// rising monotonically to 8.1% at `k1 = -0.35`, reproducibly across seeds.
    ///
    /// That is not a contradiction. Hayman & Murray analyse full
    /// self-calibration, where the rotations are unknown and solved jointly
    /// with `f`; the distortion error is then absorbed partly by the rotation
    /// estimates, and the residual bias in `f` comes out positive. Here `R` is
    /// supplied by L1, so the whole error has to land on `f`, and barrel
    /// distortion compresses peripheral flow — which the pinhole model can only
    /// explain with a *shorter* focal length.
    ///
    /// So the test asserts the magnitude and the direction we actually measure,
    /// and the citation stands for what it supports: distortion is a first-order
    /// failure mode for rotation-based calibration, which is why spec.md §6 L2
    /// makes this ablation a gate.
    #[test]
    fn barrel_distortion_biases_focal_badly_without_the_model() {
        let mut rig = SyntheticRig::pinhole(TRUE_FOCAL, WIDTH, HEIGHT);
        rig.distortion = RadialTangential::radial(-0.28, 0.09);

        let naive = run(&rig, CalibConfig::ablation_naive(), 12).expect("naive estimate");
        let modelled =
            run(&rig, CalibConfig::ablation_distortion_only(), 12).expect("modelled estimate");

        assert!(
            relative_error(&naive) > 0.03,
            "expected a large bias without the distortion model, got {:.2}% ({} vs {TRUE_FOCAL})",
            100.0 * relative_error(&naive),
            naive.focal_px
        );
        assert!(
            naive.focal_px < TRUE_FOCAL,
            "with rotation known, barrel distortion should shorten the focal estimate; \
             got {} vs {TRUE_FOCAL}. If this flips, re-derive the analysis in this \
             test's doc comment rather than relaxing the assertion.",
            naive.focal_px
        );
        assert!(
            relative_error(&modelled) < 0.25 * relative_error(&naive),
            "modelling distortion must substantially remove the bias: \
             naive {:.2}% vs modelled {:.2}%",
            100.0 * relative_error(&naive),
            100.0 * relative_error(&modelled)
        );
    }

    /// The bias must grow with the distortion, not merely be present at one
    /// value — that is what makes it attributable to distortion at all.
    #[test]
    fn naive_bias_grows_with_distortion_strength() {
        let mut errors = Vec::new();
        for k1 in [-0.10, -0.20, -0.35] {
            let mut rig = SyntheticRig::pinhole(TRUE_FOCAL, WIDTH, HEIGHT);
            rig.distortion = RadialTangential::radial(k1, 0.0);
            let estimate = run(&rig, CalibConfig::ablation_naive(), 12).expect("naive estimate");
            errors.push(relative_error(&estimate));
        }
        assert!(
            errors.windows(2).all(|w| w[1] > w[0]),
            "bias must increase with |k1|, got {errors:?}"
        );
    }

    /// A focal implying a 26-degree field of view on a phone wide-angle lens is
    /// not a calibration, it is a solver escape. Measured at `k1 = -0.35`,
    /// where the refinement converged and returned 2733 px.
    #[test]
    fn implausible_focal_is_declined_rather_than_returned() {
        let estimator = FocalEstimator::new(CalibConfig::default(), WIDTH, HEIGHT);
        assert!(estimator.is_plausible(720.0));
        assert!(estimator.is_plausible(480.0));
        assert!(!estimator.is_plausible(2733.0));
        assert!(!estimator.is_plausible(120.0));
        assert!(!estimator.is_plausible(-10.0));
        assert!(!estimator.is_plausible(Scalar::NAN));
    }

    #[test]
    fn distortion_estimate_recovers_the_sign_of_the_barrel() {
        let mut rig = SyntheticRig::pinhole(TRUE_FOCAL, WIDTH, HEIGHT);
        rig.distortion = RadialTangential::radial(-0.28, 0.09);
        let estimate = run(&rig, CalibConfig::ablation_distortion_only(), 12).expect("estimate");
        assert!(
            estimate.distortion.k1 < -0.05,
            "expected a negative k1, got {}",
            estimate.distortion.k1
        );
    }

    /// The second required ablation. Ji et al.: handheld rotation is about the
    /// wrist, which injects `t = (I - R) l` and biases the focal length.
    #[test]
    fn wrist_lever_arm_biases_focal_without_the_model() {
        let mut rig = SyntheticRig::pinhole(TRUE_FOCAL, WIDTH, HEIGHT);
        rig.lever_arm = Some(Vec3::new(0.0, 0.0, -0.20));
        rig.depth = 1.2;

        let naive = run(&rig, CalibConfig::ablation_naive(), 12).expect("naive estimate");
        let modelled = run(&rig, CalibConfig::ablation_lever_arm_only(), 12);

        assert!(
            relative_error(&naive) > 0.01,
            "a 20 cm lever arm at 1.2 m should bias the naive estimate, got {:.3}%",
            100.0 * relative_error(&naive)
        );
        if let Some(modelled) = modelled {
            assert!(
                relative_error(&modelled) <= relative_error(&naive),
                "modelling the lever arm must not make things worse: \
                 naive {:.2}% vs modelled {:.2}%",
                100.0 * relative_error(&naive),
                100.0 * relative_error(&modelled)
            );
        }
    }

    /// spec.md §6 L2 requires the ablation "across scene depth", because
    /// parallax from the lever arm grows as the scene gets closer.
    #[test]
    fn lever_arm_bias_grows_as_the_scene_gets_closer() {
        let mut near = SyntheticRig::pinhole(TRUE_FOCAL, WIDTH, HEIGHT);
        near.lever_arm = Some(Vec3::new(0.0, 0.0, -0.20));
        near.depth = 0.8;
        let mut far = near.clone();
        far.depth = 8.0;

        let near_err = relative_error(&run(&near, CalibConfig::ablation_naive(), 12).unwrap());
        let far_err = relative_error(&run(&far, CalibConfig::ablation_naive(), 12).unwrap());
        assert!(
            near_err > far_err,
            "parallax bias must grow as depth shrinks: 0.8 m {:.3}% vs 8 m {:.3}%",
            100.0 * near_err,
            100.0 * far_err
        );
    }

    #[test]
    fn declines_until_enough_pairs_have_accumulated() {
        let rig = SyntheticRig::pinhole(TRUE_FOCAL, WIDTH, HEIGHT);
        let mut rng = DeterministicRng::new("calib-test-scene", 1);
        let config = CalibConfig::default();
        let min = config.min_pairs;
        let mut estimator = FocalEstimator::new(config, WIDTH, HEIGHT);
        for (i, rotation) in pan_rotations(min + 2).into_iter().enumerate() {
            if i < min {
                assert!(estimator.estimate().is_none(), "answered after {i} pairs");
            }
            let matches = rig.raw_pair(&rotation, 120, &mut rng);
            estimator
                .push_pair(&matches, &rotation)
                .expect("usable pair");
        }
        assert!(estimator.is_ready());
        assert!(estimator.estimate().is_some());
    }

    #[test]
    fn rejects_a_stationary_camera_rather_than_guessing() {
        // Near-zero rotation makes the constraint degenerate. A confident focal
        // from a phone sitting on a table is worse than no focal at all.
        let rig = SyntheticRig::pinhole(TRUE_FOCAL, WIDTH, HEIGHT);
        let mut rng = DeterministicRng::new("calib-test-scene", 2);
        let mut estimator = FocalEstimator::new(CalibConfig::default(), WIDTH, HEIGHT);
        let tiny = So3::exp(&Vec3::new(1e-5, 1e-5, 0.0));
        let matches = rig.raw_pair(&tiny, 120, &mut rng);
        let err = estimator.push_pair(&matches, &tiny).unwrap_err();
        assert!(err.is_transient(), "{err}");
        assert_eq!(estimator.pair_count(), 0);
        assert_eq!(estimator.rejected_count(), 1);
    }

    #[test]
    fn rejects_too_few_correspondences() {
        let mut estimator = FocalEstimator::new(CalibConfig::default(), WIDTH, HEIGHT);
        let rotation = So3::exp(&Vec3::new(0.05, 0.05, 0.0));
        let err = estimator
            .push_pair(&[(Vec2::zeros(), Vec2::zeros())], &rotation)
            .unwrap_err();
        assert!(err.is_transient());
    }

    #[test]
    fn pair_count_is_bounded() {
        // A long init pan must not grow the refinement problem without limit.
        let rig = SyntheticRig::pinhole(TRUE_FOCAL, WIDTH, HEIGHT);
        let mut rng = DeterministicRng::new("calib-test-scene", 3);
        let config = CalibConfig {
            max_pairs: 5,
            ..CalibConfig::default()
        };
        let mut estimator = FocalEstimator::new(config, WIDTH, HEIGHT);
        for rotation in pan_rotations(30) {
            let matches = rig.raw_pair(&rotation, 100, &mut rng);
            let _ = estimator.push_pair(&matches, &rotation);
        }
        assert_eq!(estimator.pair_count(), 5);
    }

    #[test]
    fn reset_clears_the_pan() {
        let rig = SyntheticRig::pinhole(TRUE_FOCAL, WIDTH, HEIGHT);
        let mut rng = DeterministicRng::new("calib-test-scene", 5);
        let mut estimator = FocalEstimator::new(CalibConfig::default(), WIDTH, HEIGHT);
        for rotation in pan_rotations(8) {
            let matches = rig.raw_pair(&rotation, 120, &mut rng);
            let _ = estimator.push_pair(&matches, &rotation);
        }
        assert!(estimator.pair_count() > 0);
        estimator.reset();
        assert_eq!(estimator.pair_count(), 0);
        assert!(estimator.estimate().is_none());
    }

    #[test]
    fn is_deterministic_across_runs() {
        // spec.md §6: same seed, same answer. RANSAC included.
        let rig = SyntheticRig::pinhole(TRUE_FOCAL, WIDTH, HEIGHT);
        let a = run(&rig, CalibConfig::default(), 10).unwrap();
        let b = run(&rig, CalibConfig::default(), 10).unwrap();
        assert_eq!(a.focal_px.to_bits(), b.focal_px.to_bits());
        assert_eq!(a.variance.to_bits(), b.variance.to_bits());
    }

    #[test]
    fn survives_pixel_noise_with_an_honest_variance() {
        let mut rig = SyntheticRig::pinhole(TRUE_FOCAL, WIDTH, HEIGHT);
        rig.noise_px = 0.5;
        let estimate = run(&rig, CalibConfig::default(), 14).expect("estimate");
        let error_px = (estimate.focal_px - TRUE_FOCAL).abs();
        let sigma = estimate.variance.sqrt();
        assert!(
            relative_error(&estimate) < 0.10,
            "focal {} vs {TRUE_FOCAL}",
            estimate.focal_px
        );
        // The claimed uncertainty must be commensurate with the actual error.
        // A variance that does not bracket the truth is decorative.
        assert!(
            error_px < 4.0 * sigma,
            "error {error_px:.2} px exceeds 4 sigma ({sigma:.2} px) — variance is overconfident"
        );
    }

    #[test]
    fn prior_focal_matches_the_configured_field_of_view() {
        let estimator = FocalEstimator::new(CalibConfig::default(), WIDTH, HEIGHT);
        let k = CameraIntrinsics::from_hfov_degrees(66.0, WIDTH, HEIGHT);
        assert_relative_eq!(estimator.prior_focal(), k.fx, epsilon = 1e-9);
    }

    #[test]
    fn ablation_labels_are_distinct() {
        let labels = [
            CalibConfig::ablation_naive().ablation_label(),
            CalibConfig::ablation_distortion_only().ablation_label(),
            CalibConfig::ablation_lever_arm_only().ablation_label(),
            CalibConfig::default().ablation_label(),
        ];
        let mut sorted = labels;
        sorted.sort_unstable();
        let mut deduped = sorted;
        let unique = {
            deduped.sort_unstable();
            let mut v: Vec<&str> = deduped.to_vec();
            v.dedup();
            v.len()
        };
        assert_eq!(unique, 4, "{labels:?}");
    }

    #[test]
    fn estimate_produces_usable_intrinsics() {
        let rig = SyntheticRig::pinhole(TRUE_FOCAL, WIDTH, HEIGHT);
        let estimate = run(&rig, CalibConfig::default(), 10).expect("estimate");
        let k = estimate.intrinsics(WIDTH, HEIGHT);
        assert_eq!((k.width, k.height), (WIDTH, HEIGHT));
        assert_relative_eq!(k.fx, k.fy, epsilon = 1e-12);
        assert_relative_eq!(k.cx, WIDTH as Scalar * 0.5, epsilon = 1e-12);
        // A round trip through the camera model must land back on the pixel.
        let px = Vec2::new(300.0, 200.0);
        let bearing = k.unproject_bearing(px);
        let back = k.project(&bearing).expect("in front of the camera");
        assert_relative_eq!(back, px, epsilon = 1e-6);
    }
}
