//! One user tap on a known distance.
//!
//! spec.md §2 lists this ruler as *exact*, costing *"one tap"*, and §10 makes
//! it the open decision that determines the shape of the whole project: *"If
//! `declared` is acceptable to our consumers, most of the difficulty
//! evaporates."* The arithmetic really is one division. The interesting part is
//! the variance, which is small but must not be zero — the taps are made by a
//! thumb on a phone, and pretending otherwise makes every downstream covariance
//! overconfident, which spec.md §6 L6 calls *"worse than no covariance at
//! all"*.

use crate::ScaleSource;
use wslam_core::{Error, Result, Scalar, ScaleEstimate, ScaleKind, StateWindow, Vec3};

/// Positional uncertainty of one tap, as a fraction of the observed distance,
/// used when the caller supplies no better figure.
///
/// A user's tap lands within roughly two pixels of the point they meant. Over
/// a baseline spanning a 640 px frame that is `2 / 640` of the observed
/// distance per tap. Callers who know the frame's focal length and the tapped
/// points' depth should use [`DeclaredScale::with_tap_precision_px`] instead of
/// inheriting this.
pub const DEFAULT_TAP_RELATIVE_STDDEV: Scalar = 2.0 / 640.0;

/// Scale from a distance the user declares between two observed points.
///
/// `scale = metres_between_the_points / up_to_scale_distance_between_them`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeclaredScale {
    metres: Scalar,
    observed_units: Scalar,
    tap_stddev_units: Scalar,
}

impl DeclaredScale {
    /// From two points whose separation the user has declared in metres, and
    /// the up-to-scale distance between the same two points.
    ///
    /// `metres_between` carries the *metric* geometry — its two endpoints may
    /// be anywhere, only their separation is used. `observed_units` is what the
    /// tracker measured between the same pair, in the arbitrary units of an
    /// unscaled trajectory.
    ///
    /// # Errors
    /// [`Error::Config`] if either distance is non-finite, or if either is
    /// zero: a declaration of "these two coincident points are 30 cm apart" is
    /// not a ruler, and dividing by an up-to-scale distance of zero yields an
    /// infinity that would propagate silently.
    pub fn new(metres_between: (Vec3, Vec3), observed_units: Scalar) -> Result<Self> {
        Self::from_distance((metres_between.1 - metres_between.0).norm(), observed_units)
    }

    /// As [`DeclaredScale::new`], but from the metric distance directly — the
    /// shape the UI actually has when the user types "0.297" for an A4 sheet.
    ///
    /// # Errors
    /// See [`DeclaredScale::new`].
    pub fn from_distance(metres: Scalar, observed_units: Scalar) -> Result<Self> {
        if !metres.is_finite() || metres <= 0.0 {
            return Err(Error::Config(format!(
                "declared distance must be a positive number of metres, got {metres}"
            )));
        }
        if !observed_units.is_finite() || observed_units <= 0.0 {
            return Err(Error::Config(format!(
                "observed up-to-scale distance must be positive, got {observed_units}"
            )));
        }
        Ok(DeclaredScale {
            metres,
            observed_units,
            tap_stddev_units: DEFAULT_TAP_RELATIVE_STDDEV * observed_units,
        })
    }

    /// Override the per-tap positional uncertainty, in up-to-scale units.
    #[must_use]
    pub fn with_tap_stddev_units(mut self, tap_stddev_units: Scalar) -> Self {
        self.tap_stddev_units = tap_stddev_units.abs();
        self
    }

    /// Set the per-tap uncertainty by reprojecting a pixel-space tap precision.
    ///
    /// A tap `sigma_px` pixels from its intended point, on a feature at
    /// `depth_units` in the up-to-scale frame seen through a lens of
    /// `focal_px`, misplaces the reconstructed point by
    /// `sigma_px * depth_units / focal_px` laterally. That is the honest model,
    /// and it is why the resulting variance is small rather than zero.
    #[must_use]
    pub fn with_tap_precision_px(
        self,
        sigma_px: Scalar,
        focal_px: Scalar,
        depth_units: Scalar,
    ) -> Self {
        if focal_px.abs() < 1e-9 {
            return self;
        }
        self.with_tap_stddev_units(sigma_px * depth_units / focal_px)
    }

    /// The declared metric distance.
    #[must_use]
    pub fn metres(&self) -> Scalar {
        self.metres
    }

    /// The observed up-to-scale distance.
    #[must_use]
    pub fn observed_units(&self) -> Scalar {
        self.observed_units
    }

    /// Per-tap positional standard deviation, in up-to-scale units.
    #[must_use]
    pub fn tap_stddev_units(&self) -> Scalar {
        self.tap_stddev_units
    }

    /// The estimate, without needing a window.
    ///
    /// `s = D / d`. Both taps are independent and isotropic, so the observed
    /// distance carries `Var(d) = 2 sigma_p^2` (each tap's error projected onto
    /// the baseline direction), and the delta method gives
    /// `Var(s) = (D / d^2)^2 Var(d) = (s / d)^2 * 2 sigma_p^2`.
    #[must_use]
    pub fn scale_estimate(&self) -> ScaleEstimate {
        let value = self.metres / self.observed_units;
        let variance = (value / self.observed_units).powi(2)
            * 2.0
            * self.tap_stddev_units
            * self.tap_stddev_units;
        ScaleEstimate::metric(ScaleKind::Declared, value, variance)
    }
}

impl ScaleSource for DeclaredScale {
    fn kind(&self) -> ScaleKind {
        ScaleKind::Declared
    }

    fn estimate(&mut self, _window: &StateWindow) -> Option<ScaleEstimate> {
        // The declaration is complete the moment it is made; no window content
        // can improve or invalidate it.
        Some(self.scale_estimate())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// The closed-form answer, from synthetic geometry with a known multiplier.
    #[test]
    fn recovers_a_known_multiplier_exactly() {
        // Two points 0.5 m apart in the world; the up-to-scale tracker measured
        // 0.25 units between them, so one unit is two metres.
        let a = Vec3::new(1.0, -2.0, 0.5);
        let b = a + Vec3::new(0.3, 0.4, 0.0); // |.| == 0.5 exactly
        let d = DeclaredScale::new((a, b), 0.25).unwrap();
        let e = d.scale_estimate();

        assert_eq!(e.source, ScaleKind::Declared);
        assert_relative_eq!(e.value, 2.0, epsilon = 1e-15);
        // The declaration must reproduce the metric distance exactly.
        assert_relative_eq!(e.value * 0.25, 0.5, epsilon = 1e-15);
    }

    #[test]
    fn a4_sheet_worked_example() {
        // The long edge of A4 is 297 mm; the tracker saw 0.11 units.
        let d = DeclaredScale::from_distance(0.297, 0.11).unwrap();
        assert_relative_eq!(d.scale_estimate().value, 2.7, epsilon = 1e-12);
    }

    /// Small, but not zero. A zero here would make every downstream pose
    /// covariance overconfident (spec.md §6 L6).
    #[test]
    fn tap_variance_is_small_but_strictly_positive() {
        let d = DeclaredScale::from_distance(1.0, 1.0).unwrap();
        let e = d.scale_estimate();
        assert!(e.variance > 0.0, "taps are not exact");
        // Default model: sqrt(2) * 2/640 == 0.44% relative.
        assert_relative_eq!(
            e.relative_stddev_percent(),
            100.0 * std::f64::consts::SQRT_2 * 2.0 / 640.0,
            epsilon = 1e-9
        );
        assert!(e.relative_stddev_percent() < 1.0);
    }

    #[test]
    fn reprojected_tap_precision_matches_the_pixel_model() {
        // 3 px of tap error at f = 600 px on a point 2 units away is
        // 0.01 units of lateral error per tap.
        let d = DeclaredScale::from_distance(1.0, 0.5)
            .unwrap()
            .with_tap_precision_px(3.0, 600.0, 2.0);
        assert_relative_eq!(d.tap_stddev_units(), 0.01, epsilon = 1e-15);

        let e = d.scale_estimate();
        // s = 2, so Var(s) = (2/0.5)^2 * 2 * 1e-4.
        assert_relative_eq!(e.variance, 16.0 * 2.0 * 1e-4, epsilon = 1e-15);
    }

    #[test]
    fn a_sloppier_tap_yields_a_larger_variance() {
        let tight = DeclaredScale::from_distance(1.0, 0.5)
            .unwrap()
            .with_tap_stddev_units(0.001);
        let sloppy = DeclaredScale::from_distance(1.0, 0.5)
            .unwrap()
            .with_tap_stddev_units(0.01);
        assert_eq!(tight.scale_estimate().value, sloppy.scale_estimate().value);
        assert!(sloppy.scale_estimate().variance > tight.scale_estimate().variance);
    }

    /// A longer observed baseline makes the same tap error matter less — the
    /// variance model has to reproduce that, or the UI cannot tell the user to
    /// tap further apart.
    #[test]
    fn a_longer_baseline_is_more_precise_at_fixed_pixel_precision() {
        let short = DeclaredScale::from_distance(1.0, 0.2)
            .unwrap()
            .with_tap_precision_px(2.0, 600.0, 1.0);
        let long = DeclaredScale::from_distance(5.0, 1.0)
            .unwrap()
            .with_tap_precision_px(2.0, 600.0, 1.0);
        // Same scale value (5.0), but the long baseline is far more certain.
        assert_relative_eq!(short.scale_estimate().value, 5.0, epsilon = 1e-12);
        assert_relative_eq!(long.scale_estimate().value, 5.0, epsilon = 1e-12);
        assert!(long.scale_estimate().variance < short.scale_estimate().variance);
    }

    #[test]
    fn degenerate_declarations_are_refused_rather_than_producing_an_infinity() {
        // Coincident taps: no baseline to divide by.
        assert!(DeclaredScale::new((Vec3::zeros(), Vec3::zeros()), 1.0).is_err());
        // Zero observed distance: s would be +inf.
        assert!(DeclaredScale::from_distance(1.0, 0.0).is_err());
        // Negative or non-finite either way.
        assert!(DeclaredScale::from_distance(-1.0, 1.0).is_err());
        assert!(DeclaredScale::from_distance(1.0, -1.0).is_err());
        assert!(DeclaredScale::from_distance(Scalar::NAN, 1.0).is_err());
        assert!(DeclaredScale::from_distance(1.0, Scalar::INFINITY).is_err());
    }

    #[test]
    fn source_estimate_matches_the_direct_computation() {
        let mut d = DeclaredScale::from_distance(0.75, 0.3).unwrap();
        let direct = d.scale_estimate();
        let via_trait = d.estimate(&StateWindow::with_default_capacity()).unwrap();
        assert_eq!(direct, via_trait);
        assert_eq!(d.kind(), ScaleKind::Declared);
    }

    proptest::proptest! {
        /// Round-trip: whatever the declaration, applying the recovered scale
        /// to the observed distance must reproduce the declared metres.
        #[test]
        fn scale_times_observed_distance_recovers_the_declared_metres(
            metres in 0.01f64..100.0,
            observed in 1e-3f64..1e3,
        ) {
            let d = DeclaredScale::from_distance(metres, observed).unwrap();
            let e = d.scale_estimate();
            proptest::prop_assert!((e.value * observed - metres).abs() <= 1e-9 * metres);
            proptest::prop_assert!(e.variance > 0.0);
            proptest::prop_assert!(e.variance.is_finite());
        }
    }
}
