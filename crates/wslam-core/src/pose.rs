//! The output type. spec.md §3: *"Covariance and scale provenance travel with
//! every pose. They are not queried separately, because separate queries get
//! skipped."*

use crate::math::{Mat6, Scalar, Se3, Vec3};
use crate::time::Timestamp;

/// Which ruler produced metric scale (spec.md §2).
///
/// `None` is not an error state; it is the honest default. A pose with
/// `ScaleKind::None` is up to scale and its `position` is in arbitrary units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaleKind {
    /// Up to scale. Positions are in arbitrary units.
    None,
    /// The user tapped a known distance. Exact, free, needs one interaction.
    Declared,
    /// A fiducial marker of known physical size is visible.
    Fiducial,
    /// A learned monocular depth prior. Several percent, domain-correlated.
    Learned,
    /// Relocalized into a previously anchored map. Inherits that anchor's
    /// variance plus relocalization error, and must never claim to be more
    /// certain than its origin.
    Map,
    /// Double-integrated acceleration. ~1% given excitation; requires tier 3.
    Inertial,
}

impl ScaleKind {
    /// Stable string form, used across the wasm boundary and in the map header.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            ScaleKind::None => "none",
            ScaleKind::Declared => "declared",
            ScaleKind::Fiducial => "fiducial",
            ScaleKind::Learned => "learned",
            ScaleKind::Map => "map",
            ScaleKind::Inertial => "inertial",
        }
    }

    /// Parse the stable string form.
    ///
    /// Deliberately an inherent method returning `Option` rather than a
    /// `FromStr` impl: there is exactly one way to fail — the string is not one
    /// of six known tags — and `Result<_, SomeEmptyError>` would add a type
    /// carrying no information for every caller to unwrap.
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "none" => ScaleKind::None,
            "declared" => ScaleKind::Declared,
            "fiducial" => ScaleKind::Fiducial,
            "learned" => ScaleKind::Learned,
            "map" => ScaleKind::Map,
            "inertial" => ScaleKind::Inertial,
            _ => return None,
        })
    }

    /// Whether positions carrying this kind are in metres.
    #[must_use]
    pub fn is_metric(self) -> bool {
        !matches!(self, ScaleKind::None)
    }

    /// Compact numeric tag for the wasm ABI and the map header.
    #[must_use]
    pub fn tag(self) -> u8 {
        match self {
            ScaleKind::None => 0,
            ScaleKind::Declared => 1,
            ScaleKind::Fiducial => 2,
            ScaleKind::Learned => 3,
            ScaleKind::Map => 4,
            ScaleKind::Inertial => 5,
        }
    }

    /// Inverse of [`ScaleKind::tag`].
    #[must_use]
    pub fn from_tag(tag: u8) -> Option<Self> {
        Some(match tag {
            0 => ScaleKind::None,
            1 => ScaleKind::Declared,
            2 => ScaleKind::Fiducial,
            3 => ScaleKind::Learned,
            4 => ScaleKind::Map,
            5 => ScaleKind::Inertial,
            _ => return None,
        })
    }
}

/// A scale estimate with its provenance and uncertainty.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleEstimate {
    /// Which ruler produced it.
    pub source: ScaleKind,
    /// Multiplier taking up-to-scale units to metres. Always 1.0 for
    /// [`ScaleKind::None`], where the units are arbitrary by definition.
    pub value: Scalar,
    /// Variance of `value`. Infinite for [`ScaleKind::None`] — "we do not know"
    /// rather than "we are certain it is 1".
    pub variance: Scalar,
}

impl Default for ScaleEstimate {
    fn default() -> Self {
        Self::unscaled()
    }
}

impl ScaleEstimate {
    /// The honest default: up to scale, unbounded uncertainty.
    #[must_use]
    pub fn unscaled() -> Self {
        ScaleEstimate {
            source: ScaleKind::None,
            value: 1.0,
            variance: Scalar::INFINITY,
        }
    }

    /// A metric estimate.
    #[must_use]
    pub fn metric(source: ScaleKind, value: Scalar, variance: Scalar) -> Self {
        ScaleEstimate {
            source,
            value,
            variance,
        }
    }

    /// Relative standard deviation as a percentage — the unit spec.md §6 L5
    /// reports scale error in ("2 s and 10 s numbers against their 5% / 1%").
    #[must_use]
    pub fn relative_stddev_percent(&self) -> Scalar {
        if !self.variance.is_finite() || self.value.abs() < 1e-12 {
            Scalar::INFINITY
        } else {
            100.0 * self.variance.sqrt() / self.value.abs()
        }
    }

    /// Compose with a downstream uncertainty, e.g. a map's inherited anchor
    /// variance plus relocalization error. Variances add; the value is taken
    /// from `self`.
    ///
    /// This is the mechanism behind spec.md §4 L5: `map` *"must not report
    /// itself as more certain than its origin"*.
    #[must_use]
    pub fn inflated_by(&self, extra_variance: Scalar) -> Self {
        ScaleEstimate {
            source: self.source,
            value: self.value,
            variance: self.variance + extra_variance,
        }
    }
}

/// Why tracking is degraded but not lost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LimitedReason {
    /// Motion blur or inter-frame displacement beyond the tracker's envelope.
    ExcessiveMotion,
    /// Too few corners survived — blank wall, extreme close-up.
    InsufficientFeatures,
    /// Mean intensity below the usable threshold.
    LowLight,
}

impl LimitedReason {
    /// Stable string form matching the TypeScript union.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            LimitedReason::ExcessiveMotion => "excessive-motion",
            LimitedReason::InsufficientFeatures => "insufficient-features",
            LimitedReason::LowLight => "low-light",
        }
    }
}

/// Tracking state machine, mirroring the TypeScript union in spec.md §3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrackingState {
    /// Bootstrapping: estimating intrinsics, waiting for parallax.
    Initializing,
    /// Nominal.
    Tracking,
    /// Still producing pose, but degraded for a stated reason.
    Limited(LimitedReason),
    /// Lost, and actively querying the place-recognition database.
    Relocalizing,
    /// Lost with no map to recover into.
    Lost,
}

impl TrackingState {
    /// Whether a pose emitted in this state is usable for rendering.
    #[must_use]
    pub fn has_pose(self) -> bool {
        matches!(self, TrackingState::Tracking | TrackingState::Limited(_))
    }

    /// Stable discriminant string. `Limited` reports `"limited"`; the reason
    /// travels separately so the ABI stays flat.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            TrackingState::Initializing => "initializing",
            TrackingState::Tracking => "tracking",
            TrackingState::Limited(_) => "limited",
            TrackingState::Relocalizing => "relocalizing",
            TrackingState::Lost => "lost",
        }
    }

    /// The reason, when limited.
    #[must_use]
    pub fn limited_reason(self) -> Option<LimitedReason> {
        match self {
            TrackingState::Limited(r) => Some(r),
            _ => None,
        }
    }
}

/// A 6-DoF pose with everything a consumer needs to decide how much to trust it.
///
/// spec.md §4 L6: *"Emitting a naked transform forces every consumer to guess
/// how much to trust it. We won't."*
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    /// Capture time of the frame this pose was computed from, in the unified
    /// timebase.
    pub timestamp: Timestamp,
    /// `T_world_camera`. Metric when `scale.source != ScaleKind::None`.
    pub transform: Se3,
    /// 6x6 covariance in `[translation, rotation]` block order, in the
    /// **body frame** (right-perturbation convention, matching [`Se3::plus`]).
    pub covariance: Mat6,
    /// Provenance and uncertainty of metric scale.
    pub scale: ScaleEstimate,
    /// Tracking state at emission.
    pub state: TrackingState,
    /// Milliseconds since the current tracking session initialised. Resets on
    /// re-initialisation but *not* on relocalization into an existing map.
    pub init_age_ms: f64,
    /// Which coordinate frame this pose is expressed in.
    ///
    /// Monocular tracking recovers position only up to scale, and a fresh
    /// bootstrap picks a **new, unrelated** scale and origin. Two poses with
    /// different epochs therefore live in different worlds and must never be
    /// differenced, concatenated, or fitted with one similarity transform.
    ///
    /// The epoch increments when the tracker adopts an anchor that is not tied
    /// to the previous one — a re-bootstrap after loss without a verified
    /// relocalization. It does **not** increment on relocalization, which is
    /// precisely the operation that re-establishes the old frame.
    ///
    /// Ignoring this is not a small error. Fitting one Sim(3) across a spliced
    /// trajectory on EuRoC reported 3.1 m ATE where the individual segments sit
    /// near 0.06 m: the number was measuring the seams, not the tracking.
    pub frame_epoch: u32,
}

impl Default for Pose {
    fn default() -> Self {
        Pose::identity_at(Timestamp::ZERO)
    }
}

impl Pose {
    /// An identity pose with infinite covariance, at a given time.
    #[must_use]
    pub fn identity_at(timestamp: Timestamp) -> Self {
        Pose {
            timestamp,
            transform: Se3::identity(),
            covariance: Mat6::identity() * Scalar::INFINITY,
            scale: ScaleEstimate::unscaled(),
            state: TrackingState::Initializing,
            init_age_ms: 0.0,
            frame_epoch: 0,
        }
    }

    /// Camera position in world coordinates. Metres iff `scale.source` is
    /// metric.
    #[inline]
    #[must_use]
    pub fn position(&self) -> Vec3 {
        self.transform.translation()
    }

    /// Camera orientation.
    #[inline]
    #[must_use]
    pub fn rotation(&self) -> crate::math::So3 {
        self.transform.rotation()
    }

    /// Column-major 4x4 `f32`, renderer-ready.
    #[inline]
    #[must_use]
    pub fn matrix(&self) -> [f32; 16] {
        self.transform.to_matrix_f32()
    }

    /// Translation-block standard deviations, in the same units as `position`.
    #[must_use]
    pub fn position_stddev(&self) -> Vec3 {
        Vec3::new(
            self.covariance[(0, 0)].max(0.0).sqrt(),
            self.covariance[(1, 1)].max(0.0).sqrt(),
            self.covariance[(2, 2)].max(0.0).sqrt(),
        )
    }

    /// Rotation-block standard deviations, in radians.
    #[must_use]
    pub fn rotation_stddev(&self) -> Vec3 {
        Vec3::new(
            self.covariance[(3, 3)].max(0.0).sqrt(),
            self.covariance[(4, 4)].max(0.0).sqrt(),
            self.covariance[(5, 5)].max(0.0).sqrt(),
        )
    }

    /// Apply a metric scale to an up-to-scale pose.
    ///
    /// Scales the translation, the translation covariance block (by `s^2`), and
    /// records the provenance. Rotation and its covariance are untouched
    /// because rotation is scale-invariant.
    ///
    /// The scale's *own* variance is propagated into the translation block:
    /// `Var(s*t) = s^2 Var(t) + t^2 Var(s)` under independence. Skipping that
    /// second term is the standard way covariance ends up overconfident, and
    /// spec.md §6 L6 calls overconfidence *"worse than no covariance at all"*.
    #[must_use]
    pub fn with_scale(&self, scale: ScaleEstimate) -> Self {
        let s = scale.value;
        let t = self.transform.translation();
        let mut cov = self.covariance;

        // s^2 * translation block, and the cross terms with rotation.
        for r in 0..6 {
            for c in 0..6 {
                let factor = match (r < 3, c < 3) {
                    (true, true) => s * s,
                    (true, false) | (false, true) => s,
                    (false, false) => 1.0,
                };
                cov[(r, c)] *= factor;
            }
        }
        if scale.variance.is_finite() {
            for r in 0..3 {
                for c in 0..3 {
                    cov[(r, c)] += t[r] * t[c] * scale.variance;
                }
            }
        }

        Pose {
            timestamp: self.timestamp,
            transform: self.transform.scaled(s),
            covariance: cov,
            scale,
            state: self.state,
            init_age_ms: self.init_age_ms,
            frame_epoch: self.frame_epoch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::So3;
    use approx::assert_relative_eq;

    #[test]
    fn unscaled_variance_is_infinite_not_zero() {
        let s = ScaleEstimate::unscaled();
        assert_eq!(s.source, ScaleKind::None);
        assert!(s.variance.is_infinite());
        assert!(!s.source.is_metric());
    }

    #[test]
    fn scale_kind_string_and_tag_roundtrip() {
        for k in [
            ScaleKind::None,
            ScaleKind::Declared,
            ScaleKind::Fiducial,
            ScaleKind::Learned,
            ScaleKind::Map,
            ScaleKind::Inertial,
        ] {
            assert_eq!(ScaleKind::from_str(k.as_str()), Some(k));
            assert_eq!(ScaleKind::from_tag(k.tag()), Some(k));
        }
        assert_eq!(ScaleKind::from_str("bogus"), None);
        assert_eq!(ScaleKind::from_tag(200), None);
    }

    #[test]
    fn inflating_never_reduces_uncertainty() {
        let anchor = ScaleEstimate::metric(ScaleKind::Fiducial, 1.0, 1e-4);
        let inherited = anchor.inflated_by(4e-4);
        assert!(inherited.variance > anchor.variance);
        assert_relative_eq!(inherited.variance, 5e-4, epsilon = 1e-15);
    }

    #[test]
    fn with_scale_scales_translation_and_its_covariance() {
        let mut p = Pose::identity_at(Timestamp::ZERO);
        p.transform = crate::math::Se3::new(So3::identity(), Vec3::new(1.0, 0.0, 0.0));
        p.covariance = Mat6::identity() * 0.01;

        // Exact scale (zero variance) isolates the s^2 term.
        let scaled = p.with_scale(ScaleEstimate::metric(ScaleKind::Declared, 3.0, 0.0));
        assert_relative_eq!(scaled.position().x, 3.0, epsilon = 1e-12);
        assert_relative_eq!(scaled.covariance[(0, 0)], 0.09, epsilon = 1e-12);
        // Rotation block untouched.
        assert_relative_eq!(scaled.covariance[(3, 3)], 0.01, epsilon = 1e-12);
    }

    #[test]
    fn scale_variance_propagates_into_position_covariance() {
        let mut p = Pose::identity_at(Timestamp::ZERO);
        p.transform = crate::math::Se3::new(So3::identity(), Vec3::new(2.0, 0.0, 0.0));
        p.covariance = Mat6::zeros();

        // Pure scale uncertainty: Var(s*t) = t^2 * Var(s) = 4 * 0.01.
        let scaled = p.with_scale(ScaleEstimate::metric(ScaleKind::Inertial, 1.0, 0.01));
        assert_relative_eq!(scaled.covariance[(0, 0)], 0.04, epsilon = 1e-12);
        assert_relative_eq!(scaled.covariance[(1, 1)], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn unscaled_scale_does_not_add_infinite_covariance() {
        // ScaleKind::None has infinite variance, but applying it must not turn
        // a finite up-to-scale covariance into NaN.
        let mut p = Pose::identity_at(Timestamp::ZERO);
        p.covariance = Mat6::identity() * 0.01;
        let out = p.with_scale(ScaleEstimate::unscaled());
        assert!(out.covariance.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn tracking_state_pose_availability() {
        assert!(TrackingState::Tracking.has_pose());
        assert!(TrackingState::Limited(LimitedReason::LowLight).has_pose());
        assert!(!TrackingState::Lost.has_pose());
        assert!(!TrackingState::Relocalizing.has_pose());
        assert!(!TrackingState::Initializing.has_pose());
        assert_eq!(
            TrackingState::Limited(LimitedReason::LowLight).limited_reason(),
            Some(LimitedReason::LowLight)
        );
    }

    #[test]
    fn relative_stddev_percent_matches_campos_units() {
        // 1% scale error means stddev/value = 0.01, i.e. variance = 1e-4.
        let s = ScaleEstimate::metric(ScaleKind::Inertial, 1.0, 1e-4);
        assert_relative_eq!(s.relative_stddev_percent(), 1.0, epsilon = 1e-12);
        assert!(ScaleEstimate::unscaled()
            .relative_stddev_percent()
            .is_infinite());
    }
}
