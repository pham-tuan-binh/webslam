//! The honest default: no ruler, therefore no metres.
//!
//! spec.md §3 lists `ScaleSource.none()` as *"up-to-scale, honest default"* and
//! §1 as the reason the whole layer exists: *"The library never silently
//! guesses scale."* The temptation this type resists is returning `1.0` with a
//! small variance so that downstream code "just works" in metres. That is the
//! silent guess, and it is worse than no answer at all: a caller who receives
//! `1.0 ± 0.01` has been told the phone moved one metre when it may have moved
//! ten centimetres, and nothing downstream can detect the lie.

use crate::ScaleSource;
use wslam_core::{ScaleEstimate, ScaleKind, StateWindow};

/// Up-to-scale. Positions are in arbitrary units and the variance is infinite.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoneScale;

impl NoneScale {
    /// Construct. There is nothing to configure — that is the point.
    #[must_use]
    pub fn new() -> Self {
        NoneScale
    }
}

impl ScaleSource for NoneScale {
    fn kind(&self) -> ScaleKind {
        ScaleKind::None
    }

    fn estimate(&mut self, _window: &StateWindow) -> Option<ScaleEstimate> {
        // Some(unscaled), not None: this source *has* an answer, and the answer
        // is "the units are arbitrary and I have no idea what they mean". A
        // `None` here would be indistinguishable from a source that is still
        // waiting for data.
        Some(ScaleEstimate::unscaled())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wslam_core::{Se3, Timestamp, Vec3, WindowSample};

    fn window_with_motion() -> StateWindow {
        let mut w = StateWindow::with_default_capacity();
        for i in 0..30 {
            w.push_pose(WindowSample {
                timestamp: Timestamp::from_seconds(i as f64 / 30.0),
                pose: Se3::from_translation(Vec3::new(i as f64 * 0.01, 0.0, 0.0)),
                landmark_count: 120,
            });
        }
        w
    }

    /// The design invariant, asserted by name. If this test ever has to be
    /// relaxed, the change under review is a silent guess (spec.md §1).
    #[test]
    fn none_scale_reports_infinite_variance_never_a_finite_guess() {
        let mut s = NoneScale::new();
        for window in [StateWindow::with_default_capacity(), window_with_motion()] {
            let e = s.estimate(&window).expect("none always answers");
            assert_eq!(e.source, ScaleKind::None);
            assert!(
                e.variance.is_infinite(),
                "a finite variance here is the silent guess spec.md §1 forbids: {e:?}"
            );
            assert!(!e.variance.is_nan());
            assert!(!e.source.is_metric());
            assert!(e.relative_stddev_percent().is_infinite());
        }
    }

    /// `value` is 1.0 only because the units are arbitrary, so the multiplier
    /// is a no-op. It must never be read as "one unit is one metre" — the
    /// infinite variance above is what carries that meaning.
    #[test]
    fn none_scale_value_is_a_no_op_multiplier() {
        let mut s = NoneScale::new();
        let e = s.estimate(&StateWindow::with_default_capacity()).unwrap();
        assert_eq!(e.value, 1.0);
    }

    /// Applying it to a pose must leave a finite covariance finite — infinite
    /// scale variance times a zero-length translation is the one place this
    /// could produce NaN and poison every consumer.
    #[test]
    fn applying_none_scale_does_not_poison_a_pose_covariance() {
        use wslam_core::{Mat6, Pose};
        let mut p = Pose::identity_at(Timestamp::ZERO);
        p.transform = Se3::from_translation(Vec3::new(3.0, -1.0, 2.0));
        p.covariance = Mat6::identity() * 0.02;

        let mut s = NoneScale::new();
        let scaled = p.with_scale(s.estimate(&window_with_motion()).unwrap());
        assert!(scaled.covariance.iter().all(|v| v.is_finite()));
        assert_eq!(scaled.position(), p.position(), "no-op multiplier");
    }

    #[test]
    fn kind_is_stable_and_reset_is_a_no_op() {
        let mut s = NoneScale::new();
        assert_eq!(s.kind(), ScaleKind::None);
        s.reset();
        assert_eq!(s.kind(), ScaleKind::None);
        assert!(s
            .estimate(&StateWindow::with_default_capacity())
            .unwrap()
            .variance
            .is_infinite());
    }
}
