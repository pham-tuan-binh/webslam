//! # wslam-scale — L5, the only opinionated layer
//!
//! spec.md §1: *"The library never silently guesses scale. Callers choose an
//! anchor and accept its tradeoffs."* Everything in this crate exists to make
//! that sentence enforceable rather than aspirational.
//!
//! Monocular metric scale is unobservable — a theorem, not a gap (spec.md §2).
//! A scene twice as large at twice the distance produces pixel-identical
//! images, so scale always comes from a *ruler*, and there are only a handful:
//!
//! | Source | Ruler | Cost |
//! |---|---|---|
//! | [`NoneScale`] | none | up to scale; the honest default |
//! | [`DeclaredScale`] | the user taps a known distance | one interaction |
//! | [`FiducialScale`] | a printed tag of known size | the tag must be visible |
//! | [`MapScale`] | a previously anchored map | must relocalize |
//! | [`LearnedScale`] | a monocular depth prior | model download, GPU |
//! | [`InertialScale`] | double-integrated acceleration | needs excitation, tier 3 |
//!
//! Three design rules are structural here, not stylistic:
//!
//! 1. **No source may return a confident wrong answer.** [`NoneScale`] reports
//!    infinite variance rather than `1.0 ± small`; [`InertialScale`] returns
//!    `None` under a static hold or pure rotation rather than a number
//!    (spec.md §6 Tier 3: the static hold *"should be detected rather than
//!    silently wrong"*).
//! 2. **Uncertainty only ever grows on the way downstream.** [`MapScale`]
//!    inherits its anchor's variance and adds relocalization error, because
//!    spec.md §4 L5 says it *"must not report itself as more certain than its
//!    origin"*.
//! 3. **No wall clock and no unseeded RNG.** Time enters through
//!    [`wslam_core::Timestamp`]; the only randomness in this crate is the
//!    seeded noise in [`fiducial::render`], which is test scaffolding and not
//!    on any estimation path.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use wslam_core::{ScaleEstimate, ScaleKind, StateWindow};

pub mod declared;
pub mod fiducial;
pub mod inertial;
#[cfg(feature = "learned-scale")]
pub mod learned;
pub mod map;
pub mod none;

pub use declared::{DeclaredScale, DEFAULT_TAP_RELATIVE_STDDEV};
pub use fiducial::{FiducialScale, TagFamily};
pub use inertial::{
    solve_inertial_scale, InertialConfig, InertialRejection, InertialScale, InertialSolution,
};
#[cfg(feature = "learned-scale")]
pub use learned::{
    fit_scale_and_shift, DepthModel, DepthSample, InverseDepth, LearnedConfig, LearnedRejection,
    LearnedScale, LearnedSolution, StructurePoint, SyntheticDepthModel,
};
pub use map::MapScale;
pub use none::NoneScale;

/// A ruler for metric scale.
///
/// spec.md §4 L5 gives the interface as
/// `estimate(window: StateWindow) -> { scale, variance } | null`, and the
/// `null` is the important half: a source that cannot answer must say so.
/// Returning a plausible number under a degenerate motion is the failure mode
/// this whole layer exists to prevent.
///
/// `Send` because L4 runs on a separate thread and the orchestrator may move a
/// source across the frontend/backend split (spec.md §4).
pub trait ScaleSource: Send {
    /// Which ruler this is. Stable, and travels with every emitted pose.
    fn kind(&self) -> ScaleKind;

    /// Best current estimate, or `None` when this source cannot answer from
    /// the data it has been given.
    ///
    /// Takes `&mut self` because several sources accumulate observations
    /// (fiducial detections, depth samples) between calls.
    fn estimate(&mut self, window: &StateWindow) -> Option<ScaleEstimate>;

    /// Drop accumulated state. Called on re-initialisation so a stale
    /// observation cannot anchor a fresh session.
    fn reset(&mut self) {}
}

/// Standard deviation of a scale estimate as a fraction of its value.
///
/// The unit spec.md §6 L5 reports in — *"report our 2 s and 10 s numbers
/// against their 5% / 1%"* — divided by 100. Infinite for an unscaled
/// estimate, which is what makes "we do not know" propagate instead of
/// quietly vanishing.
#[must_use]
pub fn relative_stddev(estimate: &ScaleEstimate) -> f64 {
    estimate.relative_stddev_percent() * 0.01
}

#[cfg(test)]
mod tests {
    use super::*;
    use wslam_core::{Se3, Timestamp, Vec3, WindowSample};

    /// Every source must be usable behind `dyn ScaleSource`; the orchestrator
    /// stores exactly one of these and cannot know its concrete type.
    #[test]
    fn sources_are_object_safe_and_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Box<dyn ScaleSource>>();

        let mut sources: Vec<Box<dyn ScaleSource>> = vec![
            Box::new(NoneScale::new()),
            Box::new(
                DeclaredScale::new((Vec3::zeros(), Vec3::new(1.0, 0.0, 0.0)), 2.0)
                    .expect("valid declaration"),
            ),
            Box::new(MapScale::new(
                ScaleEstimate::metric(ScaleKind::Fiducial, 1.5, 1e-4),
                1e-5,
            )),
            Box::new(FiducialScale::new(TagFamily::AprilTag36h11, 0.1).expect("valid size")),
        ];
        // `LearnedScale` holds a `Box<dyn DepthModel>`, so this is also the
        // check that the model seam stayed `Send` and did not quietly pin the
        // source to the frontend thread.
        #[cfg(feature = "learned-scale")]
        sources.push(Box::new(LearnedScale::new(Box::new(
            SyntheticDepthModel::new(8, 8, 1.0, 0.0, Box::new(|_u, v| 1.0 + v)),
        ))));

        let mut window = StateWindow::with_default_capacity();
        window.push_pose(WindowSample {
            timestamp: Timestamp::ZERO,
            pose: Se3::identity(),
            landmark_count: 40,
        });

        for s in &mut sources {
            let kind = s.kind();
            let _ = s.estimate(&window);
            s.reset();
            assert_eq!(s.kind(), kind, "kind must not change across a reset");
        }
    }

    #[test]
    fn relative_stddev_is_infinite_for_the_unscaled_estimate() {
        assert!(relative_stddev(&ScaleEstimate::unscaled()).is_infinite());
        // 1% scale error == variance 1e-4 at unit scale (spec.md §6 L5 units).
        let s = ScaleEstimate::metric(ScaleKind::Inertial, 1.0, 1e-4);
        approx::assert_relative_eq!(relative_stddev(&s), 0.01, epsilon = 1e-12);
    }
}
