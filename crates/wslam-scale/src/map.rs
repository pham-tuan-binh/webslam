//! Scale inherited from a previously anchored map.
//!
//! spec.md §2 is why mapping is in scope at all: *"Anchor scale once by any of
//! the above, persist the map, and every subsequent session recovers metric by
//! relocalizing. This converts scale from a hard per-session estimation problem
//! into a one-time one."*
//!
//! The rule that makes it safe is spec.md §4 L5: `map` *"inherits the variance
//! of whatever anchored it, plus relocalization error — it must not report
//! itself as more certain than its origin."* Relocalization is a pose solve
//! against a stored keyframe; it lands the session near the anchored frame but
//! not on it, and that residual is real uncertainty that has to be added, never
//! averaged away.
//!
//! This type needs nothing from `wslam-map`: an anchor is a
//! [`ScaleEstimate`] and relocalization error is one number, so L5 does not
//! take a dependency on L4 to express it.

use crate::ScaleSource;
use wslam_core::{Scalar, ScaleEstimate, ScaleKind, StateWindow};

/// Scale carried forward from a map's anchor, inflated by relocalization error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapScale {
    anchor: ScaleEstimate,
    reloc_variance: Scalar,
}

impl MapScale {
    /// Wrap a map's stored anchor together with the variance the
    /// relocalization solve contributes to the scale multiplier.
    ///
    /// A negative `reloc_variance` is clamped to zero. It is nonsense as an
    /// input, and the one thing this type must never do is let a bad number
    /// *reduce* the inherited uncertainty.
    #[must_use]
    pub fn new(anchor: ScaleEstimate, reloc_variance: Scalar) -> Self {
        let reloc_variance = if reloc_variance.is_finite() {
            reloc_variance.max(0.0)
        } else {
            // A non-finite relocalization variance means "the relocalization
            // told us nothing", which is exactly infinite uncertainty.
            Scalar::INFINITY
        };
        MapScale {
            anchor,
            reloc_variance,
        }
    }

    /// The anchor this map was built with — the *origin* whose certainty must
    /// bound ours. Kept reachable so the debug surface can show which ruler
    /// really produced the metres.
    #[must_use]
    pub fn anchor(&self) -> ScaleEstimate {
        self.anchor
    }

    /// Variance contributed by relocalizing into the map.
    #[must_use]
    pub fn relocalization_variance(&self) -> Scalar {
        self.reloc_variance
    }

    /// Update the relocalization variance — it depends on the quality of the
    /// most recent relocalization, not on the map.
    pub fn set_relocalization_variance(&mut self, variance: Scalar) {
        *self = MapScale::new(self.anchor, variance);
    }

    /// The inherited estimate *keeping the origin's provenance tag*.
    ///
    /// This is literally `anchor.inflated_by(reloc_variance)`. Use it when you
    /// want to know which ruler originally anchored the map; use
    /// [`ScaleSource::estimate`] when you want to know what produced this
    /// session's metres.
    #[must_use]
    pub fn inherited(&self) -> ScaleEstimate {
        self.anchor.inflated_by(self.reloc_variance)
    }
}

impl ScaleSource for MapScale {
    fn kind(&self) -> ScaleKind {
        ScaleKind::Map
    }

    fn estimate(&mut self, _window: &StateWindow) -> Option<ScaleEstimate> {
        let inherited = self.inherited();
        if !inherited.source.is_metric() {
            // A map that was never anchored is not a ruler. Re-tagging an
            // infinite-variance estimate as `Map` would make `is_metric()`
            // report true while the variance says we know nothing — the exact
            // contradiction spec.md §1 forbids.
            return Some(ScaleEstimate::unscaled());
        }
        // Re-tagged as `Map` rather than as the anchor's kind: spec.md §3 rule
        // 3 says the API must report "which ruler guessed it", and this
        // session's ruler is the map. The origin stays reachable through
        // `anchor()` and `inherited()`.
        Some(ScaleEstimate::metric(
            ScaleKind::Map,
            inherited.value,
            inherited.variance,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn empty_window() -> StateWindow {
        StateWindow::with_default_capacity()
    }

    /// spec.md §4 L5, quoted verbatim in the name because it is the rule this
    /// type exists to enforce.
    #[test]
    fn map_scale_must_not_report_itself_as_more_certain_than_its_origin() {
        let window = empty_window();
        let anchors = [
            ScaleEstimate::metric(ScaleKind::Fiducial, 1.0, 1e-6),
            ScaleEstimate::metric(ScaleKind::Fiducial, 0.017, 4.9e-9),
            ScaleEstimate::metric(ScaleKind::Declared, 2.75, 1e-4),
            ScaleEstimate::metric(ScaleKind::Declared, 1e3, 25.0),
            ScaleEstimate::metric(ScaleKind::Inertial, 0.5, 2.5e-5),
            ScaleEstimate::metric(ScaleKind::Learned, 1.4, 3.9e-3),
            ScaleEstimate::metric(ScaleKind::Map, 0.9, 1.0),
        ];
        let reloc_variances = [1e-12, 1e-9, 1e-6, 1e-4, 1e-2, 0.5, 10.0];

        for anchor in anchors {
            for v in reloc_variances {
                let mut m = MapScale::new(anchor, v);
                let out = m.estimate(&window).expect("an anchored map always answers");
                assert!(
                    out.variance > anchor.variance,
                    "inheriting {anchor:?} with reloc variance {v} must be strictly less \
                     certain, got {out:?}"
                );
                assert_relative_eq!(
                    out.variance,
                    anchor.variance + v,
                    epsilon = 1e-18 + v * 1e-12
                );
                // Inflation must never move the estimate itself.
                assert_eq!(out.value, anchor.value);
                assert_eq!(out.source, ScaleKind::Map);
                // And the relative precision degrades, which is what a
                // consumer actually reads.
                assert!(out.relative_stddev_percent() > anchor.relative_stddev_percent());
            }
        }
    }

    /// The contract's literal requirement: the arithmetic is
    /// `anchor.inflated_by(reloc_variance)`.
    #[test]
    fn inherited_is_exactly_inflated_by() {
        let anchor = ScaleEstimate::metric(ScaleKind::Fiducial, 1.25, 2e-5);
        let m = MapScale::new(anchor, 7e-6);
        assert_eq!(m.inherited(), anchor.inflated_by(7e-6));
        // ... and it keeps the *origin's* tag, unlike `estimate()`.
        assert_eq!(m.inherited().source, ScaleKind::Fiducial);
    }

    /// A zero relocalization variance is a claim of a perfect relocalization.
    /// We do not reject it, but it must still never shrink the anchor.
    #[test]
    fn a_zero_or_negative_relocalization_variance_never_shrinks_the_anchor() {
        let anchor = ScaleEstimate::metric(ScaleKind::Declared, 3.0, 1e-4);
        for v in [0.0, -1e-6, -1.0] {
            let mut m = MapScale::new(anchor, v);
            let out = m.estimate(&empty_window()).unwrap();
            assert!(
                out.variance >= anchor.variance,
                "reloc variance {v} shrank the anchor"
            );
            assert_relative_eq!(out.variance, anchor.variance, epsilon = 1e-18);
        }
    }

    /// A map nobody ever anchored is not a ruler. Re-tagging it as `Map` would
    /// make `is_metric()` true while the variance says we know nothing.
    #[test]
    fn an_unanchored_map_stays_unscaled() {
        let mut m = MapScale::new(ScaleEstimate::unscaled(), 1e-6);
        let out = m.estimate(&empty_window()).unwrap();
        assert_eq!(out.source, ScaleKind::None);
        assert!(out.variance.is_infinite());
        assert!(!out.source.is_metric());
    }

    /// A relocalization that told us nothing must poison the estimate, not be
    /// quietly ignored.
    #[test]
    fn an_infinite_relocalization_variance_makes_the_result_uninformative() {
        let anchor = ScaleEstimate::metric(ScaleKind::Fiducial, 1.0, 1e-6);
        let mut m = MapScale::new(anchor, Scalar::INFINITY);
        let out = m.estimate(&empty_window()).unwrap();
        assert!(out.variance.is_infinite());
        assert!(out.relative_stddev_percent().is_infinite());
    }

    #[test]
    fn a_worse_relocalization_is_reported_as_worse() {
        let anchor = ScaleEstimate::metric(ScaleKind::Fiducial, 1.0, 1e-6);
        let mut good = MapScale::new(anchor, 1e-6);
        let mut poor = MapScale::new(anchor, 1e-3);
        let (g, p) = (
            good.estimate(&empty_window()).unwrap(),
            poor.estimate(&empty_window()).unwrap(),
        );
        assert!(p.variance > g.variance);

        // And the same object updated in place tracks the change.
        good.set_relocalization_variance(1e-3);
        assert_eq!(good.estimate(&empty_window()).unwrap(), p);
    }

    /// Chaining maps must accumulate, not reset: a map anchored from a map
    /// anchored from a fiducial is less certain than either.
    #[test]
    fn chained_inheritance_accumulates() {
        let fiducial = ScaleEstimate::metric(ScaleKind::Fiducial, 1.0, 1e-6);
        let mut first = MapScale::new(fiducial, 2e-6);
        let once = first.estimate(&empty_window()).unwrap();
        let mut second = MapScale::new(once, 3e-6);
        let twice = second.estimate(&empty_window()).unwrap();

        assert!(twice.variance > once.variance);
        assert!(once.variance > fiducial.variance);
        assert_relative_eq!(twice.variance, 1e-6 + 2e-6 + 3e-6, epsilon = 1e-18);
    }

    #[test]
    fn kind_is_map_and_the_anchor_stays_reachable() {
        let anchor = ScaleEstimate::metric(ScaleKind::Inertial, 1.1, 1e-4);
        let mut m = MapScale::new(anchor, 1e-5);
        assert_eq!(m.kind(), ScaleKind::Map);
        assert_eq!(m.anchor(), anchor);
        assert_relative_eq!(m.relocalization_variance(), 1e-5, epsilon = 1e-18);
        m.reset();
        assert_eq!(m.anchor(), anchor, "reset must not discard the anchor");
    }
}
