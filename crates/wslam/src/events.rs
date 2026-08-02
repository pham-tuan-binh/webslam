//! Session events.
//!
//! The push-side counterpart to `slam.currentPose()`. spec.md §3 names three
//! (`onState`, `onRelocalize`, `onLoopClosure`); `ScaleAcquired` is added
//! because a session that silently becomes metric is exactly the silent
//! behaviour §1 exists to prevent.

use wslam_core::{ScaleEstimate, Timestamp, TrackingState};

/// Something worth telling the caller about.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SlamEvent {
    /// Tracking state changed. Never fires for a self-transition.
    State {
        /// Previous state.
        from: TrackingState,
        /// New state.
        to: TrackingState,
    },
    /// The session recovered into the map after loss.
    Relocalized {
        /// When, in the unified timebase.
        at: Timestamp,
        /// Which stored keyframe it recovered into.
        keyframe: u64,
        /// Correspondences that survived geometric verification.
        inliers: usize,
    },
    /// A loop-closure candidate was considered.
    ///
    /// **Fires for rejections too.** spec.md §5 makes the false-positive rate a
    /// first-class metric, and a threshold nobody can observe is a threshold
    /// nobody can tune.
    LoopClosure {
        /// Whether geometric verification accepted it.
        accepted: bool,
        /// The keyframe place recognition proposed.
        candidate: u64,
        /// Bag-of-words similarity, in `[0, 1]`.
        score: f64,
    },
    /// The tracker adopted a new, unrelated coordinate frame.
    ///
    /// Fires when a bootstrap happens without a relocalization vouching for it.
    /// Poses before and after have **no** common origin or scale, so a consumer
    /// holding world-anchored content must discard it — the same contract as
    /// WebXR's `XRReferenceSpace` `reset` event.
    OriginReset {
        /// When, in the unified timebase.
        at: Timestamp,
        /// The new epoch.
        epoch: u32,
    },
    /// The session became metric for the first time.
    ScaleAcquired {
        /// The estimate, including which ruler produced it.
        estimate: ScaleEstimate,
    },
}

impl SlamEvent {
    /// Stable discriminant, for the wasm boundary and logs.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            SlamEvent::State { .. } => "state",
            SlamEvent::Relocalized { .. } => "relocalized",
            SlamEvent::LoopClosure { .. } => "loop",
            SlamEvent::OriginReset { .. } => "origin-reset",
            SlamEvent::ScaleAcquired { .. } => "scale",
        }
    }

    /// JSON, for the wasm boundary. Hand-written because the shapes are tiny,
    /// fixed, and not worth a serde dependency in the wasm binary.
    #[must_use]
    pub fn to_json(&self) -> String {
        match self {
            SlamEvent::State { from, to } => format!(
                r#"{{"type":"state","from":"{}","to":"{}","reason":{}}}"#,
                from.as_str(),
                to.as_str(),
                match to.limited_reason() {
                    Some(r) => format!(r#""{}""#, r.as_str()),
                    None => "null".to_string(),
                }
            ),
            SlamEvent::Relocalized {
                at,
                keyframe,
                inliers,
            } => format!(
                r#"{{"type":"relocalized","atTimestamp":{:.6},"keyframe":{keyframe},"inliers":{inliers}}}"#,
                at.millis()
            ),
            SlamEvent::LoopClosure {
                accepted,
                candidate,
                score,
            } => format!(
                r#"{{"type":"loop","accepted":{accepted},"candidate":{candidate},"score":{score:.6}}}"#
            ),
            SlamEvent::OriginReset { at, epoch } => format!(
                r#"{{"type":"origin-reset","atTimestamp":{:.6},"epoch":{epoch}}}"#,
                at.millis()
            ),
            SlamEvent::ScaleAcquired { estimate } => format!(
                // JSON has no Infinity, and `ScaleKind::None` carries infinite
                // variance by design — emitting it literally would make
                // `JSON.parse` throw on the other side. `null` means "unbounded",
                // which the TypeScript shim maps back to `Infinity`.
                r#"{{"type":"scale","source":"{}","value":{:.9},"variance":{}}}"#,
                estimate.source.as_str(),
                estimate.value,
                if estimate.variance.is_finite() {
                    format!("{:.9e}", estimate.variance)
                } else {
                    "null".to_string()
                }
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wslam_core::{LimitedReason, ScaleKind};

    #[test]
    fn every_variant_has_a_distinct_kind() {
        let events = [
            SlamEvent::State {
                from: TrackingState::Initializing,
                to: TrackingState::Tracking,
            },
            SlamEvent::Relocalized {
                at: Timestamp::ZERO,
                keyframe: 1,
                inliers: 40,
            },
            SlamEvent::LoopClosure {
                accepted: false,
                candidate: 2,
                score: 0.4,
            },
            SlamEvent::ScaleAcquired {
                estimate: ScaleEstimate::unscaled(),
            },
        ];
        let mut kinds: Vec<&str> = events.iter().map(SlamEvent::kind).collect();
        kinds.sort_unstable();
        kinds.dedup();
        assert_eq!(kinds.len(), 4);
    }

    #[test]
    fn limited_state_carries_its_reason_into_json() {
        let e = SlamEvent::State {
            from: TrackingState::Tracking,
            to: TrackingState::Limited(LimitedReason::LowLight),
        };
        let json = e.to_json();
        assert!(json.contains(r#""to":"limited""#), "{json}");
        assert!(json.contains(r#""reason":"low-light""#), "{json}");
    }

    #[test]
    fn an_unlimited_state_reports_a_null_reason_not_an_empty_string() {
        let e = SlamEvent::State {
            from: TrackingState::Initializing,
            to: TrackingState::Tracking,
        };
        assert!(e.to_json().contains(r#""reason":null"#));
    }

    #[test]
    fn rejected_loops_serialise_as_rejections() {
        // The consumer must be able to distinguish them; that is the whole
        // reason rejections are reported at all.
        let e = SlamEvent::LoopClosure {
            accepted: false,
            candidate: 7,
            score: 0.44,
        };
        let json = e.to_json();
        assert!(json.contains(r#""accepted":false"#), "{json}");
        assert!(json.contains(r#""candidate":7"#), "{json}");
    }

    #[test]
    fn scale_json_carries_provenance_and_variance() {
        let e = SlamEvent::ScaleAcquired {
            estimate: ScaleEstimate::metric(ScaleKind::Fiducial, 1.25, 1e-5),
        };
        let json = e.to_json();
        assert!(json.contains(r#""source":"fiducial""#), "{json}");
        assert!(json.contains("1.25"), "{json}");
    }

    #[test]
    fn infinite_variance_does_not_produce_invalid_json() {
        // `ScaleKind::None` has infinite variance by design, and `Infinity` is
        // not valid JSON — JSON.parse would throw on the other side.
        let e = SlamEvent::ScaleAcquired {
            estimate: ScaleEstimate::unscaled(),
        };
        let json = e.to_json();
        assert!(
            !json.contains("inf") && !json.contains("NaN"),
            "non-finite leaked into JSON: {json}"
        );
    }
}
