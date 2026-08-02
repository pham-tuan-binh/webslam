//! The regression wall.
//!
//! spec.md §6 Tier 2: *"Per-sequence ATE checked into `harness/baselines/` as
//! **data**; CI fails on regression beyond tolerance."*
//!
//! Data, not code — a human can read the diff and see exactly which number
//! moved and by how much. The format is a minimal hand-written TOML subset so
//! the file stays reviewable and no parser dependency enters the harness.
//!
//! ## What a baseline guards
//!
//! Four things, because a single ATE number is easy to game:
//!
//! - **ATE** — did the trajectory get worse?
//! - **Lost fraction** — did the tracker start giving up? A tracker that drops
//!   half the sequence and nails the rest has a *better* ATE.
//! - **p99 frame time** — did it get slower at the tail? (§6 L4 asks for the
//!   tail, not the mean.)
//! - **Map memory per minute** — is the map growing unboundedly? (§9 lists this
//!   as a tab-killing risk.)

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use wslam_core::Scalar;

use crate::report::SequenceReport;

/// Fractional headroom allowed above a recorded ATE before it counts as a
/// regression.
///
/// 10% rather than 0: RANSAC is seeded and therefore deterministic on one
/// machine, but floating-point reductions are not bit-identical across
/// architectures, and a wall that fires on x86-vs-arm noise gets disabled
/// within a week. Anything above this is a real change and should be explained.
const ATE_TOLERANCE: Scalar = 0.10;

/// Absolute headroom on the lost fraction, in percentage points.
const LOST_TOLERANCE: Scalar = 0.05;

/// Fractional headroom on p99 frame time. Looser than ATE because timing on a
/// shared CI runner is genuinely noisy; it catches order-of-magnitude
/// regressions, not 5% ones.
const TIME_TOLERANCE: Scalar = 0.60;

/// Fractional headroom on map growth.
const MEMORY_TOLERANCE: Scalar = 0.25;

/// One recorded sequence result.
#[derive(Debug, Clone, PartialEq)]
pub struct Baseline {
    /// Sequence name.
    pub name: String,
    /// Recorded ATE RMSE, metres. `None` for a sequence with no ground truth.
    pub ate_rmse: Option<Scalar>,
    /// Recorded fraction of frames with no pose.
    pub lost_fraction: Scalar,
    /// Recorded p99 frame time, ms.
    pub frame_ms_p99: Scalar,
    /// Recorded map growth, MB/min.
    pub map_mb_per_min: Scalar,
}

impl Baseline {
    /// Record a fresh baseline from a replay.
    #[must_use]
    pub fn from_report(report: &SequenceReport) -> Self {
        Baseline {
            name: report.name.clone(),
            ate_rmse: report.ate.as_ref().map(|a| a.rmse),
            lost_fraction: report.lost_fraction(),
            frame_ms_p99: report.frame_ms_p99,
            map_mb_per_min: report.map_mb_per_min,
        }
    }

    /// Check a report against this baseline.
    ///
    /// Returns `None` when everything is within tolerance, otherwise a
    /// human-readable description of every metric that regressed — all of them,
    /// not just the first, because fixing one at a time is slow.
    #[must_use]
    pub fn check(&self, report: &SequenceReport) -> Option<String> {
        let mut problems = Vec::new();

        if let (Some(recorded), Some(current)) =
            (self.ate_rmse, report.ate.as_ref().map(|a| a.rmse))
        {
            let ceiling = recorded * (1.0 + ATE_TOLERANCE);
            if current > ceiling {
                problems.push(format!(
                    "ATE {current:.4} > {ceiling:.4} (baseline {recorded:.4} +{:.0}%)",
                    100.0 * ATE_TOLERANCE
                ));
            }
        } else if self.ate_rmse.is_some() && report.ate.is_none() {
            // Losing ground truth silently would turn the wall off.
            problems.push("ATE unavailable, but the baseline has one".to_string());
        }

        let lost = report.lost_fraction();
        if lost > self.lost_fraction + LOST_TOLERANCE {
            problems.push(format!(
                "lost {:.1}% > {:.1}% (baseline {:.1}%)",
                100.0 * lost,
                100.0 * (self.lost_fraction + LOST_TOLERANCE),
                100.0 * self.lost_fraction
            ));
        }

        if self.frame_ms_p99 > 0.0
            && report.frame_ms_p99 > self.frame_ms_p99 * (1.0 + TIME_TOLERANCE)
        {
            problems.push(format!(
                "p99 frame time {:.2} ms > {:.2} ms (baseline {:.2} ms)",
                report.frame_ms_p99,
                self.frame_ms_p99 * (1.0 + TIME_TOLERANCE),
                self.frame_ms_p99
            ));
        }

        if self.map_mb_per_min > 0.0
            && report.map_mb_per_min > self.map_mb_per_min * (1.0 + MEMORY_TOLERANCE)
        {
            problems.push(format!(
                "map growth {:.2} MB/min > {:.2} MB/min (baseline {:.2})",
                report.map_mb_per_min,
                self.map_mb_per_min * (1.0 + MEMORY_TOLERANCE),
                self.map_mb_per_min
            ));
        }

        (!problems.is_empty()).then(|| problems.join("; "))
    }

    fn to_toml(&self) -> String {
        let mut out = format!("[[sequence]]\nname = \"{}\"\n", self.name);
        match self.ate_rmse {
            Some(v) => out.push_str(&format!("ate_rmse = {v:.6}\n")),
            None => out.push_str("# ate_rmse: no ground truth in this sequence\n"),
        }
        out.push_str(&format!("lost_fraction = {:.6}\n", self.lost_fraction));
        out.push_str(&format!("frame_ms_p99 = {:.4}\n", self.frame_ms_p99));
        out.push_str(&format!("map_mb_per_min = {:.4}\n", self.map_mb_per_min));
        out
    }
}

/// The whole baselines file.
#[derive(Debug, Default, Clone)]
pub struct BaselineFile {
    entries: BTreeMap<String, Baseline>,
}

impl BaselineFile {
    /// Load, or return empty if the file does not exist yet.
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Ok(Self::parse(&text))
    }

    /// Parse the hand-written TOML subset.
    #[must_use]
    pub fn parse(text: &str) -> Self {
        let mut entries = BTreeMap::new();
        let mut current: Option<Baseline> = None;

        let flush = |current: &mut Option<Baseline>, entries: &mut BTreeMap<String, Baseline>| {
            if let Some(b) = current.take() {
                entries.insert(b.name.clone(), b);
            }
        };

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line == "[[sequence]]" {
                flush(&mut current, &mut entries);
                current = Some(Baseline {
                    name: String::new(),
                    ate_rmse: None,
                    lost_fraction: 0.0,
                    frame_ms_p99: 0.0,
                    map_mb_per_min: 0.0,
                });
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let (key, value) = (key.trim(), value.trim().trim_matches('"'));
            let Some(entry) = current.as_mut() else {
                continue;
            };
            match key {
                "name" => entry.name = value.to_string(),
                "ate_rmse" => entry.ate_rmse = value.parse().ok(),
                "lost_fraction" => entry.lost_fraction = value.parse().unwrap_or(0.0),
                "frame_ms_p99" => entry.frame_ms_p99 = value.parse().unwrap_or(0.0),
                "map_mb_per_min" => entry.map_mb_per_min = value.parse().unwrap_or(0.0),
                _ => {}
            }
        }
        flush(&mut current, &mut entries);
        BaselineFile { entries }
    }

    /// Look up a sequence.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Baseline> {
        self.entries.get(name)
    }

    /// Insert or replace.
    pub fn set(&mut self, baseline: Baseline) {
        self.entries.insert(baseline.name.clone(), baseline);
    }

    /// Recorded sequences.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Serialise. Sorted, so a re-record produces a reviewable diff rather than
    /// a reordering.
    #[must_use]
    pub fn to_toml(&self) -> String {
        let mut out = String::from(
            "# Tier-2 regression baselines. spec.md §6: \"CI fails on regression\n\
             # beyond tolerance.\"\n\
             #\n\
             # Regenerate with: cargo xtask regen-baselines --confirm\n\
             # A change here must be explained in the commit message.\n\n",
        );
        for entry in self.entries.values() {
            out.push_str(&entry.to_toml());
            out.push('\n');
        }
        out
    }

    /// Write to disk.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, self.to_toml())
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::AteReport;

    fn report(rmse: Scalar, lost: Scalar, p99: Scalar, mb: Scalar) -> SequenceReport {
        SequenceReport {
            failures: Default::default(),
            stages: Default::default(),
            loop_stats: (0, 0, 0),
            rpe_frame: None,
            rpe_short: None,
            name: "MH_01_easy".into(),
            frames: 1000,
            sequence_frames: 1000,
            lost_frames: (lost * 1000.0) as usize,
            ate_segment: None,
            ate: Some(AteReport {
                rmse,
                median: rmse,
                max: rmse,
                pairs: 1000,
                scale: 1.0,
                unmatched_fraction: 0.0,
            }),
            rpe_1s: None,
            scale_error_percent: None,
            campos: Vec::new(),
            frame_ms_median: p99 * 0.4,
            frame_ms_p99: p99,
            keyframes: 40,
            map_mb_per_min: mb,
            duration_s: 60.0,
            frames_with_prior: 1000,
            frames_with_attitude: 1000,
            calibration_pairs: None,
            effective_tier: 2,
            inter_frame_err: None,
            l1_tilt_deg: None,
            l1_yaw_drift: None,
            focal_vs_truth: None,
        }
    }

    fn baseline() -> Baseline {
        Baseline::from_report(&report(0.050, 0.02, 10.0, 2.0))
    }

    #[test]
    fn an_unchanged_run_passes() {
        assert!(baseline().check(&report(0.050, 0.02, 10.0, 2.0)).is_none());
    }

    #[test]
    fn an_improvement_passes() {
        // Getting better must never fail the wall.
        assert!(baseline().check(&report(0.020, 0.00, 5.0, 1.0)).is_none());
    }

    #[test]
    fn small_noise_passes_but_a_real_regression_fails() {
        let b = baseline();
        // Within the 10% band.
        assert!(b.check(&report(0.054, 0.02, 10.0, 2.0)).is_none());
        // Beyond it.
        let failure = b
            .check(&report(0.070, 0.02, 10.0, 2.0))
            .expect("regression");
        assert!(failure.contains("ATE"), "{failure}");
    }

    #[test]
    fn a_tracker_that_gives_up_is_caught_even_though_its_ate_improves() {
        // The failure mode a single ATE number cannot see: dropping the hard
        // frames makes the remaining trajectory look excellent.
        let failure = baseline()
            .check(&report(0.010, 0.60, 10.0, 2.0))
            .expect("regression");
        assert!(failure.contains("lost"), "{failure}");
    }

    #[test]
    fn losing_ground_truth_does_not_silently_disable_the_wall() {
        let mut r = report(0.05, 0.02, 10.0, 2.0);
        r.ate = None;
        let failure = baseline().check(&r).expect("regression");
        assert!(failure.contains("ATE unavailable"), "{failure}");
    }

    #[test]
    fn unbounded_map_growth_is_caught() {
        // spec.md §9: unbounded map memory kills the tab.
        let failure = baseline()
            .check(&report(0.05, 0.02, 10.0, 8.0))
            .expect("regression");
        assert!(failure.contains("map growth"), "{failure}");
    }

    #[test]
    fn a_tail_latency_blowup_is_caught() {
        let failure = baseline()
            .check(&report(0.05, 0.02, 40.0, 2.0))
            .expect("regression");
        assert!(failure.contains("p99"), "{failure}");
    }

    #[test]
    fn every_regression_is_reported_not_just_the_first() {
        let failure = baseline()
            .check(&report(0.5, 0.9, 100.0, 20.0))
            .expect("regression");
        for expected in ["ATE", "lost", "p99", "map growth"] {
            assert!(
                failure.contains(expected),
                "missing {expected} in: {failure}"
            );
        }
    }

    #[test]
    fn the_file_round_trips() {
        let mut file = BaselineFile::default();
        file.set(baseline());
        file.set(Baseline {
            name: "V1_03_difficult".into(),
            ate_rmse: None,
            lost_fraction: 0.11,
            frame_ms_p99: 14.0,
            map_mb_per_min: 3.0,
        });

        let parsed = BaselineFile::parse(&file.to_toml());
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed.get("MH_01_easy"), file.get("MH_01_easy"));
        // A sequence with no ground truth must survive the round trip as None,
        // not as 0.0 — which would be an impossibly good baseline.
        assert_eq!(parsed.get("V1_03_difficult").unwrap().ate_rmse, None);
    }

    #[test]
    fn the_file_is_sorted_so_a_rerecord_diffs_cleanly() {
        let mut file = BaselineFile::default();
        for name in ["V2_03", "MH_01", "V1_01"] {
            file.set(Baseline {
                name: name.into(),
                ate_rmse: Some(0.05),
                lost_fraction: 0.0,
                frame_ms_p99: 1.0,
                map_mb_per_min: 1.0,
            });
        }
        let toml = file.to_toml();
        let order: Vec<usize> = ["MH_01", "V1_01", "V2_03"]
            .iter()
            .map(|n| toml.find(n).expect("present"))
            .collect();
        assert!(
            order.windows(2).all(|w| w[0] < w[1]),
            "not sorted: {order:?}"
        );
    }

    #[test]
    fn an_empty_or_garbage_file_parses_to_nothing_rather_than_panicking() {
        assert_eq!(BaselineFile::parse("").len(), 0);
        assert_eq!(
            BaselineFile::parse("nonsense\n[[not-a-sequence]]\nx = ").len(),
            0
        );
    }
}
