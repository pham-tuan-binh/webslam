//! Per-sequence reporting.
//!
//! spec.md §6 asks for particular numbers in particular units, and asks for
//! some of them *not* to be aggregated. This module is where that discipline is
//! encoded once so no caller has to remember it:
//!
//! - ATE after **Sim(3)** alignment for L3, because L3 claims no scale.
//! - **p99** frame time, not mean (§6 L4: "Measure frame-time tail (p99), not
//!   mean").
//! - Map memory in **MB/min**, because the absolute number is meaningless
//!   without a duration.
//! - The **Campos curve** for scale, at 2 s and 10 s.

use wslam_core::{stats, Scalar};

use crate::dataset::Sequence;
use crate::metrics::{self, AteReport, Stamped};

/// Everything measured from one replayed sequence.
#[derive(Debug, Clone)]
pub struct SequenceReport {
    /// Sequence name, e.g. `MH_01_easy`.
    pub name: String,
    /// Frames actually processed. With `--limit` this is below the sequence
    /// length, and using the sequence length here made the loss rate read 4.2%
    /// when 155 of 300 processed frames had produced no pose.
    pub frames: usize,
    /// Frames in the sequence on disk.
    pub sequence_frames: usize,
    /// Frames that produced no usable pose.
    pub lost_frames: usize,
    /// ATE after Sim(3) alignment over the **whole** trajectory.
    ///
    /// Kept only for continuity with older baselines. When `segments > 1` this
    /// number is measuring the seams between coordinate frames, not tracking
    /// quality — read `ate_segment` instead.
    pub ate: Option<AteReport>,
    /// ATE over the largest coordinate-frame segment, with the segment count
    /// and the fraction of poses it covers. This is the honest figure.
    pub ate_segment: Option<(AteReport, usize, Scalar)>,
    /// Relative pose error: `(translation, rotation_radians, effective_delta_s)`.
    /// In ground-truth units.
    pub rpe_1s: Option<(Scalar, Scalar, Scalar)>,
    /// RPE over the shortest measurable window — consecutive emitted poses.
    ///
    /// This is the metric an AR consumer actually feels. Absolute error can be
    /// large and slowly varying without anything looking wrong; inter-frame
    /// error is jitter, and it is visible immediately. Reported separately from
    /// `rpe_1s` because the two can move in opposite directions: a filter that
    /// smooths pose will improve this and worsen drift.
    pub rpe_frame: Option<(Scalar, Scalar, Scalar)>,
    /// RPE over ~0.1 s, the horizon over which a hand moves perceptibly.
    pub rpe_short: Option<(Scalar, Scalar, Scalar)>,
    /// Scale error percentage against metric ground truth.
    pub scale_error_percent: Option<Scalar>,
    /// Scale error at the Campos checkpoints.
    pub campos: Vec<(Scalar, Option<Scalar>)>,
    /// Median per-frame processing time, ms.
    pub frame_ms_median: Scalar,
    /// p99 per-frame processing time, ms. The number that matters.
    pub frame_ms_p99: Scalar,
    /// Keyframes retained at the end.
    pub keyframes: usize,
    /// Map memory growth, MB per minute.
    pub map_mb_per_min: Scalar,
    /// Sequence duration, seconds.
    pub duration_s: Scalar,
    /// Frames on which L1's prediction was used (gated).
    pub frames_with_prior: u64,
    /// Loop closures accepted, rejected, and relocalizations.
    pub loop_stats: (usize, usize, usize),
    /// Mean per-stage cost, for attributing the frame budget.
    pub stages: wslam::layers::track::StageTimings,
    /// Why poseless frames were poseless.
    pub failures: wslam::layers::track::FailureCounts,
    /// Frames on which L1 had an attitude at all. Zero at tier 2 is the wiring
    /// alarm; `frames_with_prior` being lower is the gate doing its job.
    pub frames_with_attitude: u64,
    /// Matched pairs L2 accepted, or `None` when L2 is disabled because the
    /// caller supplied known intrinsics. Distinguished so a legitimate "off"
    /// does not read as "silently broken" — replay always supplies the
    /// dataset's published calibration, so 0 here is expected, and a future
    /// reader should not go hunting for a bug.
    pub calibration_pairs: Option<usize>,
    /// Sensor tier actually in force.
    pub effective_tier: u8,
    /// L1 tilt error against ground-truth orientation: `(rms_deg, max_deg)`.
    /// Roll and pitch are observable from gravity, so this should stay small.
    pub l1_tilt_deg: Option<(Scalar, Scalar)>,
    /// L1 yaw drift against ground truth, offset-removed: `(final_deg, rate_deg_per_min)`.
    /// Yaw is unobservable inertially, so it drifts; the question is how fast.
    pub l1_yaw_drift: Option<(Scalar, Scalar)>,
    /// Error in the **inter-frame** rotation L1 hands to L3's flow prediction:
    /// `(rms_deg, p95_deg, induced_px_at_edge)`.
    ///
    /// This, not the absolute attitude error, is what decides whether the prior
    /// helps: a constant attitude offset cancels in the frame-to-frame
    /// difference. The pixel figure is the displacement the error induces at
    /// the image edge, which is what KLT has to absorb.
    pub inter_frame_err: Option<(Scalar, Scalar, Scalar)>,
    /// L2's estimate scored against the dataset's published calibration:
    /// `(estimated_px, truth_px, relative_error)`.
    ///
    /// `Some` only when the run hid the calibration and asked L2 to find it —
    /// the device-realistic path, since a browser never knows its focal length.
    /// This is the L2 metric spec.md §6 L2 specifies, `|f̂ − f| / f`, measured on
    /// real imagery: the dataset's `sensor.yaml` plays the part the ChArUco
    /// board plays on a phone.
    pub focal_vs_truth: Option<(Scalar, Scalar, Scalar)>,
}

impl SequenceReport {
    /// Compute every metric from a completed replay.
    // Every argument is a CLI ablation knob. Grouping them into a struct
    // would move the same list one level down and make the call sites read
    // worse, so the lint is allowed rather than worked around.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        sequence: &Sequence,
        estimated: &[Stamped],
        frames_processed: usize,
        lost_frames: usize,
        frame_times_ms: &[Scalar],
        slam: &wslam::WebSlam,
        intrinsics_truth: Option<wslam_core::CameraIntrinsics>,
        tilt_errors_deg: &[Scalar],
        yaw_errors_deg: &[Scalar],
        inter_frame_err_deg: &[Scalar],
        loop_stats: (usize, usize, usize),
        stage_mean: wslam::layers::track::StageTimings,
    ) -> Self {
        const TOLERANCE: Scalar = 0.02;
        let debug = slam.debug();
        let duration = sequence.duration().max(1e-9);

        SequenceReport {
            name: sequence.name.clone(),
            frames: frames_processed,
            sequence_frames: sequence.len(),
            lost_frames,
            // Sim(3): L3 makes no metric claim, so aligning with scale fixed
            // would measure the arbitrary unit choice (spec.md §6 L3).
            ate: metrics::ate(estimated, &sequence.truth, TOLERANCE, true),
            ate_segment: metrics::ate_largest_segment(estimated, &sequence.truth, TOLERANCE, true),
            rpe_1s: metrics::rpe(estimated, &sequence.truth, 1.0, TOLERANCE),
            // 0.0 requests the smallest stride the data supports (>=1).
            rpe_frame: metrics::rpe(estimated, &sequence.truth, 0.0, TOLERANCE),
            rpe_short: metrics::rpe(estimated, &sequence.truth, 0.1, TOLERANCE),
            scale_error_percent: slam
                .scale()
                .source
                .is_metric()
                .then(|| metrics::scale_error_percent(estimated, &sequence.truth, TOLERANCE))
                .flatten(),
            // Gated on metric scale: on an up-to-scale session this reports the
            // arbitrary bootstrap baseline as a scale error (1251% measured).
            campos: if slam.scale().source.is_metric() {
                metrics::campos_curve(estimated, &sequence.truth, TOLERANCE, &[2.0, 10.0])
            } else {
                Vec::new()
            },
            frame_ms_median: stats::median(frame_times_ms).unwrap_or(0.0),
            frame_ms_p99: stats::percentile(frame_times_ms, 0.99).unwrap_or(0.0),
            keyframes: debug.keyframe_count(),
            map_mb_per_min: (debug.map_memory_bytes() as Scalar / 1_048_576.0) / (duration / 60.0),
            duration_s: duration,
            frames_with_prior: slam.frames_with_rotation_prior(),
            failures: slam.failures(),
            stages: stage_mean,
            loop_stats,
            frames_with_attitude: slam.frames_with_attitude(),
            calibration_pairs: slam
                .intrinsics_are_estimated()
                .then(|| slam.calibration_pairs()),
            effective_tier: slam.effective_tier().number(),
            l1_tilt_deg: (!tilt_errors_deg.is_empty()).then(|| {
                let rms = (tilt_errors_deg.iter().map(|e| e * e).sum::<Scalar>()
                    / tilt_errors_deg.len() as Scalar)
                    .sqrt();
                let max = tilt_errors_deg.iter().cloned().fold(0.0, Scalar::max);
                (rms, max)
            }),
            l1_yaw_drift: (!yaw_errors_deg.is_empty()).then(|| {
                let final_deg = *yaw_errors_deg.last().expect("non-empty");
                let minutes = (duration / 60.0).max(1e-9);
                (final_deg, final_deg / minutes)
            }),
            inter_frame_err: (!inter_frame_err_deg.is_empty()).then(|| {
                let rms = (inter_frame_err_deg.iter().map(|e| e * e).sum::<Scalar>()
                    / inter_frame_err_deg.len() as Scalar)
                    .sqrt();
                let p95 = stats::percentile(inter_frame_err_deg, 0.95).unwrap_or(rms);
                // Displacement at the image edge for a p95-sized rotation error.
                let focal = sequence.intrinsics.map_or(458.0, |k| k.fx);
                let px = focal * p95.to_radians().tan();
                (rms, p95, px)
            }),
            focal_vs_truth: intrinsics_truth.map(|truth| {
                let estimated = slam.intrinsics().fx;
                (estimated, truth.fx, (estimated - truth.fx).abs() / truth.fx)
            }),
        }
    }

    /// Fraction of frames that produced no pose.
    #[must_use]
    pub fn lost_fraction(&self) -> Scalar {
        self.lost_frames as Scalar / self.frames.max(1) as Scalar
    }

    /// One line, for a progress log.
    #[must_use]
    pub fn summary(&self) -> String {
        let ate = match &self.ate {
            Some(a) => format!("{:.4} m", a.rmse),
            None => "no ground truth".to_string(),
        };
        format!(
            "{:<22} ATE {ate:>16}  lost {:>5.1}%  p99 {:>6.2} ms  {:>4} kf",
            self.name,
            100.0 * self.lost_fraction(),
            self.frame_ms_p99,
            self.keyframes
        )
    }

    /// The full report for one sequence.
    #[must_use]
    pub fn detailed(&self) -> String {
        let mut out = format!("{}\n{}\n", self.name, "-".repeat(self.name.len().max(40)));
        out.push_str(&format!(
            "  frames            {} processed{} ({:.1}% produced no pose)\n",
            self.frames,
            if self.frames < self.sequence_frames {
                format!(" of {} in the sequence", self.sequence_frames)
            } else {
                format!(" over {:.1}s", self.duration_s)
            },
            100.0 * self.lost_fraction()
        ));
        // A loss rate without a cause is not actionable. Break it down.
        let f = self.failures;
        if f.total() > 0 {
            out.push_str(&format!(
                "  no-pose cause     bootstrap {}, few correspondences {}, ransac {}, few inliers {}\n",
                f.awaiting_bootstrap,
                f.too_few_correspondences,
                f.ransac_failed,
                f.too_few_inliers
            ));
        }
        match &self.ate_segment {
            Some((a, segments, coverage)) => {
                out.push_str(&format!(
                    "  ATE (per segment) {a}\n                    largest of {segments} \
                     coordinate frame(s), covering {:.1}% of emitted poses{}\n",
                    100.0 * coverage,
                    if *segments > 1 {
                        "   <-- trajectory is discontinuous"
                    } else {
                        ""
                    }
                ));
            }
            None => out.push_str("  ATE (per segment) no ground truth in this sequence\n"),
        }
        if let (Some(whole), Some((seg, n, _))) = (&self.ate, &self.ate_segment) {
            if *n > 1 {
                out.push_str(&format!(
                    "  ATE (whole)       {:.4} m — measures the seams, not the tracking \
                     (segment: {:.4} m)\n",
                    whole.rmse, seg.rmse
                ));
            }
        }
        for (label, m) in [("inter-frame", self.rpe_frame), ("short", self.rpe_short)] {
            if let Some((t, r, delta)) = m {
                out.push_str(&format!(
                    "  RPE {label:<11} {:.4} m, {:.3} deg  over {delta:.3}s\n",
                    t,
                    r.to_degrees()
                ));
            }
        }
        if let Some((t, r, delta)) = self.rpe_1s {
            // The measured window, not the requested one.
            out.push_str(&format!(
                "  RPE over {delta:.2}s     {:.4} m, {:.3} deg\n",
                t,
                r.to_degrees()
            ));
        }
        match self.scale_error_percent {
            Some(e) => out.push_str(&format!("  scale error       {e:.2}%\n")),
            None => out.push_str("  scale error       n/a (session is up to scale)\n"),
        }
        if !self.campos.is_empty() {
            // spec.md §6 L5: report our 2 s and 10 s numbers against Campos et
            // al.'s 5% and 1%, so the comparison is printed rather than left
            // for the reader to look up.
            out.push_str("  Campos curve      ");
            for (at, value) in &self.campos {
                let target = if *at <= 2.0 { 5.0 } else { 1.0 };
                match value {
                    Some(v) => out.push_str(&format!("{at:.0}s: {v:.2}% (target {target:.0}%)  ")),
                    None => out.push_str(&format!("{at:.0}s: n/a  ")),
                }
            }
            out.push('\n');
        }
        out.push_str(&format!(
            "  L4                loops {} accepted / {} rejected, {} relocalizations\n",
            self.loop_stats.0, self.loop_stats.1, self.loop_stats.2,
        ));
        out.push_str(&format!(
            "  stage mean ms     pyramid {:.1}, corners {:.1}, flow {:.1}, pnp {:.1}\n",
            self.stages.pyramid_ms, self.stages.corners_ms, self.stages.flow_ms, self.stages.pnp_ms,
        ));
        out.push_str(&format!(
            "  frame time        median {:.2} ms, p99 {:.2} ms\n",
            self.frame_ms_median, self.frame_ms_p99
        ));
        out.push_str(&format!(
            "  map               {} keyframes, {:.2} MB/min\n",
            self.keyframes, self.map_mb_per_min
        ));
        if let Some((rms, max)) = self.l1_tilt_deg {
            // spec.md §6 L1: roll/pitch error vs gravity. Gravity makes these
            // observable, so a large number here means the filter is broken,
            // not that the problem is hard.
            out.push_str(&format!(
                "  L1 tilt vs truth  {rms:.2} deg rms, {max:.2} deg max{}\n",
                if rms > 5.0 { "   <-- L1 IS WRONG" } else { "" }
            ));
        }
        if let Some((final_deg, rate)) = self.l1_yaw_drift {
            out.push_str(&format!(
                "  L1 yaw drift      {final_deg:+.2} deg total, {rate:+.2} deg/min\n"
            ));
        }
        if let Some((rms, p95, px)) = self.inter_frame_err {
            out.push_str(&format!(
                "  L1 inter-frame    {rms:.3} deg rms, {p95:.3} deg p95 -> {px:.1} px at edge{}\n",
                if px > 10.0 {
                    "   <-- prior is worse than no prior"
                } else {
                    ""
                }
            ));
        }
        if let Some((estimated, truth, relative)) = self.focal_vs_truth {
            // spec.md §6 L2 reports focal error as a relative figure, and gives
            // the ablations as the gate. This is that number on real imagery.
            out.push_str(&format!(
                "  L2 focal          {estimated:.1} px vs {truth:.1} px published \
                 ({:+.2}% error){}\n",
                100.0 * (estimated - truth) / truth,
                if relative > 0.05 {
                    "   <-- over 5%"
                } else {
                    ""
                }
            ));
        }
        // Wiring health. A prior count of zero at tier 2 means L1 never reached
        // L3 and L2 never ran — which degrades the ATE above without producing
        // a single error, so it belongs on the same page as the number.
        out.push_str(&format!(
            "  wiring            tier {}, L1 attitude {}/{} frames, prior used {}, L2 {}{}\n",
            self.effective_tier,
            self.frames_with_attitude,
            self.frames,
            self.frames_with_prior,
            match self.calibration_pairs {
                Some(n) => format!("pairs {n}"),
                None => "off (known intrinsics)".to_string(),
            },
            if self.effective_tier >= 2 && self.frames_with_attitude == 0 {
                "   <-- L1 NOT REACHING L3"
            } else {
                ""
            }
        ));
        out
    }
}

/// A table across sequences.
///
/// Per sequence, never pooled — spec.md §6 is explicit that pooling hides the
/// device- and sequence-specific failures that are the actual bug surface.
#[must_use]
pub fn table(reports: &[SequenceReport]) -> String {
    let mut out = String::from(
        "sequence               ATE rmse    lost%    p99 ms   kf   MB/min\n\
         ---------------------------------------------------------------\n",
    );
    for r in reports {
        out.push_str(&format!(
            "{:<22} {:>8}  {:>6.1}  {:>8.2} {:>4}  {:>7.2}\n",
            r.name,
            match &r.ate {
                Some(a) => format!("{:.4}", a.rmse),
                None => "—".to_string(),
            },
            100.0 * r.lost_fraction(),
            r.frame_ms_p99,
            r.keyframes,
            r.map_mb_per_min
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(name: &str, rmse: Scalar, lost: usize, frames: usize) -> SequenceReport {
        SequenceReport {
            failures: Default::default(),
            stages: Default::default(),
            name: name.into(),
            frames,
            sequence_frames: frames,
            lost_frames: lost,
            ate_segment: None,
            ate: Some(AteReport {
                rmse,
                median: rmse * 0.8,
                max: rmse * 3.0,
                pairs: frames - lost,
                scale: 1.0,
                unmatched_fraction: 0.0,
            }),
            rpe_1s: Some((0.01, 0.002, 1.0)),
            loop_stats: (0, 0, 0),
            rpe_frame: Some((0.001, 0.0002, 0.05)),
            rpe_short: Some((0.002, 0.0004, 0.1)),
            scale_error_percent: None,
            campos: vec![(2.0, Some(4.2)), (10.0, Some(0.9))],
            frame_ms_median: 8.0,
            frame_ms_p99: 21.0,
            keyframes: 42,
            map_mb_per_min: 1.5,
            duration_s: 60.0,
            frames_with_prior: 100,
            frames_with_attitude: 100,
            calibration_pairs: None,
            effective_tier: 2,
            inter_frame_err: None,
            l1_tilt_deg: None,
            l1_yaw_drift: None,
            focal_vs_truth: None,
        }
    }

    #[test]
    fn lost_fraction_is_a_fraction() {
        let r = report("MH_01", 0.05, 25, 100);
        assert!((r.lost_fraction() - 0.25).abs() < 1e-12);
    }

    #[test]
    fn lost_fraction_survives_an_empty_sequence() {
        let mut r = report("empty", 0.0, 0, 1);
        r.frames = 0;
        assert!(r.lost_fraction().is_finite());
    }

    #[test]
    fn the_detailed_report_prints_the_campos_targets() {
        // spec.md §6 L5 asks for our numbers *against* Campos et al.'s 5% / 1%,
        // so the comparison must be on the page rather than in the reader's head.
        let text = report("MH_01", 0.05, 5, 100).detailed();
        assert!(text.contains("Campos curve"), "{text}");
        assert!(text.contains("target 5%"), "{text}");
        assert!(text.contains("target 1%"), "{text}");
    }

    #[test]
    fn the_detailed_report_names_p99_not_just_the_mean() {
        let text = report("MH_01", 0.05, 5, 100).detailed();
        assert!(text.contains("p99"), "{text}");
    }

    #[test]
    fn an_up_to_scale_session_says_so_rather_than_printing_zero() {
        let text = report("MH_01", 0.05, 5, 100).detailed();
        assert!(text.contains("up to scale"), "{text}");
    }

    #[test]
    fn the_table_lists_every_sequence_separately() {
        // Never pooled: spec.md §6 says pooling hides the failures that matter.
        let reports = vec![
            report("MH_01_easy", 0.04, 2, 100),
            report("V1_03_difficult", 0.19, 30, 100),
        ];
        let text = table(&reports);
        assert!(text.contains("MH_01_easy"));
        assert!(text.contains("V1_03_difficult"));
        assert_eq!(text.lines().count(), 4); // header, rule, two rows
    }

    #[test]
    fn a_sequence_without_ground_truth_renders_without_panicking() {
        let mut r = report("no_gt", 0.0, 0, 100);
        r.ate = None;
        r.rpe_1s = None;
        let text = r.detailed();
        assert!(text.contains("no ground truth"), "{text}");
        assert!(table(std::slice::from_ref(&r)).contains("—"));
    }
}
