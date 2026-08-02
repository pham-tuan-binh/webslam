//! The native replay harness — Tier 2, **the regression wall** (spec.md §6).
//!
//! Runs the same code that ships to the browser, at full native speed, against
//! datasets with published reference numbers. That doubles as port validation:
//! EuRoC and TUM-VI have known ATE figures, so a divergence points at our port
//! rather than at the algorithm.
//!
//! ```text
//! wslam-replay run datasets/euroc/MH_01_easy       one sequence, verbose
//! wslam-replay regress                             every sequence vs baselines
//! wslam-replay regress --write                     re-record the baselines
//! ```

mod baseline;
mod dataset;
mod metrics;
mod report;

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use crate::baseline::{Baseline, BaselineFile};
use crate::metrics::Stamped;
use crate::report::SequenceReport;
use wslam_core::HostClock;

#[derive(Parser)]
#[command(name = "wslam-replay", about = "native replay harness", version)]
struct Cli {
    /// Log level.
    #[arg(long, default_value = "info", global = true)]
    log: String,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Replay one sequence and print its metrics.
    Run {
        /// Sequence directory.
        sequence: PathBuf,
        /// Stop after this many frames.
        #[arg(long)]
        limit: Option<usize>,
        /// Write a rerun session for later scrubbing.
        #[arg(long)]
        rrd: Option<PathBuf>,
        /// RNG seed. Logged, and reproducible (spec.md §6).
        #[arg(long, default_value_t = 0x5eed)]
        seed: u64,
        /// Local-BA window size. `0` disables it, which is the ablation that
        /// shows what joint optimisation is worth.
        #[arg(long)]
        ba_window: Option<usize>,
        /// Feature budget, for sweeping landmark supply.
        #[arg(long)]
        max_features: Option<usize>,
        /// Use the GPU image front-end. Requires the `gpu` build feature.
        #[arg(long)]
        gpu: bool,
        /// Forward-backward tolerance in pixels.
        #[arg(long)]
        klt_fb: Option<f64>,
        /// Fixed anchor poses in the BA window.
        #[arg(long)]
        ba_fixed: Option<usize>,
        /// Max per-solve window scale change.
        #[arg(long)]
        ba_scale: Option<f64>,
        /// Trained vocabulary artifact. Without one, L4 is inert.
        #[arg(long)]
        vocab: Option<PathBuf>,
        /// Evaluate the loop-closure-corrected trajectory instead of the poses as
        /// they were emitted. Off by default: see `refined_trajectory`.
        #[arg(long)]
        refined: bool,
        /// Hold landmarks fixed in local BA.
        #[arg(long)]
        ba_motion_only: bool,
        /// Fixed context keyframes behind the BA window.
        #[arg(long)]
        ba_context: Option<usize>,
        /// LM iterations per BA solve.
        #[arg(long)]
        ba_iters: Option<usize>,
        /// Sensor tier ceiling. `1` disables the orientation prior entirely,
        /// which isolates whether the prior is helping or hurting.
        #[arg(long, default_value_t = 2)]
        tier: u8,
        /// How to interpret the dataset's `T_BS`.
        ///
        /// `auto` uses it as `R_body_camera`, which is the Furgale convention
        /// EuRoC documents. `inverse` and `identity` exist because a frame
        /// convention is exactly the kind of thing that is easier to settle by
        /// measurement than by re-reading a spec, and getting it backwards is
        /// silent.
        #[arg(long, default_value = "auto")]
        extrinsic: String,
        /// Hide the dataset's published calibration and make L2 estimate it.
        ///
        /// This is the device-realistic path: a browser never knows its focal
        /// length. The published `sensor.yaml` is then held back as **ground
        /// truth** and the run reports relative focal error `|f̂ − f| / f` —
        /// which is exactly the L2 metric spec.md §6 L2 asks for, on real
        /// imagery, without needing a ChArUco board.
        #[arg(long)]
        estimate_intrinsics: bool,
    },
    /// Run every sequence and compare against the checked-in baselines.
    Regress {
        /// Dataset root.
        #[arg(long, default_value = "datasets/euroc")]
        root: PathBuf,
        /// Baselines file.
        #[arg(long, default_value = "harness/baselines/euroc.toml")]
        baselines: PathBuf,
        /// Overwrite the baselines with this run's numbers.
        #[arg(long)]
        write: bool,
        /// Run only the fast subset (per-commit CI).
        #[arg(long)]
        subset: bool,
        /// Run everything (nightly).
        #[arg(long)]
        full: bool,
        /// Write a rerun session.
        #[arg(long)]
        rrd: Option<PathBuf>,
        #[arg(long, default_value_t = 0x5eed)]
        seed: u64,
    },
    /// Train a vocabulary artifact from a descriptor dump.
    /// Dump FAST+BRIEF descriptors from a sequence, to train a vocabulary.
    DumpDescriptors {
        /// Sequence directory.
        sequence: PathBuf,
        /// Output descriptor blob.
        #[arg(long)]
        out: PathBuf,
        /// Use every Nth frame.
        #[arg(long, default_value_t = 10)]
        stride: usize,
    },
    TrainVocab {
        descriptors: PathBuf,
        #[arg(long, default_value_t = 10)]
        branching: usize,
        #[arg(long, default_value_t = 5)]
        depth: usize,
        #[arg(long, default_value_t = 20260801)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    env_logger::Builder::new()
        .parse_filters(&cli.log)
        .format_timestamp(None)
        .init();

    match cli.command {
        Command::Run {
            sequence,
            limit,
            rrd,
            seed,
            tier,
            extrinsic,
            estimate_intrinsics,
            ba_window,
            max_features,
            gpu,
            klt_fb,
            ba_fixed,
            ba_scale,
            vocab,
            refined,
            ba_motion_only,
            ba_context,
            ba_iters,
        } => {
            let report = replay(
                &sequence,
                limit,
                rrd.as_deref(),
                seed,
                estimate_intrinsics,
                tier,
                &extrinsic,
                ba_window,
                max_features,
                gpu,
                klt_fb,
                ba_fixed,
                ba_scale,
                vocab.as_deref(),
                refined,
                ba_motion_only,
                ba_context,
                ba_iters,
            )?;
            println!("\n{}", report.detailed());
            Ok(())
        }
        Command::Regress {
            root,
            baselines,
            write,
            subset,
            full,
            rrd,
            seed,
        } => regress(
            &root,
            &baselines,
            write,
            subset && !full,
            rrd.as_deref(),
            seed,
        ),
        Command::DumpDescriptors {
            sequence,
            out,
            stride,
        } => {
            let seq = dataset::load_euroc(&sequence)?;
            let mut bytes: Vec<u8> = Vec::new();
            for i in 0..seq.frames.len() {
                if i % stride != 0 {
                    continue;
                }
                let image = seq.load_frame(i)?.image;
                let detections = wslam_map::fast_keypoints(&image, 20, 400);
                if detections.is_empty() {
                    continue;
                }
                let (kp, ang): (Vec<_>, Vec<_>) = detections.into_iter().unzip();
                for d in wslam_map::describe(&image, &kp, &ang) {
                    bytes.extend_from_slice(&d.0);
                }
            }
            std::fs::write(&out, &bytes)?;
            println!("wrote {} descriptors to {}", bytes.len() / 32, out.display());
            Ok(())
        }
        Command::TrainVocab {
            descriptors,
            branching,
            depth,
            seed,
            out,
        } => train_vocab(&descriptors, branching, depth, seed, &out),
    }
}

/// Replay one sequence end to end.
fn replay(
    path: &std::path::Path,
    limit: Option<usize>,
    rrd: Option<&std::path::Path>,
    seed: u64,
    estimate_intrinsics: bool,
    tier: u8,
    extrinsic: &str,
    ba_window: Option<usize>,
    max_features: Option<usize>,
    gpu: bool,
    klt_fb: Option<f64>,
    ba_fixed: Option<usize>,
    ba_scale: Option<f64>,
    vocab: Option<&std::path::Path>,
    refined_trajectory: bool,
    ba_motion_only: bool,
    ba_context: Option<usize>,
    ba_iters: Option<usize>,
) -> Result<SequenceReport> {
    let sequence = dataset::load_euroc(path)?;
    let first = sequence
        .frames
        .first()
        .map(|(t, _)| *t)
        .context("sequence has no frames")?;

    let mut config = wslam::SlamConfig::new(
        sequence.intrinsics.map_or(752, |k| k.width),
        sequence.intrinsics.map_or(480, |k| k.height),
    );
    config.seed = seed;
    config.track.use_gpu = gpu;
    config.track.local_ba_motion_only = ba_motion_only;
    if let Some(i) = ba_iters {
        config.track.local_ba_iterations = i;
    }
    if let Some(c) = ba_context {
        config.track.local_ba_context = c;
    }
    if let Some(f) = ba_fixed {
        config.track.local_ba_fixed = f;
    }
    if let Some(g) = ba_scale {
        config.track.local_ba_max_scale_change = g;
    }
    if let Some(t) = klt_fb {
        config.track.klt_forward_backward = (t > 0.0).then_some(t);
    }
    if let Some(n) = max_features {
        config.track.max_features = n;
        // Keep the refill trigger proportional, or a larger budget simply never
        // refills and the sweep measures nothing.
        config.track.min_features = (n as f64 * 0.24) as usize;
    }
    if let Some(w) = ba_window {
        config.track.local_ba_window = w;
    }
    config.tier = match tier {
        1 => wslam::SensorTier::VisionOnly,
        3 => wslam::SensorTier::TightVisualInertial,
        _ => wslam::SensorTier::VisionOrientation,
    };
    if estimate_intrinsics {
        // Device-realistic: the browser never knows its focal length, so hide
        // the published value and let L2 find it. The truth is kept aside and
        // scored in the report.
        config.intrinsics.known = None;
        config.intrinsics.estimate_online = true;
    } else {
        // The dataset publishes its intrinsics, so L2 has nothing to estimate.
        // Using them isolates L3: spec.md §6 L3 grades the tracker, not the
        // calibration.
        config.intrinsics.known = sequence.intrinsics;
    }

    // The published camera-IMU extrinsic. Without it L1's attitude reaches L2
    // and L3 in the wrong frame.
    match extrinsic {
        "identity" => {}
        "inverse" => {
            if let Some(r) = sequence.body_from_camera {
                config.body_from_camera = r.inverse();
            }
        }
        _ => {
            if let Some(r) = sequence.body_from_camera {
                config.body_from_camera = r;
            }
        }
    }

    let mut slam = wslam::WebSlam::new(config)?;
    if let Some(path) = vocab {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let v = wslam_map::Vocabulary::deserialize(&bytes)?;
        log::info!("loaded vocabulary: {} words", v.word_count());
        slam.set_vocabulary(std::sync::Arc::new(v))?;
    }
    let mut sink = wslam_viewer::make_sink(rrd);

    let count = limit.unwrap_or(sequence.len()).min(sequence.len());
    let mut estimated: Vec<Stamped> = Vec::with_capacity(count);
    let mut previous = first;
    let mut lost_frames = 0usize;
    let mut frame_times = Vec::with_capacity(count);
    let mut loops_accepted = 0usize;
    let mut loops_rejected = 0usize;
    let mut relocalizations = 0usize;
    let mut stage_sum = wslam::layers::track::StageTimings::default();
    let mut stage_n = 0usize;
    let host = wslam_core::time::StdHostClock::new();

    // L1 accuracy against the dataset's ground-truth orientation. EuRoC's
    // `state_groundtruth_estimate0` publishes `R_world_body`, which is exactly
    // what `OrientationFilter::attitude` estimates — so this is the spec.md §6
    // L1 measurement (tilt error, yaw drift) on real inertial data, with no
    // turntable required.
    //
    // Decomposed rather than reported as one angle: gravity makes roll and
    // pitch observable, and leaves yaw unobservable, so a single combined
    // number would be dominated by a drift the filter cannot help.
    let mut tilt_errors_deg: Vec<f64> = Vec::new();
    let mut yaw_errors_deg: Vec<f64> = Vec::new();
    let mut yaw_offset: Option<f64> = None;

    // The quantity `Tracker::predict` actually consumes is the rotation
    // *between* consecutive frames, not the absolute attitude. A constant
    // attitude offset — including all of the yaw drift — cancels in that
    // difference, so absolute error is the wrong thing to judge the prior by.
    // This measures the inter-frame rotation error against ground truth, and
    // converts it to the pixel displacement it induces at the image edge, which
    // is what decides whether KLT converges.
    let mut inter_frame_err_deg: Vec<f64> = Vec::new();
    let mut prev_pair: Option<(wslam_core::So3, wslam_core::So3)> = None;

    for index in 0..count {
        let frame = sequence.load_frame(index)?;

        // Feed the IMU up to this frame before the frame itself, so the
        // orientation prior is available when the tracker asks for it.
        for sample in sequence.imu_between(previous, frame.timestamp) {
            slam.push_imu(*sample);
        }
        previous = frame.timestamp;

        let started = host.elapsed_seconds();
        slam.push_frame(frame.clone());
        let pose = slam.step();
        for e in slam.take_events() {
            match e {
                wslam::SlamEvent::LoopClosure { accepted: true, .. } => loops_accepted += 1,
                wslam::SlamEvent::LoopClosure { accepted: false, .. } => loops_rejected += 1,
                wslam::SlamEvent::Relocalized { .. } => relocalizations += 1,
                _ => {}
            }
        }
        frame_times.push((host.elapsed_seconds() - started) * 1000.0);
        {
            let t = slam.stage_timings();
            stage_sum.pyramid_ms += t.pyramid_ms;
            stage_sum.corners_ms += t.corners_ms;
            stage_sum.flow_ms += t.flow_ms;
            stage_sum.pnp_ms += t.pnp_ms;
            stage_n += 1;
        }

        if let Some(truth) = nearest(&sequence.truth, frame.timestamp) {
            if let Some(estimate) = slam.body_attitude() {
                // Tilt: the angle between the two frames' idea of "down".
                let down = wslam_core::Vec3::new(0.0, 0.0, 1.0);
                let a = estimate.inverse().act(&down);
                let b = truth.rotation().inverse().act(&down);
                tilt_errors_deg.push(a.angle(&b).to_degrees());

                // Yaw, minus the arbitrary starting offset: L1 initialises its
                // heading at zero, ground truth does not.
                let yaw = |r: &wslam_core::So3| {
                    let m = r.matrix();
                    m[(1, 0)].atan2(m[(0, 0)])
                };
                let delta = yaw(&estimate) - yaw(&truth.rotation());
                let offset = *yaw_offset.get_or_insert(delta);
                let mut wrapped = delta - offset;
                while wrapped > std::f64::consts::PI {
                    wrapped -= std::f64::consts::TAU;
                }
                while wrapped < -std::f64::consts::PI {
                    wrapped += std::f64::consts::TAU;
                }
                yaw_errors_deg.push(wrapped.to_degrees());

                if let Some((prev_est, prev_truth)) = prev_pair {
                    let d_est = estimate.inverse().compose(&prev_est);
                    let d_truth = truth.rotation().inverse().compose(&prev_truth);
                    inter_frame_err_deg
                        .push(d_est.inverse().compose(&d_truth).angle().to_degrees());
                }
                prev_pair = Some((estimate, truth.rotation()));
            }
        }

        match pose {
            Some(pose) if pose.state.has_pose() => {
                estimated.push(Stamped {
                    timestamp: pose.timestamp,
                    // Into the body frame, which is what EuRoC's ground truth
                    // is expressed in. `T_world_body = T_world_camera *
                    // T_camera_body`. Skipping this leaves a ~0.038 m floor on
                    // every ATE, because the lever arm rotates with the body and
                    // a single Sim(3) cannot absorb a term that moves.
                    pose: match sequence.body_from_camera_se3 {
                        Some(t_bc) => pose.transform.compose(&t_bc.inverse()),
                        None => pose.transform,
                    },
                    epoch: pose.frame_epoch,
                });
                let debug = slam.debug();
                let landmarks = debug.landmarks();
                sink.log_frame(&wslam_viewer::SessionFrame {
                    timestamp: pose.timestamp,
                    frame_index: index as u64,
                    image: Some(&frame.image),
                    pose: Some(pose.transform),
                    covariance: Some(pose.covariance),
                    ground_truth: nearest(&sequence.truth, pose.timestamp),
                    landmarks: &landmarks,
                    scale: Some(pose.scale),
                    state: Some(pose.state),
                    ..Default::default()
                });
            }
            _ => lost_frames += 1,
        }

        if index % 200 == 0 && index > 0 {
            log::info!("  {index}/{count} frames, {} tracked", estimated.len());
        }
    }
    sink.finish();

    // Evaluate the trajectory as the map finally holds it, not as it was
    // emitted frame by frame. Loop closure corrects keyframes; without this the
    // correction is invisible to every metric.
    let refined = slam.refined_trajectory();
    let estimated = if refined_trajectory && refined.len() >= estimated.len() / 2 && !refined.is_empty() {
        log::info!(
            "evaluating {} refined poses (emitted {})",
            refined.len(),
            estimated.len()
        );
        refined
            .into_iter()
            .map(|(timestamp, pose)| Stamped {
                timestamp,
                // The same camera -> body conversion the emitted path applies.
                // Omitting it here was not a small offset: a single Sim(3)
                // cannot absorb a lever arm that rotates with the body, so
                // absolute error survived while inter-frame RPE went from
                // 0.130 deg to 0.834 deg. The bug is the frame convention, not
                // the reconstruction.
                pose: match sequence.body_from_camera_se3 {
                    Some(t_bc) => pose.compose(&t_bc.inverse()),
                    None => pose,
                },
                // The refined trajectory is anchored to one pose graph, so it
                // is by construction a single coordinate frame.
                epoch: 0,
            })
            .collect()
    } else {
        estimated
    };

    Ok(SequenceReport::build(
        &sequence,
        &estimated,
        count,
        lost_frames,
        &frame_times,
        &slam,
        // Held back as ground truth when L2 was asked to estimate.
        estimate_intrinsics.then_some(sequence.intrinsics).flatten(),
        &tilt_errors_deg,
        &yaw_errors_deg,
        &inter_frame_err_deg,
        (loops_accepted, loops_rejected, relocalizations),
        {
            let n = stage_n.max(1) as f64;
            wslam::layers::track::StageTimings {
                pyramid_ms: stage_sum.pyramid_ms / n,
                corners_ms: stage_sum.corners_ms / n,
                flow_ms: stage_sum.flow_ms / n,
                pnp_ms: stage_sum.pnp_ms / n,
                ..Default::default()
            }
        },
    ))
}

fn nearest(truth: &[Stamped], at: wslam_core::Timestamp) -> Option<wslam_core::Se3> {
    if truth.is_empty() {
        return None;
    }
    let index = truth
        .partition_point(|s| s.timestamp < at)
        .min(truth.len() - 1);
    let candidate = truth[index];
    (candidate.timestamp.since(at).abs() < 0.05).then_some(candidate.pose)
}

/// Compare every sequence against the checked-in baselines.
///
/// spec.md §6 Tier 2: *"Per-sequence ATE checked into `harness/baselines/` as
/// data; CI fails on regression beyond tolerance."*
fn regress(
    root: &std::path::Path,
    baselines_path: &std::path::Path,
    write: bool,
    subset: bool,
    rrd: Option<&std::path::Path>,
    seed: u64,
) -> Result<()> {
    let sequences = dataset::discover(root);
    if sequences.is_empty() {
        bail!(
            "no sequences under {}. Run `cargo xtask fetch-datasets euroc` first.",
            root.display()
        );
    }
    let sequences = if subset {
        // Per-commit CI runs two: one easy, one hard. The hard one is where
        // regressions show up first.
        sequences.into_iter().take(2).collect::<Vec<_>>()
    } else {
        sequences
    };

    let mut baselines = BaselineFile::load(baselines_path).unwrap_or_default();
    log::info!(
        "{} sequence(s) to run against {} recorded baseline(s)",
        sequences.len(),
        baselines.len()
    );
    let mut results = Vec::new();
    let mut failures = Vec::new();

    for path in &sequences {
        log::info!("replaying {}", path.display());
        let report = replay(path, None, rrd, seed, false, 2, "auto", None, None, false, None, None, None, None, false, false, None, None)?;
        println!("{}", report.summary());

        match baselines.get(&report.name) {
            Some(baseline) => {
                if let Some(reason) = baseline.check(&report) {
                    failures.push(format!("{}: {reason}", report.name));
                }
            }
            None if !write => {
                // Silently passing an unbaselined sequence would let a new
                // sequence ship with no regression protection at all.
                failures.push(format!(
                    "{}: no baseline recorded. Run `cargo xtask regen-baselines --confirm`.",
                    report.name
                ));
            }
            None => {}
        }
        results.push(report);
    }

    if write {
        for report in &results {
            baselines.set(Baseline::from_report(report));
        }
        baselines.save(baselines_path)?;
        println!(
            "\nwrote {} baselines to {}",
            results.len(),
            baselines_path.display()
        );
        println!("Explain the change in the commit message — this is the regression wall.");
        return Ok(());
    }

    println!("\n{}", report::table(&results));
    if failures.is_empty() {
        println!(
            "\n\x1b[32mall {} sequences within tolerance\x1b[0m",
            results.len()
        );
        Ok(())
    } else {
        for f in &failures {
            eprintln!("\x1b[31mREGRESSION\x1b[0m {f}");
        }
        bail!("{} sequence(s) regressed", failures.len())
    }
}

fn train_vocab(
    descriptors: &std::path::Path,
    branching: usize,
    depth: usize,
    seed: u64,
    out: &std::path::Path,
) -> Result<()> {
    let bytes =
        std::fs::read(descriptors).with_context(|| format!("reading {}", descriptors.display()))?;
    if bytes.len() % 32 != 0 {
        bail!(
            "{} is {} bytes, not a multiple of the 32-byte descriptor size",
            descriptors.display(),
            bytes.len()
        );
    }
    let descriptors: Vec<wslam_map::BinaryDescriptor> = bytes
        .chunks_exact(32)
        .map(|c| {
            let mut d = [0u8; 32];
            d.copy_from_slice(c);
            wslam_map::BinaryDescriptor(d)
        })
        .collect();
    log::info!(
        "training on {} descriptors, branching {branching}, depth {depth}, seed {seed}",
        descriptors.len()
    );

    let mut rng = wslam_core::DeterministicRng::new("vocab-train", seed);
    let vocab = wslam_map::Vocabulary::train(&descriptors, branching, depth, &mut rng);
    let bytes = vocab.serialize();
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out, &bytes).with_context(|| format!("writing {}", out.display()))?;

    // The provenance file is what makes a vocabulary reproducible, and
    // reproducibility is what makes a recall regression bisectable.
    let provenance = out.with_extension("json");
    std::fs::write(
        &provenance,
        format!(
            "{{\n  \"words\": {},\n  \"branching\": {branching},\n  \"depth\": {depth},\n  \"seed\": {seed},\n  \"descriptors\": {}\n}}\n",
            vocab.word_count(),
            descriptors.len()
        ),
    )?;
    println!(
        "wrote {} ({} words, {:.1} KiB) and {}",
        out.display(),
        vocab.word_count(),
        bytes.len() as f64 / 1024.0,
        provenance.display()
    );
    Ok(())
}
