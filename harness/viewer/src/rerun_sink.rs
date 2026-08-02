//! The rerun-backed [`ViewerSink`].
//!
//! Behind the `rerun-viewer` feature, because `rerun` is a large dependency and
//! `cargo test` should stay fast. Everything logs through [`ViewerSink`] so the
//! call sites in the harness are identical either way.
//!
//! ## Two timelines
//!
//! Every entity is logged on both `frame` (an integer sequence) and `time` (the
//! unified timebase, in seconds). Scrubbing by frame is what you want when
//! chasing a tracking failure; scrubbing by time is what you want when
//! correlating against IMU. Logging both costs nothing and rerun lets you pick.

use rerun::{RecordingStream, RecordingStreamBuilder};
use wslam_core::{Scalar, Se3, Timestamp};

use crate::{uncertainty_ellipsoid, SessionFrame, ViewerSink};

/// Logs a session to rerun.
pub struct RerunSink {
    stream: RecordingStream,
}

impl RerunSink {
    /// Open a recording.
    ///
    /// With a path, writes an `.rrd` for later scrubbing — which is what CI
    /// does (spec.md §6: *"Nightly runs record a rerun session as a build
    /// artifact. When a regression fires, scrub the failing run visually
    /// instead of bisecting from a single ATE number."*). Without one, connects
    /// to a viewer if one is listening.
    pub fn new(
        application_id: &str,
        path: Option<&std::path::Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let builder = RecordingStreamBuilder::new(application_id);
        let stream = match path {
            Some(path) => builder.save(path)?,
            None => builder.connect_grpc()?,
        };
        Ok(RerunSink { stream })
    }

    /// Position both timelines.
    fn set_time(&self, timestamp: Timestamp, frame_index: u64) {
        self.stream.set_time_sequence("frame", frame_index as i64);
        self.stream.set_duration_secs("time", timestamp.seconds());
    }

    fn log_pose(&self, path: &str, pose: &Se3) {
        let t = pose.translation();
        let m = pose.rotation().matrix();
        let _ = self.stream.log(
            path,
            &rerun::Transform3D::from_translation_mat3x3(
                [t.x as f32, t.y as f32, t.z as f32],
                [
                    [m[(0, 0)] as f32, m[(1, 0)] as f32, m[(2, 0)] as f32],
                    [m[(0, 1)] as f32, m[(1, 1)] as f32, m[(2, 1)] as f32],
                    [m[(0, 2)] as f32, m[(1, 2)] as f32, m[(2, 2)] as f32],
                ],
            ),
        );
    }
}

impl ViewerSink for RerunSink {
    fn log_frame(&mut self, frame: &SessionFrame<'_>) {
        self.set_time(frame.timestamp, frame.frame_index);

        if let Some(image) = frame.image {
            let _ = self.stream.log(
                "world/camera/image",
                &rerun::Image::from_l8(image.data().to_vec(), [image.width(), image.height()]),
            );
        }

        if !frame.features.is_empty() {
            let points: Vec<(f32, f32)> = frame
                .features
                .iter()
                .map(|f| (f.px.x as f32, f.px.y as f32))
                .collect();
            let colours: Vec<rerun::Color> = frame
                .features
                .iter()
                .map(|f| {
                    let [r, g, b] = f.state.colour();
                    rerun::Color::from_rgb(r, g, b)
                })
                .collect();
            let _ = self.stream.log(
                "world/camera/image/features",
                &rerun::Points2D::new(points)
                    .with_colors(colours)
                    .with_radii([2.0]),
            );
        }

        if let Some(pose) = frame.pose {
            self.log_pose("world/camera", &pose);
        }
        if let Some(truth) = frame.ground_truth {
            self.log_pose("world/ground_truth", &truth);
            if let Some(pose) = frame.pose {
                // The instantaneous position error, as a time series. Cheap, and
                // it turns "the ATE got worse" into "it got worse *here*".
                let error = (pose.translation() - truth.translation()).norm();
                let _ = self
                    .stream
                    .log("metrics/position_error", &rerun::Scalars::single(error));
            }
        }

        // The covariance ellipsoid. spec.md §8 lists it as required and says
        // why: "it is our differentiator, we should be looking at it daily."
        if let (Some(pose), Some(covariance)) = (frame.pose, frame.covariance) {
            if let Some(ellipsoid) = uncertainty_ellipsoid(&covariance, 0, 2.0) {
                let t = pose.translation();
                let _ = self.stream.log(
                    "world/camera/uncertainty",
                    &rerun::Ellipsoids3D::from_centers_and_half_sizes(
                        [(t.x as f32, t.y as f32, t.z as f32)],
                        [(
                            ellipsoid.semi_axes.x as f32,
                            ellipsoid.semi_axes.y as f32,
                            ellipsoid.semi_axes.z as f32,
                        )],
                    )
                    .with_colors([rerun::Color::from_rgb(251, 191, 36)]),
                );
                let _ = self.stream.log(
                    "metrics/uncertainty_max_sigma",
                    &rerun::Scalars::single(ellipsoid.max_extent() / 2.0),
                );
            }
        }

        if !frame.landmarks.is_empty() {
            let points: Vec<(f32, f32, f32)> = frame
                .landmarks
                .iter()
                .map(|p| (p.x as f32, p.y as f32, p.z as f32))
                .collect();
            let _ = self.stream.log(
                "world/landmarks",
                &rerun::Points3D::new(points)
                    .with_colors([rerun::Color::from_rgb(94, 234, 212)])
                    .with_radii([0.01]),
            );
        }

        for (id, pose) in frame.keyframes {
            self.log_pose(&format!("world/keyframes/{id}"), pose);
        }

        // Pose-graph edges as line strips, coloured by acceptance. Rejected
        // loop candidates are drawn deliberately (spec.md §8) — a graph view
        // that hides them cannot answer "was the threshold too tight?".
        if !frame.edges.is_empty() {
            let mut accepted: Vec<Vec<(f32, f32, f32)>> = Vec::new();
            let mut rejected: Vec<Vec<(f32, f32, f32)>> = Vec::new();
            let lookup = |id: u64| {
                frame
                    .keyframes
                    .iter()
                    .find(|(k, _)| *k == id)
                    .map(|(_, p)| p.translation())
            };
            for edge in frame.edges {
                let (Some(a), Some(b)) = (lookup(edge.from), lookup(edge.to)) else {
                    continue;
                };
                let strip = vec![
                    (a.x as f32, a.y as f32, a.z as f32),
                    (b.x as f32, b.y as f32, b.z as f32),
                ];
                if edge.accepted {
                    accepted.push(strip);
                } else {
                    rejected.push(strip);
                }
            }
            if !accepted.is_empty() {
                let _ = self.stream.log(
                    "world/pose_graph/accepted",
                    &rerun::LineStrips3D::new(accepted)
                        .with_colors([rerun::Color::from_rgb(94, 234, 212)]),
                );
            }
            if !rejected.is_empty() {
                let _ = self.stream.log(
                    "world/pose_graph/rejected",
                    &rerun::LineStrips3D::new(rejected)
                        .with_colors([rerun::Color::from_rgb(248, 113, 113)]),
                );
            }
        }

        if let Some(scale) = frame.scale {
            let _ = self.stream.log(
                "status/scale",
                &rerun::TextDocument::new(format!(
                    "{} · {:.4} ± {:.3}%",
                    scale.source.as_str(),
                    scale.value,
                    scale.relative_stddev_percent()
                )),
            );
            if scale.variance.is_finite() {
                let _ = self.stream.log(
                    "metrics/scale_stddev_percent",
                    &rerun::Scalars::single(scale.relative_stddev_percent()),
                );
            }
        }

        if let Some(state) = frame.state {
            let _ = self.stream.log(
                "status/tracking_state",
                &rerun::TextDocument::new(match state.limited_reason() {
                    Some(reason) => format!("limited · {}", reason.as_str()),
                    None => state.as_str().to_string(),
                }),
            );
        }

        // Per-stage timings, as separate series so the GPU budget is readable
        // at a glance (spec.md §8: "Required to manage the WebGPU budget").
        if let Some(t) = frame.timings {
            for (name, value) in [
                ("upload", t.upload_ms),
                ("pyramid", t.pyramid_ms),
                ("corners", t.corners_ms),
                ("flow", t.flow_ms),
                ("pnp", t.pnp_ms),
                ("total", t.total_ms),
            ] {
                let _ = self
                    .stream
                    .log(format!("timings/{name}"), &rerun::Scalars::single(value));
            }
        }
    }

    fn log_scalar(&mut self, path: &str, timestamp: Timestamp, value: Scalar) {
        // Non-finite values poison a rerun plot's axis range, hiding every
        // other series on the same view. Dropping them is better than that.
        if !value.is_finite() {
            return;
        }
        self.stream.set_duration_secs("time", timestamp.seconds());
        let _ = self.stream.log(path, &rerun::Scalars::single(value));
    }

    fn log_note(&mut self, path: &str, timestamp: Timestamp, text: &str) {
        self.stream.set_duration_secs("time", timestamp.seconds());
        let _ = self.stream.log(path, &rerun::TextLog::new(text));
    }

    fn finish(&mut self) {
        // A failed flush loses the recording, which matters for a nightly CI
        // artifact — but the run's actual results are already computed and
        // reported, so this warns rather than failing the run.
        if let Err(err) = self.stream.flush_blocking() {
            log::warn!("rerun flush failed; the session recording may be truncated: {err}");
        }
    }
}
