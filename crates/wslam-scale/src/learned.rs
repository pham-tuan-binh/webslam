//! Scale from a monocular depth prior — the fallback ruler, and the only one
//! that has to argue for itself.
//!
//! spec.md §2 lists the learned prior at *"several %, domain-correlated"*, and
//! spec.md §5 is blunter still: MDE-VIO (arXiv:2602.11323) *"reports that direct
//! metric depth predictions were insufficient — use them as constraints, never
//! as truth."* That sentence is the whole design of this module. Nothing here
//! ever reads a depth off the model and calls it metres.
//!
//! # What a learned prior can and cannot tell you
//!
//! [`DepthModel`] predicts **affine-invariant inverse depth**: an output `q`
//! that is related to metric inverse depth by an unknown gain and an unknown
//! shift. The visual structure from L3 supplies up-to-scale depth `z`, related
//! to metric depth by the one unknown multiplier `s` this crate exists to find.
//!
//! Write both relations out and the identifiability question answers itself:
//!
//! ```text
//! 1 / (s z_i)  =  a q_i + b          metric inverse depth, two unknowns a, b
//! ```
//!
//! `(a, b, s)` and `(a/c, b/c, c s)` fit the data identically for any `c > 0`.
//! **A purely affine-invariant prediction plus up-to-scale structure contains
//! no metric information whatsoever.** No estimator, however clever, recovers
//! metres from those two inputs alone — the same theorem as spec.md §2, one
//! level up.
//!
//! So the metric content has to be declared, and it is: [`DepthModel::metric_gain`]
//! is the model's own claim about `a`, and it is the *only* number this
//! estimator takes on faith. The shift `b` is refitted from the data every
//! time, because affine invariance means the model's shift is meaningless by
//! construction — which is exactly the "scale **and** shift jointly" fit of
//! *Learned Monocular Depth Priors in Visual-Inertial Initialization*
//! (arXiv:2204.09171, ECCV 2022).
//!
//! # The estimator
//!
//! With `ρ_i = gain · q_i` the model's metric inverse depth, `x_i = 1 / z_i`
//! the up-to-scale inverse depth, and `u = 1 / s`:
//!
//! ```text
//! u x_i - t = ρ_i
//! ```
//!
//! Two unknowns, `n` observations, one ordinary least squares. `u` is the
//! slope, `t` the refitted shift. Three things fall out of it that a
//! read-the-depth-and-divide implementation cannot produce:
//!
//! 1. **A degeneracy test.** The slope is unidentifiable when the `x_i` do not
//!    vary — a fronto-parallel wall at one distance, where scale and shift
//!    trade off exactly. [`LearnedRejection::NoDepthSpread`].
//! 2. **A consistency gate.** The residual of the fit measures whether the
//!    model's *geometry* agrees with the reconstructed geometry up to an
//!    affine transform. When it does not, the model is wrong about this scene
//!    and there is nothing to salvage: [`LearnedRejection::Inconsistent`], and
//!    [`LearnedScale::estimate`] returns `None`. This is MDE-VIO's variance
//!    gate and it is the reason this source is allowed to exist.
//! 3. **A calibrated variance**, combining the fit's own precision with the
//!    declared gain uncertainty — because `s` is exactly inversely
//!    proportional to the gain, so the gain's error passes straight through.
//!
//! # What is deliberately absent
//!
//! No weights, no download, no inference runtime, no new dependency. spec.md §3
//! says `ScaleSource.learned({ model })` *"downloads weights"* — the download
//! belongs to whoever implements [`DepthModel`], behind this seam. The crate
//! compiles the seam and the estimator and nothing else, which is what the
//! `learned-scale` feature gates (spec.md §7).
//!
//! There is no RNG and no wall clock here: the estimate is a deterministic
//! function of the samples and the window.

use crate::ScaleSource;
use std::collections::VecDeque;
use wslam_core::{
    Error, Frame, Result, Scalar, ScaleEstimate, ScaleKind, StateWindow, Timestamp, Vec2,
    WindowSample,
};

/// Affine-invariant inverse depth predicted over one frame.
///
/// Stored at the model's own resolution rather than the frame's, because real
/// monocular models run at a fixed low resolution and upsampling their output
/// to 1280x720 would cost bandwidth to add no information. Sampling is in
/// normalised coordinates so the two resolutions never have to agree.
///
/// The values are in the model's units. They are *not* metres, they are *not*
/// 1/metres, and treating them as either is the mistake this whole module is
/// arranged to prevent.
#[derive(Debug, Clone, PartialEq)]
pub struct InverseDepth {
    width: u32,
    height: u32,
    data: Vec<Scalar>,
}

impl InverseDepth {
    /// Wrap a row-major prediction grid.
    ///
    /// # Errors
    /// [`Error::Config`] if the grid is empty or `data.len() != width * height`.
    pub fn new(width: u32, height: u32, data: Vec<Scalar>) -> Result<Self> {
        let want = (width as usize) * (height as usize);
        if want == 0 {
            return Err(Error::Config("inverse depth grid is empty".into()));
        }
        if data.len() != want {
            return Err(Error::Config(format!(
                "inverse depth grid is {}x{} but carries {} values",
                width,
                height,
                data.len()
            )));
        }
        Ok(InverseDepth {
            width,
            height,
            data,
        })
    }

    /// Grid width.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Grid height.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Row-major values, in the model's units.
    #[must_use]
    pub fn data(&self) -> &[Scalar] {
        &self.data
    }

    /// Bilinear sample at normalised coordinates, both in `[0, 1)` across the
    /// frame. Returns `None` outside that range or at a non-finite value.
    ///
    /// Pixel-centre convention (`align_corners = false`), so a grid at the
    /// frame's own resolution samples its pixels exactly.
    #[must_use]
    pub fn sample_normalized(&self, u: Scalar, v: Scalar) -> Option<Scalar> {
        if !u.is_finite() || !v.is_finite() {
            return None;
        }
        if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
            return None;
        }
        let gx = (u * self.width as Scalar - 0.5).clamp(0.0, (self.width - 1) as Scalar);
        let gy = (v * self.height as Scalar - 0.5).clamp(0.0, (self.height - 1) as Scalar);
        let x0 = gx.floor() as u32;
        let y0 = gy.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = gx - x0 as Scalar;
        let fy = gy - y0 as Scalar;
        let at = |x: u32, y: u32| self.data[(y * self.width + x) as usize];
        let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
        let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
        let value = top * (1.0 - fy) + bottom * fy;
        value.is_finite().then_some(value)
    }
}

/// A monocular depth prior, as a seam rather than an implementation.
///
/// Implementors own the model download and the inference runtime; this crate
/// owns neither, and adds no dependency for either (spec.md §3: the learned
/// source is *"opt-in, downloads weights"* — opt-in at the seam).
///
/// `Send` for the same reason [`ScaleSource`] is: the orchestrator may move a
/// source across the frontend/backend thread split (spec.md §4).
pub trait DepthModel: Send {
    /// Affine-invariant inverse depth over `frame`, in the model's own units.
    ///
    /// `None` when the model declines the frame — too dark, wrong aspect,
    /// still loading. Declining is not an error; the caller keeps feeding
    /// frames.
    fn predict(&mut self, frame: &Frame) -> Option<InverseDepth>;

    /// Gain taking the model's inverse-depth units to metric inverse depth,
    /// in `1/m` per model unit.
    ///
    /// **The only metric information this crate takes from a model**, and the
    /// only one it can take: see the module header for why the shift carries
    /// none and why something has to. A model that has no metric calibration
    /// at all cannot be a ruler, and should not be wired up as one.
    fn metric_gain(&self) -> Scalar;

    /// Relative standard deviation of [`DepthModel::metric_gain`], as a
    /// fraction — the model's honest opinion of its own calibration.
    ///
    /// It propagates straight into the reported scale variance because `s` is
    /// exactly inversely proportional to the gain. The default is the 5% that
    /// spec.md §2 implies for a learned prior (*"several %,
    /// domain-correlated"*, and spec.md §5's *"1% beats every learned prior"*
    /// puts a floor under it).
    fn gain_relative_stddev(&self) -> Scalar {
        0.05
    }
}

/// One up-to-scale structure observation: a triangulated landmark seen at a
/// pixel, with the depth L3 reconstructed for it in window units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructurePoint {
    /// Where it was observed, in frame pixels.
    pub pixel: Vec2,
    /// Depth along the optical axis in up-to-scale units. Multiplying by the
    /// recovered scale yields metres.
    pub depth_units: Scalar,
}

/// A model prediction paired with the up-to-scale structure at the same point.
///
/// This is the row of the least-squares problem, kept as a public type so a
/// caller that runs inference on another thread can hand rows across without
/// this crate ever seeing a [`Frame`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthSample {
    /// Capture time of the frame it came from.
    pub timestamp: Timestamp,
    /// Model output at the point, in the model's units.
    pub predicted: Scalar,
    /// `1 / depth_units` from the visual structure.
    pub inverse_depth_units: Scalar,
}

/// Gates and sizing for the learned solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearnedConfig {
    /// Samples required before answering. Two is the algebraic minimum for a
    /// two-parameter fit and leaves zero degrees of freedom for the residual
    /// gate — which would disable the only check that matters — so the floor
    /// enforced below is four regardless of what is configured here.
    pub min_samples: usize,
    /// Samples retained. Bounded so a scale source cannot grow the frontend's
    /// memory without limit (the discipline spec.md §6 L4 applies to the map).
    pub max_samples: usize,
    /// Landmarks the associated frame's pose solve must have used. A pose
    /// fitted from a handful of points has depths not worth regressing
    /// against.
    pub min_landmarks: usize,
    /// How far a sample may sit from a window pose to be associated with it,
    /// in seconds.
    pub max_time_delta: Scalar,
    /// Smallest coefficient of variation of the up-to-scale inverse depths
    /// that still identifies the slope. Below it, scale and shift trade off
    /// against each other and the answer is whatever the noise says.
    pub min_relative_depth_spread: Scalar,
    /// Residual RMS as a fraction of the predicted inverse-depth spread, above
    /// which the model's geometry is judged inconsistent with the
    /// reconstruction. **The gate**, per MDE-VIO. A release-gate number to be
    /// tuned on rig data (spec.md §6 L5), not a constant to be trusted.
    pub max_residual_ratio: Scalar,
    /// Relative standard deviation above which the estimate is too imprecise
    /// to report at all.
    pub max_relative_stddev: Scalar,
}

impl Default for LearnedConfig {
    fn default() -> Self {
        LearnedConfig {
            // A tracked frame carries hundreds of landmarks; asking for 16 is
            // asking for one frame's worth of usable structure.
            min_samples: 16,
            max_samples: 2048,
            min_landmarks: 20,
            max_time_delta: 0.05,
            min_relative_depth_spread: 0.05,
            max_residual_ratio: 0.10,
            // Wider than the 5% default gain uncertainty, so the fit's own
            // imprecision has room to speak before this fires.
            max_relative_stddev: 0.30,
        }
    }
}

/// Why the learned solve declined to answer.
///
/// Every variant is a named condition under which the prior carries no usable
/// metric information, not a numerical accident. spec.md §1: the library
/// *"never silently guesses scale"*.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LearnedRejection {
    /// Not enough usable samples, after window association.
    InsufficientSamples {
        /// Samples that associated with a window pose.
        have: usize,
        /// Samples required.
        need: usize,
    },
    /// The scene has one depth. Scale and shift are then exchangeable and the
    /// slope is unidentifiable — the learned source's analogue of the static
    /// hold that spec.md §6 Tier 3 requires be *detected*.
    NoDepthSpread {
        /// Coefficient of variation of the up-to-scale inverse depths.
        relative_spread: Scalar,
        /// Threshold it failed.
        threshold: Scalar,
    },
    /// The model predicted a constant, which carries no geometry at all.
    NoPredictedSpread,
    /// The model's geometry does not match the reconstruction up to an affine
    /// transform. spec.md §5: *"use them as constraints, never as truth"* —
    /// this is the constraint failing, and the only honest response is silence.
    Inconsistent {
        /// Residual RMS over the predicted inverse-depth spread.
        residual_ratio: Scalar,
        /// Threshold it failed.
        threshold: Scalar,
    },
    /// The fit produced a non-positive or non-finite multiplier, or the model
    /// declared a gain that is not a positive finite number.
    Degenerate,
    /// A solution exists but is too imprecise to be worth reporting.
    Imprecise {
        /// Relative standard deviation of the recovered scale.
        relative_stddev: Scalar,
        /// Threshold it failed.
        threshold: Scalar,
    },
}

impl std::fmt::Display for LearnedRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LearnedRejection::InsufficientSamples { have, need } => {
                write!(f, "insufficient depth samples: {have} of {need}")
            }
            LearnedRejection::NoDepthSpread {
                relative_spread,
                threshold,
            } => write!(
                f,
                "depth spread {relative_spread:.3e} below {threshold:.3e}: \
                 scale and shift are exchangeable"
            ),
            LearnedRejection::NoPredictedSpread => {
                write!(f, "model predicted a constant inverse depth")
            }
            LearnedRejection::Inconsistent {
                residual_ratio,
                threshold,
            } => write!(
                f,
                "depth prediction inconsistent with structure: residual ratio \
                 {:.1}% exceeds {:.1}%",
                100.0 * residual_ratio,
                100.0 * threshold
            ),
            LearnedRejection::Degenerate => write!(f, "degenerate solution"),
            LearnedRejection::Imprecise {
                relative_stddev,
                threshold,
            } => write!(
                f,
                "scale stddev {:.1}% exceeds {:.1}%",
                100.0 * relative_stddev,
                100.0 * threshold
            ),
        }
    }
}

/// Everything the joint scale-and-shift fit recovers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearnedSolution {
    /// Multiplier taking up-to-scale units to metres.
    pub scale: Scalar,
    /// Variance of `scale`: the fit's own precision plus the declared gain
    /// uncertainty, which passes through undiminished.
    pub variance: Scalar,
    /// The refitted affine shift, in metric inverse depth (`1/m`). Reported
    /// because a large one means the model's zero point is far off, which is
    /// diagnostic even when the scale is fine.
    pub shift: Scalar,
    /// Residual RMS of the fit, in `1/m`.
    pub residual_rms: Scalar,
    /// Residual RMS over the predicted inverse-depth spread — the number the
    /// consistency gate is applied to.
    pub residual_ratio: Scalar,
    /// Coefficient of variation of the up-to-scale inverse depths, i.e. how
    /// much depth range the fit had to work with.
    pub relative_depth_spread: Scalar,
    /// Samples the fit consumed.
    pub samples: usize,
}

impl LearnedSolution {
    /// Scale error as a fractional standard deviation.
    #[must_use]
    pub fn relative_stddev(&self) -> Scalar {
        if self.scale.abs() < 1e-12 {
            Scalar::INFINITY
        } else {
            self.variance.sqrt() / self.scale.abs()
        }
    }
}

/// Fit metric scale and the affine shift jointly against up-to-scale structure.
///
/// `gain` is [`DepthModel::metric_gain`] and `gain_relative_stddev` its
/// uncertainty. Public because it is the natural place to check the estimator:
/// feed it samples generated from a known multiplier and it must return that
/// multiplier, which is a stronger statement than anything the wrapper can
/// make.
///
/// # Errors
/// One of the named [`LearnedRejection`] cases.
pub fn fit_scale_and_shift(
    samples: &[DepthSample],
    gain: Scalar,
    gain_relative_stddev: Scalar,
    config: &LearnedConfig,
) -> std::result::Result<LearnedSolution, LearnedRejection> {
    // Four, not two: the residual gate needs degrees of freedom to exist, and
    // a two-point fit is exact by construction and would pass it always.
    let need = config.min_samples.max(4);
    let n = samples.len();
    if n < need {
        return Err(LearnedRejection::InsufficientSamples { have: n, need });
    }
    if !(gain.is_finite() && gain > 0.0) {
        return Err(LearnedRejection::Degenerate);
    }

    let nf = n as Scalar;
    // x: up-to-scale inverse depth (1/unit). rho: the model's metric inverse
    // depth (1/m), which is a claim, not a measurement.
    let x: Vec<Scalar> = samples.iter().map(|s| s.inverse_depth_units).collect();
    let rho: Vec<Scalar> = samples.iter().map(|s| gain * s.predicted).collect();
    if x.iter().chain(rho.iter()).any(|v| !v.is_finite()) {
        return Err(LearnedRejection::Degenerate);
    }

    let mean_x = x.iter().sum::<Scalar>() / nf;
    let mean_rho = rho.iter().sum::<Scalar>() / nf;
    let sxx: Scalar = x.iter().map(|v| (v - mean_x).powi(2)).sum();
    let srr: Scalar = rho.iter().map(|v| (v - mean_rho).powi(2)).sum();

    // Identifiability, before any solving: with no spread in x the slope and
    // the shift are the same parameter wearing two hats.
    let spread = if mean_x.abs() > 1e-18 {
        (sxx / nf).sqrt() / mean_x.abs()
    } else {
        0.0
    };
    if !(spread.is_finite() && spread >= config.min_relative_depth_spread) {
        return Err(LearnedRejection::NoDepthSpread {
            relative_spread: spread,
            threshold: config.min_relative_depth_spread,
        });
    }
    if srr <= 0.0 || !srr.is_finite() {
        return Err(LearnedRejection::NoPredictedSpread);
    }

    // u x - t = rho: ordinary least squares, u the slope and -t the intercept.
    let sxr: Scalar = x
        .iter()
        .zip(rho.iter())
        .map(|(a, b)| (a - mean_x) * (b - mean_rho))
        .sum();
    let u = sxr / sxx;
    let shift = u * mean_x - mean_rho;

    let sse: Scalar = x
        .iter()
        .zip(rho.iter())
        .map(|(a, b)| (u * a - shift - b).powi(2))
        .sum();
    let residual_rms = (sse / nf).sqrt();
    // Normalised by the spread of the prediction, because a constant offset is
    // absorbed by the shift and an absolute residual in 1/m means nothing
    // without knowing how far away the scene is.
    let residual_ratio = residual_rms / (srr / nf).sqrt();
    if residual_ratio > config.max_residual_ratio {
        return Err(LearnedRejection::Inconsistent {
            residual_ratio,
            threshold: config.max_residual_ratio,
        });
    }

    if !(u.is_finite() && u > 0.0) {
        // A non-positive slope means the model's depth ordering is inverted
        // relative to the reconstruction. That is not a scale, at any value.
        return Err(LearnedRejection::Degenerate);
    }
    let scale = 1.0 / u;

    // var(slope) = sigma^2 / Sxx, then the delta method through s = 1/u.
    let dof = n.saturating_sub(2).max(1) as Scalar;
    let var_u = (sse / dof) / sxx;
    let var_fit = var_u / u.powi(4);
    // s is exactly inversely proportional to the gain, so the gain's relative
    // error is the scale's relative error, added in quadrature.
    let var_gain = (gain_relative_stddev * scale).powi(2);
    let variance = var_fit + var_gain;
    if !(variance.is_finite() && variance >= 0.0) {
        return Err(LearnedRejection::Degenerate);
    }

    let solution = LearnedSolution {
        scale,
        variance,
        shift,
        residual_rms,
        residual_ratio,
        relative_depth_spread: spread,
        samples: n,
    };
    let rel = solution.relative_stddev();
    if rel > config.max_relative_stddev {
        return Err(LearnedRejection::Imprecise {
            relative_stddev: rel,
            threshold: config.max_relative_stddev,
        });
    }
    Ok(solution)
}

/// Metric scale from a monocular depth prior. Opt-in; feature `learned-scale`.
pub struct LearnedScale {
    model: Box<dyn DepthModel>,
    config: LearnedConfig,
    samples: VecDeque<DepthSample>,
    last_solution: Option<LearnedSolution>,
    last_rejection: Option<LearnedRejection>,
}

impl std::fmt::Debug for LearnedScale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The model is a trait object with no Debug bound — requiring one would
        // push a derive onto every implementor for the sake of a log line.
        f.debug_struct("LearnedScale")
            .field("metric_gain", &self.model.metric_gain())
            .field("config", &self.config)
            .field("samples", &self.samples.len())
            .field("last_solution", &self.last_solution)
            .field("last_rejection", &self.last_rejection)
            .finish()
    }
}

impl LearnedScale {
    /// Wrap a depth model. Infallible: a model that turns out to have declared
    /// a nonsensical gain is rejected at estimate time with
    /// [`LearnedRejection::Degenerate`] rather than at construction, so a
    /// model whose calibration only becomes known after its weights load is
    /// still usable.
    #[must_use]
    pub fn new(model: Box<dyn DepthModel>) -> Self {
        LearnedScale {
            model,
            config: LearnedConfig::default(),
            samples: VecDeque::new(),
            last_solution: None,
            last_rejection: None,
        }
    }

    /// Override the gates.
    #[must_use]
    pub fn with_config(mut self, config: LearnedConfig) -> Self {
        self.config = config;
        self
    }

    /// The configuration in force.
    #[must_use]
    pub fn config(&self) -> &LearnedConfig {
        &self.config
    }

    /// Run the model on a frame and pair its output with the up-to-scale
    /// structure observed in that frame.
    ///
    /// Returns how many usable samples were added. Points whose pixel falls
    /// outside the frame, or whose depth is not a positive finite number, are
    /// dropped rather than clamped — a clamped depth is a fabricated
    /// constraint.
    pub fn observe(&mut self, frame: &Frame, points: &[StructurePoint]) -> usize {
        let (w, h) = (frame.image.width(), frame.image.height());
        if w == 0 || h == 0 || points.is_empty() {
            return 0;
        }
        let Some(prediction) = self.model.predict(frame) else {
            log::debug!("learned: model declined frame {}", frame.id);
            return 0;
        };

        let mut added = 0;
        for p in points {
            if !(p.depth_units.is_finite() && p.depth_units > 0.0) {
                continue;
            }
            let u = (p.pixel.x + 0.5) / w as Scalar;
            let v = (p.pixel.y + 0.5) / h as Scalar;
            let Some(predicted) = prediction.sample_normalized(u, v) else {
                continue;
            };
            self.push_sample(DepthSample {
                timestamp: frame.timestamp,
                predicted,
                inverse_depth_units: 1.0 / p.depth_units,
            });
            added += 1;
        }
        added
    }

    /// Record a sample produced elsewhere — the orchestrator may run inference
    /// on the GPU thread and hand the rows across.
    pub fn push_sample(&mut self, sample: DepthSample) {
        if self.samples.len() >= self.config.max_samples.max(1) {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    /// Samples held, oldest first.
    pub fn samples(&self) -> impl Iterator<Item = &DepthSample> + '_ {
        self.samples.iter()
    }

    /// The most recent successful fit, with its residual and depth spread.
    #[must_use]
    pub fn last_solution(&self) -> Option<&LearnedSolution> {
        self.last_solution.as_ref()
    }

    /// Why the most recent call declined. The consistency gate is only useful
    /// if callers can see it fire.
    #[must_use]
    pub fn last_rejection(&self) -> Option<LearnedRejection> {
        self.last_rejection
    }
}

/// Nearest window pose to `t`, within `max_delta` seconds.
fn nearest_pose(window: &StateWindow, t: Timestamp, max_delta: Scalar) -> Option<&WindowSample> {
    window
        .poses()
        .map(|p| (p.timestamp.since(t).abs(), p))
        .filter(|(dt, _)| *dt <= max_delta)
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, p)| p)
}

impl ScaleSource for LearnedScale {
    fn kind(&self) -> ScaleKind {
        ScaleKind::Learned
    }

    fn estimate(&mut self, window: &StateWindow) -> Option<ScaleEstimate> {
        // Association against the window is what stops a stale sample from a
        // previous scene anchoring the current one, and it is where the
        // window's own quality gate (landmark_count) is applied.
        let usable: Vec<DepthSample> = self
            .samples
            .iter()
            .filter(|s| {
                nearest_pose(window, s.timestamp, self.config.max_time_delta)
                    .is_some_and(|p| p.landmark_count >= self.config.min_landmarks)
            })
            .copied()
            .collect();

        match fit_scale_and_shift(
            &usable,
            self.model.metric_gain(),
            self.model.gain_relative_stddev(),
            &self.config,
        ) {
            Ok(solution) => {
                let estimate =
                    ScaleEstimate::metric(ScaleKind::Learned, solution.scale, solution.variance);
                self.last_solution = Some(solution);
                self.last_rejection = None;
                Some(estimate)
            }
            Err(rejection) => {
                log::debug!("learned scale declined: {rejection}");
                self.last_rejection = Some(rejection);
                None
            }
        }
    }

    fn reset(&mut self) {
        self.samples.clear();
        self.last_solution = None;
        self.last_rejection = None;
    }
}

/// A deterministic [`DepthModel`] over an analytic metric depth field.
///
/// Test and harness scaffolding: it lets a Tier-1 test state a scene's true
/// geometry in metres, hand the estimator only the affine-invariant view of it
/// that a real model would produce, and demand the metres back. There is no
/// RNG in it and no learned anything — the "model" is a closed-form function,
/// which is exactly what makes the assertions about the estimator rather than
/// about a network.
pub struct SyntheticDepthModel {
    width: u32,
    height: u32,
    /// Gain the scene is *encoded* with.
    gain: Scalar,
    /// Gain the model *declares*. Equal to `gain` for a correctly calibrated
    /// model; separate so a miscalibration can be injected without altering
    /// the geometry the model reports.
    declared_gain: Scalar,
    shift: Scalar,
    gain_relative_stddev: Scalar,
    #[allow(clippy::type_complexity)]
    depth_m: Box<dyn Fn(Scalar, Scalar) -> Scalar + Send>,
}

impl std::fmt::Debug for SyntheticDepthModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyntheticDepthModel")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("gain", &self.gain)
            .field("declared_gain", &self.declared_gain)
            .field("shift", &self.shift)
            .finish_non_exhaustive()
    }
}

impl SyntheticDepthModel {
    /// A model that sees `depth_m(u, v)` metres at normalised image coordinate
    /// `(u, v)`, and reports it as `q = (1/Z - shift) / gain`.
    ///
    /// `gain` is what [`DepthModel::metric_gain`] returns, so a model built
    /// this way is *correctly calibrated*; `shift` is the model's arbitrary
    /// affine offset, which the estimator must refit and must not be affected
    /// by. Pass a different `gain` to the estimator than the one used to
    /// encode, and the recovered scale moves by exactly that ratio — which is
    /// the honest statement of where the metres come from.
    #[must_use]
    pub fn new(
        width: u32,
        height: u32,
        gain: Scalar,
        shift: Scalar,
        depth_m: Box<dyn Fn(Scalar, Scalar) -> Scalar + Send>,
    ) -> Self {
        SyntheticDepthModel {
            width,
            height,
            gain,
            declared_gain: gain,
            shift,
            gain_relative_stddev: 0.05,
            depth_m,
        }
    }

    /// Override the declared gain uncertainty.
    #[must_use]
    pub fn with_gain_relative_stddev(mut self, sigma: Scalar) -> Self {
        self.gain_relative_stddev = sigma;
        self
    }

    /// Miscalibrate the declared gain by `factor` without touching the scene
    /// the model sees. Used to show that the gain is the single channel the
    /// metres travel down.
    #[must_use]
    pub fn with_gain_scaled_by(mut self, factor: Scalar) -> Self {
        self.declared_gain = self.gain * factor;
        self
    }
}

impl DepthModel for SyntheticDepthModel {
    fn predict(&mut self, _frame: &Frame) -> Option<InverseDepth> {
        let mut data = Vec::with_capacity((self.width as usize) * (self.height as usize));
        for y in 0..self.height {
            let v = (y as Scalar + 0.5) / self.height as Scalar;
            for x in 0..self.width {
                let u = (x as Scalar + 0.5) / self.width as Scalar;
                let z = (self.depth_m)(u, v);
                data.push((1.0 / z - self.shift) / self.gain);
            }
        }
        InverseDepth::new(self.width, self.height, data).ok()
    }

    fn metric_gain(&self) -> Scalar {
        self.declared_gain
    }

    fn gain_relative_stddev(&self) -> Scalar {
        self.gain_relative_stddev
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use wslam_core::{FrameId, GrayImage, Se3, So3, Vec3};

    const W: u32 = 64;
    const H: u32 = 48;

    fn frame(t: Scalar) -> Frame {
        Frame::new(FrameId(0), Timestamp::from_seconds(t), GrayImage::new(W, H))
    }

    /// A window with one well-constrained pose per supplied timestamp.
    fn window_at(times: &[Scalar], landmarks: usize) -> StateWindow {
        let mut w = StateWindow::with_default_capacity();
        for (i, t) in times.iter().enumerate() {
            w.push_pose(WindowSample {
                timestamp: Timestamp::from_seconds(*t),
                pose: Se3::new(So3::identity(), Vec3::new(i as Scalar * 0.01, 0.0, 0.0)),
                landmark_count: landmarks,
            });
        }
        w
    }

    /// The scene: a floor slanting away from the camera, 1.0 m at the top of
    /// the frame to 4.0 m at the bottom.
    fn slanted_floor(_u: Scalar, v: Scalar) -> Scalar {
        1.0 + 3.0 * v
    }

    /// Structure points on the same scene, in up-to-scale units: the tracker
    /// reconstructed `Z / scale`, because it does not know `scale`.
    ///
    /// The rows are placed on grid-cell centres so the model's prediction is
    /// read back without interpolation. Bilinear interpolation of a
    /// *nonlinear* depth field is not the field, and a tenth of a percent of
    /// resampling error would show up as scale error and muddy assertions that
    /// are about the estimator, not about the sampler.
    fn structure(scale: Scalar, rows: u32) -> Vec<StructurePoint> {
        let stride = H / rows;
        assert!(
            stride >= 1 && rows * stride <= H,
            "rows must divide the frame height"
        );
        (0..rows)
            .map(|i| {
                let py = (i * stride) as Scalar;
                let v = (py + 0.5) / H as Scalar;
                StructurePoint {
                    pixel: Vec2::new((W / 2) as Scalar, py),
                    depth_units: slanted_floor(0.5, v) / scale,
                }
            })
            .collect()
    }

    #[test]
    fn inverse_depth_rejects_a_mismatched_grid() {
        assert!(InverseDepth::new(4, 4, vec![0.0; 15]).is_err());
        assert!(InverseDepth::new(0, 4, Vec::new()).is_err());
        assert!(InverseDepth::new(4, 4, vec![0.0; 16]).is_ok());
    }

    #[test]
    fn inverse_depth_samples_its_own_pixels_exactly() {
        // Pixel-centre convention: a grid at frame resolution must round-trip.
        let data: Vec<Scalar> = (0..12).map(|i| i as Scalar).collect();
        let d = InverseDepth::new(4, 3, data).unwrap();
        for y in 0..3u32 {
            for x in 0..4u32 {
                let u = (x as Scalar + 0.5) / 4.0;
                let v = (y as Scalar + 0.5) / 3.0;
                assert_relative_eq!(
                    d.sample_normalized(u, v).unwrap(),
                    (y * 4 + x) as Scalar,
                    epsilon = 1e-12
                );
            }
        }
        assert!(d.sample_normalized(-0.1, 0.5).is_none());
        assert!(d.sample_normalized(0.5, 1.0).is_none());
    }

    /// The headline: a model that is right about the scene, seen only through
    /// its affine-invariant output, yields the metric multiplier back.
    #[test]
    fn recovers_a_known_scale_from_a_consistent_model() {
        let truth = 0.37;
        let mut source = LearnedScale::new(Box::new(SyntheticDepthModel::new(
            W,
            H,
            0.8,  // gain: model units -> 1/m
            0.13, // the model's arbitrary affine shift
            Box::new(slanted_floor),
        )));

        let f = frame(1.0);
        let points = structure(truth, 24);
        assert_eq!(source.observe(&f, &points), 24);

        let window = window_at(&[1.0], 60);
        let estimate = source
            .estimate(&window)
            .expect("consistent model must answer");

        assert_eq!(estimate.source, ScaleKind::Learned);
        // Noise-free synthetic data: the multiplier is exact, not approximate.
        assert_relative_eq!(estimate.value, truth, epsilon = 1e-9);

        let solution = source.last_solution().copied().unwrap();
        assert!(
            solution.residual_ratio < 1e-9,
            "consistent model must fit exactly, got {}",
            solution.residual_ratio
        );
        // With a perfect fit the only uncertainty left is the model's declared
        // gain error, and it must survive intact rather than being polished
        // away by a tight residual.
        assert_relative_eq!(crate::relative_stddev(&estimate), 0.05, epsilon = 1e-6);
        // The model's shift is refitted, so it comes back as declared.
        assert_relative_eq!(solution.shift, 0.13, epsilon = 1e-9);
    }

    /// The gate, and the reason this source is allowed to exist: a model whose
    /// geometry disagrees with the reconstruction must produce silence, not a
    /// number.
    #[test]
    fn declines_when_the_model_disagrees_with_the_structure() {
        let truth = 0.37;
        // The structure is a floor slanting monotonically away. The model sees
        // a V — nearest at the middle of the frame, far at both edges. No
        // affine transform maps one onto the other, which is exactly the
        // "prediction inconsistent with the visual structure" case.
        let mut source = LearnedScale::new(Box::new(SyntheticDepthModel::new(
            W,
            H,
            0.8,
            0.13,
            Box::new(|_u, v| 1.0 + 6.0 * (v - 0.5).abs()),
        )));

        let f = frame(1.0);
        let points = structure(truth, 24);
        assert_eq!(source.observe(&f, &points), 24);

        let window = window_at(&[1.0], 60);
        assert!(
            source.estimate(&window).is_none(),
            "an inconsistent prior must decline, not guess"
        );
        match source.last_rejection() {
            Some(LearnedRejection::Inconsistent { residual_ratio, .. }) => {
                assert!(residual_ratio > 0.1, "ratio {residual_ratio}");
            }
            other => panic!("expected Inconsistent, got {other:?}"),
        }
    }

    /// A fronto-parallel wall: the model can be perfectly right and the scale
    /// is still unidentifiable, because scale and shift are the same parameter
    /// when every point is at one depth.
    #[test]
    fn declines_on_a_single_depth_scene_where_scale_and_shift_are_exchangeable() {
        let truth = 0.37;
        let mut source = LearnedScale::new(Box::new(SyntheticDepthModel::new(
            W,
            H,
            0.8,
            0.0,
            Box::new(|_u, _v| 2.5),
        )));
        let f = frame(1.0);
        let points: Vec<StructurePoint> = (0..24)
            .map(|i| StructurePoint {
                pixel: Vec2::new((W / 2) as Scalar, i as Scalar),
                depth_units: 2.5 / truth,
            })
            .collect();
        source.observe(&f, &points);

        let window = window_at(&[1.0], 60);
        assert!(source.estimate(&window).is_none());
        assert!(matches!(
            source.last_rejection(),
            Some(LearnedRejection::NoDepthSpread { .. })
        ));
    }

    /// The model's shift carries no information, by definition of affine
    /// invariance — so changing it must not move the answer by one bit.
    #[test]
    fn the_recovered_scale_is_invariant_to_the_models_shift() {
        let truth = 0.37;
        let window = window_at(&[1.0], 60);
        let mut values = Vec::new();
        for shift in [-2.0, 0.0, 0.19, 7.5] {
            let mut source = LearnedScale::new(Box::new(SyntheticDepthModel::new(
                W,
                H,
                0.8,
                shift,
                Box::new(slanted_floor),
            )));
            source.observe(&frame(1.0), &structure(truth, 24));
            values.push(source.estimate(&window).unwrap().value);
        }
        for v in &values {
            assert_relative_eq!(*v, truth, epsilon = 1e-9);
        }
    }

    /// ...and the gain carries all of it. Doubling the model's declared gain
    /// halves the recovered scale, exactly. This is the module's central
    /// honesty claim stated as an assertion: one number of the model's is
    /// taken on faith, and this is which one.
    #[test]
    fn the_recovered_scale_is_exactly_inverse_in_the_declared_gain() {
        let truth = 0.37;
        let window = window_at(&[1.0], 60);
        let mut recovered = Vec::new();
        for factor in [0.5, 1.0, 2.0] {
            let model = SyntheticDepthModel::new(W, H, 0.8, 0.13, Box::new(slanted_floor))
                .with_gain_scaled_by(factor);
            let mut source = LearnedScale::new(Box::new(model));
            source.observe(&frame(1.0), &structure(truth, 24));
            recovered.push(source.estimate(&window).unwrap().value);
        }
        assert_relative_eq!(recovered[0], truth / 0.5, epsilon = 1e-9);
        assert_relative_eq!(recovered[1], truth, epsilon = 1e-9);
        assert_relative_eq!(recovered[2], truth / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn samples_that_do_not_associate_with_a_window_pose_are_not_used() {
        let truth = 0.37;
        let mut source = LearnedScale::new(Box::new(SyntheticDepthModel::new(
            W,
            H,
            0.8,
            0.13,
            Box::new(slanted_floor),
        )));
        source.observe(&frame(1.0), &structure(truth, 24));

        // Window a full second away: the samples describe a different moment.
        let far = window_at(&[2.0], 60);
        assert!(source.estimate(&far).is_none());
        assert!(matches!(
            source.last_rejection(),
            Some(LearnedRejection::InsufficientSamples { have: 0, .. })
        ));

        // Same samples, a pose fitted from too little structure to trust.
        let thin = window_at(&[1.0], 3);
        assert!(source.estimate(&thin).is_none());

        // And with a good pose at the right time, the same samples answer.
        assert!(source.estimate(&window_at(&[1.0], 60)).is_some());
    }

    #[test]
    fn a_model_that_declines_the_frame_adds_nothing() {
        struct Silent;
        impl DepthModel for Silent {
            fn predict(&mut self, _frame: &Frame) -> Option<InverseDepth> {
                None
            }
            fn metric_gain(&self) -> Scalar {
                1.0
            }
        }
        let mut source = LearnedScale::new(Box::new(Silent));
        assert_eq!(source.observe(&frame(1.0), &structure(0.37, 24)), 0);
        assert!(source.estimate(&window_at(&[1.0], 60)).is_none());
    }

    #[test]
    fn a_model_with_a_nonsense_gain_is_degenerate_not_a_panic() {
        let mut source = LearnedScale::new(Box::new(SyntheticDepthModel::new(
            W,
            H,
            0.8,
            0.13,
            Box::new(slanted_floor),
        )));
        source.observe(&frame(1.0), &structure(0.37, 24));
        let samples: Vec<DepthSample> = source.samples().copied().collect();
        for bad in [0.0, -1.0, Scalar::NAN, Scalar::INFINITY] {
            assert_eq!(
                fit_scale_and_shift(&samples, bad, 0.05, &LearnedConfig::default()),
                Err(LearnedRejection::Degenerate)
            );
        }
    }

    #[test]
    fn an_inverted_depth_ordering_is_degenerate_rather_than_a_negative_scale() {
        // A model whose depth ordering runs backwards fits an affine line
        // perfectly — with a negative slope. A negative multiplier is not a
        // scale, so the residual gate cannot be the only check.
        let truth = 0.37;
        let mut source = LearnedScale::new(Box::new(SyntheticDepthModel::new(
            W,
            H,
            0.8,
            0.13,
            // Metric inverse depth 1.5 - 1/(1+3v), which is exactly
            // `1.5 - x/scale` in the structure's own inverse depth: a perfect
            // affine fit with a NEGATIVE slope.
            Box::new(|_u, v| 1.0 / (1.5 - 1.0 / (1.0 + 3.0 * v))),
        )));
        source.observe(&frame(1.0), &structure(truth, 24));
        assert!(source.estimate(&window_at(&[1.0], 60)).is_none());
        assert_eq!(source.last_rejection(), Some(LearnedRejection::Degenerate));
    }

    #[test]
    fn reset_drops_every_sample_so_a_stale_scene_cannot_anchor_a_new_one() {
        let mut source = LearnedScale::new(Box::new(SyntheticDepthModel::new(
            W,
            H,
            0.8,
            0.13,
            Box::new(slanted_floor),
        )));
        source.observe(&frame(1.0), &structure(0.37, 24));
        let window = window_at(&[1.0], 60);
        assert!(source.estimate(&window).is_some());

        source.reset();
        assert_eq!(source.samples().count(), 0);
        assert!(source.last_solution().is_none());
        assert!(source.last_rejection().is_none());
        assert!(source.estimate(&window).is_none());
        assert_eq!(source.kind(), ScaleKind::Learned);
    }

    #[test]
    fn the_sample_buffer_is_bounded() {
        let mut source = LearnedScale::new(Box::new(SyntheticDepthModel::new(
            W,
            H,
            0.8,
            0.13,
            Box::new(slanted_floor),
        )))
        .with_config(LearnedConfig {
            max_samples: 10,
            ..LearnedConfig::default()
        });
        for _ in 0..20 {
            source.observe(&frame(1.0), &structure(0.37, 24));
        }
        assert_eq!(source.samples().count(), 10);
    }

    #[test]
    fn a_two_point_fit_is_refused_because_it_would_pass_the_gate_vacuously() {
        // Two points determine a line exactly, so the residual is zero
        // whatever the model said. Answering there would report a confident
        // number derived from a check that could not fail.
        let samples: Vec<DepthSample> = [(1.0, 0.5), (2.0, 1.0)]
            .iter()
            .map(|&(x, q)| DepthSample {
                timestamp: Timestamp::ZERO,
                predicted: q,
                inverse_depth_units: x,
            })
            .collect();
        let config = LearnedConfig {
            min_samples: 2,
            ..LearnedConfig::default()
        };
        assert_eq!(
            fit_scale_and_shift(&samples, 1.0, 0.05, &config),
            Err(LearnedRejection::InsufficientSamples { have: 2, need: 4 })
        );
    }

    #[test]
    fn every_rejection_renders() {
        let cases = [
            LearnedRejection::InsufficientSamples { have: 1, need: 4 },
            LearnedRejection::NoDepthSpread {
                relative_spread: 0.0,
                threshold: 0.05,
            },
            LearnedRejection::NoPredictedSpread,
            LearnedRejection::Inconsistent {
                residual_ratio: 0.4,
                threshold: 0.1,
            },
            LearnedRejection::Degenerate,
            LearnedRejection::Imprecise {
                relative_stddev: 0.5,
                threshold: 0.3,
            },
        ];
        for c in cases {
            assert!(!c.to_string().is_empty());
        }
    }
}
