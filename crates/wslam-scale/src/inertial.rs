//! Closed-form visual-inertial scale, and the gates that stop it lying.
//!
//! spec.md §5 traces this lineage: Martinelli and Dong-Si & Mourikis for the
//! closed form; Mur-Artal & Tardós for map reuse; Campos et al. for *"consistent
//! init in under 2 s at 5% scale error, converging to 1% after 10 s"*; and
//! ORB-SLAM3's inertial-only MAP over an up-to-scale visual trajectory
//! recovering scale, gravity, velocities and biases before joint VI bundle
//! adjustment. spec.md §5 also calls this our primary source: *"1% beats every
//! learned prior, with zero model download."*
//!
//! # The estimator
//!
//! The window gives up-to-scale camera poses `(R_i, p̄_i)`; the metric position
//! is `s p̄_i` for the unknown multiplier `s`. Preintegrating the IMU between
//! consecutive poses gives `Δp` and `Δv` in the body frame of the earlier pose,
//! and rigid-body kinematics ties them together:
//!
//! ```text
//! s (p̄_j - p̄_i) - Δt v_i - ½ Δt² g = R_i Δp - (R_j - R_i) t_cb
//! v_j - v_i - Δt g                  = R_i Δv
//! ```
//!
//! Every unknown — `s`, gravity `g`, and one velocity per pose — enters
//! linearly, so the whole thing is one least-squares solve. Gravity is kept as
//! a free 3-vector first and then clamped to its known magnitude for a second
//! pass, which is ORB-SLAM3's two-stage structure and buys a useful fraction of
//! a percent.
//!
//! # Why the gates matter more than the solve
//!
//! spec.md §2: *"metric scale becomes unobservable whenever translational
//! acceleration vanishes — which is exactly how people hold phones."* The
//! solve above will happily return a number in that regime; it will just be
//! wrong. Every rejection in [`InertialRejection`] is a named degenerate case,
//! and spec.md §6 Tier 3 requires the static hold to be *"detected rather than
//! silently wrong"*.
//!
//! There is no RNG here and no wall clock: the estimator is a deterministic
//! function of the window's contents.

use crate::ScaleSource;
use nalgebra::{DMatrix, DVector};
// Only the tier-2 build refuses to construct an `InertialScale`, so the error
// type is only named there.
#[cfg(not(feature = "tight-vi"))]
use wslam_core::Error;
use wslam_core::{
    imu::GRAVITY, ImuSample, Mat3, Result, Scalar, ScaleEstimate, ScaleKind, Se3, So3, StateWindow,
    Timestamp, Vec3,
};

/// Configuration for the inertial solve and its gates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InertialConfig {
    /// Poses required. Four is the algebraic minimum — `6(n-1)` equations
    /// against `4 + 3n` unknowns first balances at `n = 4` — but a real window
    /// should carry a second or two of history.
    pub min_poses: usize,
    /// Inertial samples required inside each pose interval. Below two there is
    /// nothing to integrate.
    pub min_imu_per_interval: usize,
    /// Mean translational excitation below which scale is not observable, in
    /// m/s². Roughly the level at which two seconds of double integration
    /// produces less displacement than the accelerometer's own noise.
    pub min_excitation: Scalar,
    /// Relative magnitude of the visual translation below which the scale
    /// column of the design matrix is numerically absent — pure rotation.
    pub min_scale_column_ratio: Scalar,
    /// Smallest ratio of least to greatest singular value the design matrix may
    /// have before the solve is called rank-deficient.
    pub min_condition: Scalar,
    /// Known gravity magnitude, m/s².
    pub gravity_magnitude: Scalar,
    /// How far the recovered gravity magnitude may stray before the solve is
    /// disbelieved. A free-gravity solve that does not land near 9.81 has
    /// fitted noise, and that check costs nothing.
    pub gravity_tolerance: Scalar,
    /// Relative standard deviation above which we decline to answer at all.
    /// This is the scale-free observability gate: it fires whenever the data
    /// cannot pin `s` down, whatever the reason.
    pub max_relative_stddev: Scalar,
    /// Accelerometer noise standard deviation, m/s², used to weight the
    /// position and velocity rows against each other.
    pub accel_noise: Scalar,
    /// Re-solve with gravity clamped to [`InertialConfig::gravity_magnitude`]
    /// after the free solve (ORB-SLAM3's second stage).
    pub refine_with_fixed_gravity: bool,
    /// Body-to-camera transform: `x_camera = R * x_body + t`. Identity is the
    /// browser's situation — spec.md §2 notes the browser gives *"one stream,
    /// no extrinsics"* — but the handheld lever arm of spec.md §5 (rotation
    /// about the wrist, ~20 cm from the optical centre) is exactly this term,
    /// so it is modelled rather than assumed away.
    pub t_camera_body: Se3,
}

impl Default for InertialConfig {
    fn default() -> Self {
        InertialConfig {
            min_poses: 8,
            min_imu_per_interval: 2,
            min_excitation: 0.35,
            min_scale_column_ratio: 1e-6,
            min_condition: 1e-9,
            gravity_magnitude: GRAVITY,
            gravity_tolerance: 2.5,
            max_relative_stddev: 0.20,
            accel_noise: 0.05,
            refine_with_fixed_gravity: true,
            t_camera_body: Se3::identity(),
        }
    }
}

/// Why the solve declined to answer.
///
/// spec.md §6 L5 requires the excitation dependence to be *reported*, not
/// merely handled; this enum is that report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InertialRejection {
    /// Not enough poses in the window yet.
    InsufficientPoses {
        /// Poses present.
        have: usize,
        /// Poses required.
        need: usize,
    },
    /// A pose interval had too few inertial samples to preintegrate.
    InsufficientImu {
        /// Index of the offending interval.
        interval: usize,
    },
    /// Translational acceleration has essentially vanished — the static hold.
    LowExcitation {
        /// Mean excitation measured over the window, m/s².
        measured: Scalar,
        /// Threshold it failed.
        threshold: Scalar,
    },
    /// The camera rotated but did not translate, so no ruler can be calibrated
    /// in any units.
    PureRotation {
        /// Scale-column energy relative to the design matrix.
        ratio: Scalar,
        /// Threshold it failed.
        threshold: Scalar,
    },
    /// The linear system is rank-deficient.
    Singular {
        /// Ratio of least to greatest singular value.
        condition: Scalar,
    },
    /// The free-gravity solve did not recover a plausible gravity vector, so
    /// it fitted something other than gravity.
    GravityMagnitude {
        /// Magnitude recovered, m/s².
        measured: Scalar,
        /// Magnitude expected, m/s².
        expected: Scalar,
    },
    /// A solution exists but is too imprecise to be worth reporting.
    Imprecise {
        /// Relative standard deviation of the recovered scale.
        relative_stddev: Scalar,
        /// Threshold it failed.
        threshold: Scalar,
    },
    /// The solve produced a non-positive or non-finite multiplier.
    Degenerate,
}

impl std::fmt::Display for InertialRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InertialRejection::InsufficientPoses { have, need } => {
                write!(f, "insufficient poses: {have} of {need}")
            }
            InertialRejection::InsufficientImu { interval } => {
                write!(f, "insufficient imu in interval {interval}")
            }
            InertialRejection::LowExcitation {
                measured,
                threshold,
            } => write!(
                f,
                "excitation {measured:.3} m/s^2 below {threshold:.3}: scale unobservable"
            ),
            InertialRejection::PureRotation { ratio, threshold } => write!(
                f,
                "near-pure rotation: scale column ratio {ratio:.3e} below {threshold:.3e}"
            ),
            InertialRejection::Singular { condition } => {
                write!(f, "rank-deficient system, condition {condition:.3e}")
            }
            InertialRejection::GravityMagnitude { measured, expected } => write!(
                f,
                "recovered gravity {measured:.3} m/s^2 is not {expected:.3}"
            ),
            InertialRejection::Imprecise {
                relative_stddev,
                threshold,
            } => write!(
                f,
                "scale stddev {:.1}% exceeds {:.1}%",
                100.0 * relative_stddev,
                100.0 * threshold
            ),
            InertialRejection::Degenerate => write!(f, "degenerate solution"),
        }
    }
}

/// Everything the closed-form solve recovers.
#[derive(Debug, Clone, PartialEq)]
pub struct InertialSolution {
    /// Multiplier taking up-to-scale units to metres.
    pub scale: Scalar,
    /// Variance of `scale`.
    pub variance: Scalar,
    /// Gravity in the visual world frame, m/s². Its direction is the other
    /// half of what makes a monocular VI system metric.
    pub gravity: Vec3,
    /// Metric body velocity at each pose, m/s.
    pub velocities: Vec<Vec3>,
    /// Weighted residual RMS of the linear system.
    pub residual_rms: Scalar,
    /// Mean excitation over the window, m/s². spec.md §6 L5: report error
    /// against measured excitation, never as a single aggregate.
    pub excitation: Scalar,
    /// Time the window spans, seconds — the x-axis of the Campos curve.
    pub span_seconds: Scalar,
}

impl InertialSolution {
    /// Scale error as a percentage standard deviation.
    #[must_use]
    pub fn relative_stddev(&self) -> Scalar {
        if self.scale.abs() < 1e-12 {
            Scalar::INFINITY
        } else {
            self.variance.sqrt() / self.scale.abs()
        }
    }
}

/// IMU preintegration between two times, in the earlier body frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Preintegrated {
    /// Rotation increment.
    pub delta_rotation: Mat3,
    /// Velocity increment from specific force, m/s.
    pub delta_velocity: Vec3,
    /// Position increment from specific force, m.
    pub delta_position: Vec3,
    /// Interval length, seconds.
    pub dt: Scalar,
    /// Samples consumed.
    pub samples: usize,
}

/// Preintegrate the samples spanning `[from, to]`.
///
/// Midpoint rule rather than forward Euler: at 30 Hz keyframes and 200 Hz IMU
/// the Euler bias on `Δp` is a percent of the increment, which lands directly
/// on the scale it is supposed to measure. Endpoints are interpolated so the
/// integrated interval is exactly `to - from` — a truncated interval biases
/// `Δp` low and the recovered scale high.
///
/// Returns `None` if fewer than two samples bracket the interval.
#[must_use]
pub fn preintegrate(
    samples: &[ImuSample],
    from: Timestamp,
    to: Timestamp,
) -> Option<Preintegrated> {
    let dt_total = to.since(from);
    if dt_total <= 0.0 {
        return None;
    }
    let inside: Vec<&ImuSample> = samples
        .iter()
        .filter(|s| s.timestamp > from && s.timestamp < to)
        .collect();

    let before = samples.iter().rfind(|s| s.timestamp <= from);
    let after = samples.iter().find(|s| s.timestamp >= to);
    let first_inside = inside.first().copied();
    let last_inside = inside.last().copied();

    // Endpoint values: interpolate where we can, extend the nearest sample
    // where we cannot.
    let start = match (before, first_inside) {
        (Some(b), Some(f)) => ImuSample::lerp(b, f, from),
        (Some(b), None) => match after {
            Some(a) => ImuSample::lerp(b, a, from),
            None => ImuSample::new(from, b.gyro, b.accel),
        },
        (None, Some(f)) => ImuSample::new(from, f.gyro, f.accel),
        (None, None) => return None,
    };
    let end = match (last_inside, after) {
        (Some(l), Some(a)) => ImuSample::lerp(l, a, to),
        (Some(l), None) => ImuSample::new(to, l.gyro, l.accel),
        (None, Some(a)) => match before {
            Some(b) => ImuSample::lerp(b, a, to),
            None => ImuSample::new(to, a.gyro, a.accel),
        },
        (None, None) => return None,
    };

    let mut chain: Vec<ImuSample> = Vec::with_capacity(inside.len() + 2);
    chain.push(start);
    chain.extend(inside.iter().map(|s| **s));
    chain.push(end);
    if chain.len() < 2 {
        return None;
    }

    let mut delta_r = Mat3::identity();
    let mut delta_v = Vec3::zeros();
    let mut delta_p = Vec3::zeros();
    for pair in chain.windows(2) {
        let dt = pair[1].timestamp.since(pair[0].timestamp);
        if dt <= 0.0 {
            continue;
        }
        let a_mid = (pair[0].accel + pair[1].accel) * 0.5;
        let w_mid = (pair[0].gyro + pair[1].gyro) * 0.5;
        let a_world = delta_r * a_mid;
        delta_p += delta_v * dt + a_world * (0.5 * dt * dt);
        delta_v += a_world * dt;
        delta_r *= So3::exp(&(w_mid * dt)).matrix();
    }

    Some(Preintegrated {
        delta_rotation: delta_r,
        delta_velocity: delta_v,
        delta_position: delta_p,
        dt: dt_total,
        samples: chain.len(),
    })
}

/// The linear system `A x = b` with `x = [s, g, v_0 .. v_{n-1}]`.
#[derive(Debug, Clone)]
pub struct LinearSystem {
    /// Design matrix, `6(n-1)` rows by `4 + 3n` columns, row-weighted.
    pub a: DMatrix<Scalar>,
    /// Right-hand side, weighted to match.
    pub b: DVector<Scalar>,
    /// Poses the system was built from.
    pub poses: usize,
}

impl LinearSystem {
    /// Residual `A x - b` for an arbitrary state. Exposed so the harness can
    /// check the assembly against an independently generated ground truth.
    #[must_use]
    pub fn residual(&self, x: &DVector<Scalar>) -> DVector<Scalar> {
        &self.a * x - &self.b
    }

    /// Pack a state vector in the layout the columns expect.
    #[must_use]
    pub fn pack(scale: Scalar, gravity: Vec3, velocities: &[Vec3]) -> DVector<Scalar> {
        let mut x = DVector::zeros(4 + 3 * velocities.len());
        x[0] = scale;
        x.fixed_rows_mut::<3>(1).copy_from(&gravity);
        for (i, v) in velocities.iter().enumerate() {
            x.fixed_rows_mut::<3>(4 + 3 * i).copy_from(v);
        }
        x
    }
}

/// Assemble the visual-inertial linear system from a window.
///
/// Public because it is the natural place to check the physics: feeding it the
/// true state of a synthetic trajectory must produce a zero residual, which is
/// a much stronger assertion than any numerical Jacobian check.
pub fn assemble_system(
    window: &StateWindow,
    config: &InertialConfig,
) -> std::result::Result<LinearSystem, InertialRejection> {
    let poses: Vec<_> = window.poses().copied().collect();
    let n = poses.len();
    if n < config.min_poses.max(4) {
        return Err(InertialRejection::InsufficientPoses {
            have: n,
            need: config.min_poses.max(4),
        });
    }
    let imu: Vec<ImuSample> = window.imu().copied().collect();

    let r_cb = config.t_camera_body.rotation().matrix();
    let t_cb = config.t_camera_body.translation();

    let rows = 6 * (n - 1);
    let cols = 4 + 3 * n;
    let mut a = DMatrix::<Scalar>::zeros(rows, cols);
    let mut b = DVector::<Scalar>::zeros(rows);

    for i in 0..(n - 1) {
        let (pi, pj) = (&poses[i], &poses[i + 1]);
        let pre = preintegrate(&imu, pi.timestamp, pj.timestamp)
            .ok_or(InertialRejection::InsufficientImu { interval: i })?;
        if pre.samples < config.min_imu_per_interval.max(2) {
            return Err(InertialRejection::InsufficientImu { interval: i });
        }
        let dt = pre.dt;

        let r_i = pi.pose.rotation().matrix();
        let r_j = pj.pose.rotation().matrix();
        let r_wb = r_i * r_cb; // body-to-world at pose i

        // Row scaling: a position residual accumulates roughly ½ σ_a Δt² and a
        // velocity residual σ_a Δt, so dividing by those puts both blocks in
        // the same (dimensionless) units and stops the velocity rows
        // dominating the fit.
        let w_p = 1.0 / (0.5 * config.accel_noise * dt * dt).max(1e-12);
        let w_v = 1.0 / (config.accel_noise * dt).max(1e-12);

        let dp = pi.pose.translation() - pj.pose.translation();
        let rhs_p = r_wb * pre.delta_position - (r_j - r_i) * t_cb;
        let rhs_v = r_wb * pre.delta_velocity;

        let pr = 6 * i;
        let vr = 6 * i + 3;
        for k in 0..3 {
            // s (p̄_j - p̄_i) - Δt v_i - ½ Δt² g = rhs_p
            a[(pr + k, 0)] = -dp[k] * w_p;
            a[(pr + k, 1 + k)] = -0.5 * dt * dt * w_p;
            a[(pr + k, 4 + 3 * i + k)] = -dt * w_p;
            b[pr + k] = rhs_p[k] * w_p;

            // v_j - v_i - Δt g = rhs_v
            a[(vr + k, 1 + k)] = -dt * w_v;
            a[(vr + k, 4 + 3 * i + k)] = -w_v;
            a[(vr + k, 4 + 3 * (i + 1) + k)] = w_v;
            b[vr + k] = rhs_v[k] * w_v;
        }
    }

    Ok(LinearSystem { a, b, poses: n })
}

/// Solve for scale, gravity and velocities from an up-to-scale window.
///
/// # Errors
/// One of the named [`InertialRejection`] cases. Every one of them is a
/// documented degenerate condition rather than a numerical accident.
pub fn solve_inertial_scale(
    window: &StateWindow,
    config: &InertialConfig,
) -> std::result::Result<InertialSolution, InertialRejection> {
    let excitation = window.mean_excitation();
    if excitation < config.min_excitation {
        return Err(InertialRejection::LowExcitation {
            measured: excitation,
            threshold: config.min_excitation,
        });
    }

    let system = assemble_system(window, config)?;
    let n = system.poses;

    // Pure rotation shows up here as an exactly (or numerically) empty scale
    // column: the up-to-scale trajectory never moved, so no multiplier can be
    // calibrated in any units. spec.md §2 names this as a case where scale is
    // unobservable in principle.
    let scale_column = system.a.column(0).norm();
    let frobenius = system.a.norm();
    let ratio = if frobenius > 0.0 {
        scale_column / frobenius
    } else {
        0.0
    };
    if ratio < config.min_scale_column_ratio {
        return Err(InertialRejection::PureRotation {
            ratio,
            threshold: config.min_scale_column_ratio,
        });
    }

    let solved = least_squares(&system.a, &system.b, config.min_condition)?;
    let mut scale = solved.x[0];
    let mut gravity = Vec3::new(solved.x[1], solved.x[2], solved.x[3]);
    let mut velocities: Vec<Vec3> = (0..n)
        .map(|i| {
            Vec3::new(
                solved.x[4 + 3 * i],
                solved.x[5 + 3 * i],
                solved.x[6 + 3 * i],
            )
        })
        .collect();
    let mut residual_rms = solved.residual_rms;
    let mut scale_variance = solved.variance_unit[0] * solved.sigma_sq;

    let g_norm = gravity.norm();
    if (g_norm - config.gravity_magnitude).abs() > config.gravity_tolerance {
        return Err(InertialRejection::GravityMagnitude {
            measured: g_norm,
            expected: config.gravity_magnitude,
        });
    }

    // ORB-SLAM3's second stage: gravity's magnitude is known exactly, so
    // spending three degrees of freedom on it wastes information the free
    // solve had to buy from the data.
    if config.refine_with_fixed_gravity && g_norm > 1e-6 {
        let g_fixed = gravity * (config.gravity_magnitude / g_norm);
        if let Some(refined) = solve_with_fixed_gravity(&system, &g_fixed, config.min_condition) {
            scale = refined.x[0];
            velocities = (0..n)
                .map(|i| {
                    Vec3::new(
                        refined.x[1 + 3 * i],
                        refined.x[2 + 3 * i],
                        refined.x[3 + 3 * i],
                    )
                })
                .collect();
            residual_rms = refined.residual_rms;
            scale_variance = refined.variance_unit[0] * refined.sigma_sq;
            gravity = g_fixed;
        }
    }

    if !scale.is_finite() || scale <= 0.0 || !scale_variance.is_finite() || scale_variance < 0.0 {
        return Err(InertialRejection::Degenerate);
    }

    let solution = InertialSolution {
        scale,
        variance: scale_variance,
        gravity,
        velocities,
        residual_rms,
        excitation,
        span_seconds: window.span_seconds(),
    };

    let rel = solution.relative_stddev();
    if rel > config.max_relative_stddev {
        return Err(InertialRejection::Imprecise {
            relative_stddev: rel,
            threshold: config.max_relative_stddev,
        });
    }
    Ok(solution)
}

struct LeastSquares {
    x: DVector<Scalar>,
    /// Diagonal of `(A^T A)^-1`, i.e. the unit-variance covariance.
    variance_unit: DVector<Scalar>,
    /// Residual variance per degree of freedom.
    sigma_sq: Scalar,
    residual_rms: Scalar,
}

/// SVD least squares, returning the parameter covariance alongside the
/// solution. The covariance is not decoration — it is the observability gate.
fn least_squares(
    a: &DMatrix<Scalar>,
    b: &DVector<Scalar>,
    min_condition: Scalar,
) -> std::result::Result<LeastSquares, InertialRejection> {
    let svd = a.clone().svd(true, true);
    let (Some(u), Some(v_t)) = (svd.u.as_ref(), svd.v_t.as_ref()) else {
        return Err(InertialRejection::Singular { condition: 0.0 });
    };
    let s = &svd.singular_values;
    let smax = s.iter().copied().fold(0.0, Scalar::max);
    let smin = s.iter().copied().fold(Scalar::INFINITY, Scalar::min);
    let condition = if smax > 0.0 { smin / smax } else { 0.0 };
    if !(condition.is_finite() && condition > min_condition) {
        return Err(InertialRejection::Singular { condition });
    }

    let utb = u.transpose() * b;
    let mut y = DVector::zeros(s.len());
    for i in 0..s.len() {
        y[i] = utb[i] / s[i];
    }
    let x = v_t.transpose() * y;

    // diag(V S^-2 V^T) without forming the full matrix.
    let cols = a.ncols();
    let mut variance_unit = DVector::zeros(cols);
    for (j, var) in variance_unit.iter_mut().enumerate() {
        let mut acc = 0.0;
        for i in 0..s.len() {
            let vij = v_t[(i, j)];
            acc += vij * vij / (s[i] * s[i]);
        }
        *var = acc;
    }

    let residual = a * &x - b;
    let rows = a.nrows();
    let dof = (rows.saturating_sub(cols)).max(1) as Scalar;
    let sse = residual.norm_squared();
    Ok(LeastSquares {
        x,
        variance_unit,
        sigma_sq: sse / dof,
        residual_rms: (sse / rows as Scalar).sqrt(),
    })
}

/// Re-solve for `[s, v_0 .. v_{n-1}]` with gravity held at a known vector.
fn solve_with_fixed_gravity(
    system: &LinearSystem,
    gravity: &Vec3,
    min_condition: Scalar,
) -> Option<LeastSquares> {
    let rows = system.a.nrows();
    let cols = system.a.ncols() - 3;
    let mut a = DMatrix::<Scalar>::zeros(rows, cols);
    a.set_column(0, &system.a.column(0));
    for j in 0..(cols - 1) {
        a.set_column(1 + j, &system.a.column(4 + j));
    }
    let g_contrib =
        system.a.columns(1, 3) * DVector::from_column_slice(&[gravity.x, gravity.y, gravity.z]);
    let b = &system.b - g_contrib;
    least_squares(&a, &b, min_condition).ok()
}

/// Metric scale from double-integrated acceleration. Requires sensor tier 3.
#[derive(Debug, Clone)]
pub struct InertialScale {
    config: InertialConfig,
    last_solution: Option<InertialSolution>,
    last_rejection: Option<InertialRejection>,
}

impl InertialScale {
    /// Construct.
    ///
    /// # Errors
    /// [`Error::SensorTier`] when the `tight-vi` feature is off. spec.md §3
    /// specifies `ScaleSource.inertial()` *"requires tier 3; throws if L0
    /// unavailable"* — inertial scale is the one capability that genuinely
    /// needs sub-frame camera-IMU alignment, and degrading silently to a
    /// worse ruler would violate the rule that callers choose their anchor.
    ///
    /// The solver itself is always available as [`solve_inertial_scale`]; the
    /// gate is on wiring it into a tier-2 session, not on the mathematics.
    pub fn new(config: InertialConfig) -> Result<Self> {
        #[cfg(not(feature = "tight-vi"))]
        {
            let _ = &config;
            Err(Error::SensorTier {
                required: 3,
                reason: "inertial scale needs the L0 clock layer; build with feature `tight-vi`"
                    .into(),
            })
        }
        #[cfg(feature = "tight-vi")]
        Ok(InertialScale {
            config,
            last_solution: None,
            last_rejection: None,
        })
    }

    /// The most recent successful solve, with its gravity, velocities and
    /// measured excitation.
    #[must_use]
    pub fn last_solution(&self) -> Option<&InertialSolution> {
        self.last_solution.as_ref()
    }

    /// Why the most recent call declined. spec.md §6 L5 wants the excitation
    /// dependence reported, and this is where a caller reads it.
    #[must_use]
    pub fn last_rejection(&self) -> Option<InertialRejection> {
        self.last_rejection
    }

    /// The configuration in force.
    #[must_use]
    pub fn config(&self) -> &InertialConfig {
        &self.config
    }
}

impl ScaleSource for InertialScale {
    fn kind(&self) -> ScaleKind {
        ScaleKind::Inertial
    }

    fn estimate(&mut self, window: &StateWindow) -> Option<ScaleEstimate> {
        match solve_inertial_scale(window, &self.config) {
            Ok(sol) => {
                let e = ScaleEstimate::metric(ScaleKind::Inertial, sol.scale, sol.variance);
                self.last_solution = Some(sol);
                self.last_rejection = None;
                Some(e)
            }
            Err(rejection) => {
                log::debug!("inertial scale declined: {rejection}");
                self.last_rejection = Some(rejection);
                None
            }
        }
    }

    fn reset(&mut self) {
        self.last_solution = None;
        self.last_rejection = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn preintegrating_a_constant_specific_force_matches_the_closed_form() {
        // a = (1, 0, 0) held for one second, no rotation: Δv = a t,
        // Δp = ½ a t².
        let samples: Vec<ImuSample> = (0..=100)
            .map(|i| {
                ImuSample::new(
                    Timestamp::from_seconds(i as Scalar * 0.01),
                    Vec3::zeros(),
                    Vec3::new(1.0, 0.0, 0.0),
                )
            })
            .collect();
        let pre = preintegrate(&samples, Timestamp::ZERO, Timestamp::from_seconds(1.0)).unwrap();
        assert_relative_eq!(pre.dt, 1.0, epsilon = 1e-12);
        assert_relative_eq!(
            pre.delta_velocity,
            Vec3::new(1.0, 0.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            pre.delta_position,
            Vec3::new(0.5, 0.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(pre.delta_rotation, Mat3::identity(), epsilon = 1e-12);
    }

    #[test]
    fn preintegrating_a_constant_angular_rate_matches_the_closed_form() {
        let omega = Vec3::new(0.0, 0.0, 1.2);
        let samples: Vec<ImuSample> = (0..=200)
            .map(|i| {
                ImuSample::new(
                    Timestamp::from_seconds(i as Scalar * 0.005),
                    omega,
                    Vec3::zeros(),
                )
            })
            .collect();
        let pre = preintegrate(&samples, Timestamp::ZERO, Timestamp::from_seconds(1.0)).unwrap();
        assert_relative_eq!(
            pre.delta_rotation,
            So3::exp(&omega).matrix(),
            epsilon = 1e-9
        );
    }

    /// Endpoints must be interpolated, not snapped to the nearest sample: a
    /// truncated interval biases Δp low and the recovered scale high.
    #[test]
    fn interval_endpoints_are_interpolated_not_truncated() {
        let samples: Vec<ImuSample> = (0..=20)
            .map(|i| {
                ImuSample::new(
                    Timestamp::from_seconds(i as Scalar * 0.1),
                    Vec3::zeros(),
                    Vec3::new(2.0, 0.0, 0.0),
                )
            })
            .collect();
        // 0.05 .. 0.95 is 0.9 s and lines up with no sample at either end.
        let pre = preintegrate(
            &samples,
            Timestamp::from_seconds(0.05),
            Timestamp::from_seconds(0.95),
        )
        .unwrap();
        assert_relative_eq!(pre.dt, 0.9, epsilon = 1e-12);
        assert_relative_eq!(pre.delta_velocity.x, 1.8, epsilon = 1e-12);
        assert_relative_eq!(pre.delta_position.x, 0.81, epsilon = 1e-12);
    }

    #[test]
    fn preintegration_refuses_an_empty_or_inverted_interval() {
        let samples = vec![ImuSample::new(
            Timestamp::ZERO,
            Vec3::zeros(),
            Vec3::zeros(),
        )];
        assert!(preintegrate(&samples, Timestamp::from_seconds(1.0), Timestamp::ZERO).is_none());
        assert!(preintegrate(&[], Timestamp::ZERO, Timestamp::from_seconds(1.0)).is_none());
    }

    #[test]
    fn packing_round_trips_the_state_layout() {
        let v = vec![Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0)];
        let x = LinearSystem::pack(2.5, Vec3::new(0.0, 0.0, -9.8), &v);
        assert_eq!(x.len(), 4 + 6);
        assert_eq!(x[0], 2.5);
        assert_eq!(x[3], -9.8);
        assert_eq!(x[4], 1.0);
        assert_eq!(x[9], 6.0);
    }

    #[cfg(not(feature = "tight-vi"))]
    #[test]
    fn constructing_without_tier_three_fails_with_sensor_tier() {
        let err = InertialScale::new(InertialConfig::default()).unwrap_err();
        match err {
            Error::SensorTier {
                required,
                ref reason,
            } => {
                assert_eq!(required, 3);
                assert!(reason.contains("tight-vi"), "{reason}");
            }
            other => panic!("expected SensorTier, got {other:?}"),
        }
        assert!(!err.is_transient(), "a missing sensor tier is not a retry");
    }

    #[cfg(feature = "tight-vi")]
    #[test]
    fn constructing_with_tier_three_succeeds() {
        let s = InertialScale::new(InertialConfig::default()).unwrap();
        assert_eq!(s.kind(), ScaleKind::Inertial);
        assert!(s.last_solution().is_none());
        assert!(s.last_rejection().is_none());
    }

    #[test]
    fn rejection_messages_name_the_degenerate_case() {
        let r = InertialRejection::LowExcitation {
            measured: 0.02,
            threshold: 0.35,
        };
        assert!(r.to_string().contains("unobservable"), "{r}");
        let r = InertialRejection::PureRotation {
            ratio: 0.0,
            threshold: 1e-6,
        };
        assert!(r.to_string().contains("pure rotation"), "{r}");
    }
}
