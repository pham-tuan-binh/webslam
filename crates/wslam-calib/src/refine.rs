//! Joint nonlinear refinement of focal length, radial distortion and the
//! wrist lever arm, over every pair at once.
//!
//! This module is where spec.md §6 L2's two required ablations actually live.
//! The linear solve in [`crate::focal`] assumes a distortion-free camera
//! rotating exactly about its optical centre. Neither holds on a phone:
//!
//! * **Radial distortion.** Hayman & Murray (CVIU 2004) show barrel distortion
//!   produces "a sharply increasing overestimate of focal length, then outright
//!   failure". Estimating `k1`/`k2` jointly with `f` is the mitigation, and
//!   turning it off is how the ablation reproduces the finding.
//! * **The wrist lever arm.** Ji et al. show pure rotation is unachievable
//!   handheld: you rotate about your wrist, roughly 20 cm from the optical
//!   centre, and that injects a translation `t = (I - R) l`. spec.md §5 is
//!   explicit that the fix is to *fold it in* rather than bolt on a hack,
//!   because `l` is exactly the camera-IMU extrinsic a VI system already has.
//!
//! ## Why the lever arm brings a depth parameter with it
//!
//! With a lever arm the frame-to-frame map is no longer a homography for a
//! general scene, because the induced parallax scales with inverse depth. For a
//! scene at a single depth `z` it *is* a homography again:
//!
//! ```text
//! x2 ~ K (R + rho (I - R) l e3^T) K^-1 x1,     rho = 1/z
//! ```
//!
//! so one inverse depth per pair recovers an exact model, and that is what this
//! refinement estimates. `l` itself is held at the configured value rather than
//! estimated: only the product `rho * l` is observable per pair, so with a
//! per-pair `rho` free the magnitude of `l` has an exact gauge freedom. Trying
//! to estimate both would produce a rank-deficient system that looks like it
//! converged.

use nalgebra::{DMatrix, DVector, Matrix2x3};
use wslam_core::{Mat3, RadialTangential, Scalar, So3, Vec2, Vec3};

/// One matched frame pair, with the rotation L1 measured between the frames.
#[derive(Debug, Clone)]
pub struct PairObservation {
    /// Correspondences in **principal-point-centred** pixels, `(frame1, frame2)`.
    pub matches: Vec<(Vec2, Vec2)>,
    /// `R_cam2_cam1` from the gyro.
    pub rotation: So3,
}

/// What the refinement is allowed to model. The two `Option`/`bool` switches
/// here are the ablation knobs spec.md §6 L2 makes a gate.
#[derive(Debug, Clone)]
pub struct RefineOptions {
    /// Estimate `k1` and `k2` jointly with the focal length.
    pub model_distortion: bool,
    /// Camera-IMU translation (the pivot expressed in camera coordinates).
    /// `None` asserts the textbook pure-rotation model.
    pub lever_arm: Option<Vec3>,
    /// Levenberg-Marquardt iteration cap.
    pub max_iterations: usize,
    /// Huber transition, in pixels.
    pub huber_delta_px: Scalar,
    /// Sigma of a zero-mean Gaussian prior on `k1`/`k2`.
    ///
    /// Two residual rows total, not two per point, so it is negligible against
    /// a healthy pan and only bites when the correspondence set is thin enough
    /// that `k2` would otherwise be unidentifiable.
    pub distortion_prior_sigma: Scalar,
}

impl Default for RefineOptions {
    fn default() -> Self {
        RefineOptions {
            model_distortion: true,
            lever_arm: None,
            max_iterations: 40,
            huber_delta_px: 3.0,
            distortion_prior_sigma: 0.5,
        }
    }
}

/// Outcome of a refinement run.
#[derive(Debug, Clone)]
pub struct RefineReport {
    /// Refined focal length in pixels.
    pub focal_px: Scalar,
    /// Refined distortion; identity when `model_distortion` was off.
    pub distortion: RadialTangential,
    /// Per-pair scene inverse depth in 1/m; empty when the lever arm was off.
    pub inverse_depths: Vec<Scalar>,
    /// Variance of [`RefineReport::focal_px`], px^2, from the normal equations.
    pub focal_variance: Scalar,
    /// Levenberg-Marquardt iterations actually run.
    pub iterations: usize,
    /// Robust cost before the first step.
    pub initial_cost: Scalar,
    /// Robust cost at the returned parameters.
    pub final_cost: Scalar,
    /// Whether the step/cost tolerance was met before the iteration cap.
    pub converged: bool,
    /// RMS transfer residual at the solution, in pixels.
    pub residual_rms_px: Scalar,
}

/// Smallest usable focal, as a fraction of the starting guess. The solver is
/// free to move a long way but not through zero, where the model is singular.
const FOCAL_FLOOR_RATIO: Scalar = 0.05;
/// Largest usable focal, as a multiple of the starting guess.
const FOCAL_CEIL_RATIO: Scalar = 20.0;
/// Inverse-depth ceiling, 1/m. 20 corresponds to a scene 5 cm away — closer
/// than a phone camera can focus, so anything beyond it is the solver escaping.
const INVERSE_DEPTH_CEIL: Scalar = 20.0;

/// Relative cost improvement below which LM is declared converged.
const COST_TOLERANCE: Scalar = 1e-12;

/// Everything the residual function needs, with the parameter layout resolved
/// once: `[f, (k1, k2)?, (rho_0 .. rho_{n-1})?]`.
struct Problem<'a> {
    pairs: &'a [PairObservation],
    opts: &'a RefineOptions,
    /// Row offset of each pair's first correspondence block.
    offsets: Vec<usize>,
    total_matches: usize,
}

/// One transfer of a point through the model, with its analytic derivatives.
struct Transfer {
    /// Predicted centred pixel in the target frame.
    pred: Vec2,
    /// `d(pred)/d(f)`.
    d_f: Vec2,
    /// `d(pred)/d(k1)`, `d(pred)/d(k2)`.
    d_k: [Vec2; 2],
    /// `d(pred)/d(v)` where `v` is the 3-vector before dehomogenisation.
    d_v: Matrix2x3<Scalar>,
    /// The 3-vector before dehomogenisation, needed for the `rho` chain rule.
    v: Vec3,
}

/// Push a centred pixel through `undistort -> m -> distort`, accumulating
/// derivatives by the chain rule.
///
/// The only subtle step is differentiating `undistort`, which has no closed
/// form. Implicit differentiation of `distort(n; k) = d` gives
/// `dn/dd = A^-1` and `dn/dk = -A^-1 * d(distort)/dk`, with `A` the distortion
/// Jacobian `wslam_core` already provides — so no numerical derivative is
/// needed anywhere in this file.
fn transfer(q: Vec2, f: Scalar, dist: &RadialTangential, m: &Mat3) -> Option<Transfer> {
    if !(f > 0.0) || !f.is_finite() {
        return None;
    }
    let d = q / f;
    let n = dist.undistort(d);
    let a1 = dist.distort_jacobian(n);
    let a1_inv = a1.try_inverse()?;
    let nh = Vec3::new(n.x, n.y, 1.0);
    let v = m * nh;
    if v.z.abs() < 1e-9 || !v.iter().all(|c| c.is_finite()) {
        return None;
    }
    let inv_z = 1.0 / v.z;
    let mm = Vec2::new(v.x * inv_z, v.y * inv_z);
    let a2 = dist.distort_jacobian(mm);
    let dm = dist.distort(mm);
    let pred = f * dm;

    let j_proj = Matrix2x3::new(inv_z, 0.0, -mm.x * inv_z, 0.0, inv_z, -mm.y * inv_z);
    let d_v = f * (a2 * j_proj);
    let m_xy = m.fixed_view::<3, 2>(0, 0).into_owned();

    // d = q/f, so dd/df = -d/f; then through the implicit undistort Jacobian.
    let dn_df = a1_inv * (-d / f);
    let d_f = dm + d_v * (m_xy * dn_df);

    let r1sq = n.norm_squared();
    let r2sq = mm.norm_squared();
    let mut d_k = [Vec2::zeros(); 2];
    for (j, slot) in d_k.iter_mut().enumerate() {
        let pow_n = if j == 0 { r1sq } else { r1sq * r1sq };
        let pow_m = if j == 0 { r2sq } else { r2sq * r2sq };
        let dn_dk = -(a1_inv * (n * pow_n));
        *slot = d_v * (m_xy * dn_dk) + f * (mm * pow_m);
    }

    Some(Transfer {
        pred,
        d_f,
        d_k,
        d_v,
        v,
    })
}

/// Residual blocks: correspondences contribute a 2-vector each way, the
/// distortion prior a scalar per coefficient. Robust weighting is applied per
/// block, so a bad correspondence is down-weighted as a unit rather than
/// component-wise.
struct Block {
    start: usize,
    len: usize,
    /// Whether the block is a data residual (as opposed to a prior).
    is_data: bool,
}

impl<'a> Problem<'a> {
    fn new(pairs: &'a [PairObservation], opts: &'a RefineOptions) -> Self {
        let mut offsets = Vec::with_capacity(pairs.len());
        let mut total = 0usize;
        for p in pairs {
            offsets.push(total);
            total += p.matches.len();
        }
        Problem {
            pairs,
            opts,
            offsets,
            total_matches: total,
        }
    }

    fn n_distortion(&self) -> usize {
        if self.opts.model_distortion {
            2
        } else {
            0
        }
    }

    fn n_depth(&self) -> usize {
        if self.opts.lever_arm.is_some() {
            self.pairs.len()
        } else {
            0
        }
    }

    fn n_params(&self) -> usize {
        1 + self.n_distortion() + self.n_depth()
    }

    fn n_residuals(&self) -> usize {
        4 * self.total_matches + self.n_distortion()
    }

    fn depth_index(&self, pair: usize) -> Option<usize> {
        if self.opts.lever_arm.is_some() {
            Some(1 + self.n_distortion() + pair)
        } else {
            None
        }
    }

    fn distortion(&self, x: &DVector<Scalar>) -> RadialTangential {
        if self.opts.model_distortion {
            RadialTangential::radial(x[1], x[2])
        } else {
            RadialTangential::NONE
        }
    }

    /// Parameter vector at a focal start with no distortion.
    ///
    /// Only the Jacobian tests use this directly; the solver goes through
    /// [`Problem::initial_with`] so the multi-start can vary `k1`.
    #[cfg(test)]
    fn initial(&self, focal: Scalar) -> DVector<Scalar> {
        self.initial_with(focal, 0.0)
    }

    fn initial_with(&self, focal: Scalar, k1: Scalar) -> DVector<Scalar> {
        let mut x = DVector::zeros(self.n_params());
        x[0] = focal;
        if self.opts.model_distortion {
            x[1] = k1;
        }
        // rho starts at zero: the pure-rotation model is the honest prior, and
        // it is exactly what the linear stage assumed.
        x
    }

    fn clamp(&self, x: &mut DVector<Scalar>, focal0: Scalar) {
        x[0] = x[0].clamp(FOCAL_FLOOR_RATIO * focal0, FOCAL_CEIL_RATIO * focal0);
        for p in 0..self.pairs.len() {
            if let Some(i) = self.depth_index(p) {
                x[i] = x[i].clamp(0.0, INVERSE_DEPTH_CEIL);
            }
        }
    }

    fn blocks(&self) -> Vec<Block> {
        let mut blocks = Vec::with_capacity(2 * self.total_matches + self.n_distortion());
        for i in 0..self.total_matches {
            blocks.push(Block {
                start: 4 * i,
                len: 2,
                is_data: true,
            });
            blocks.push(Block {
                start: 4 * i + 2,
                len: 2,
                is_data: true,
            });
        }
        for j in 0..self.n_distortion() {
            blocks.push(Block {
                start: 4 * self.total_matches + j,
                len: 1,
                is_data: false,
            });
        }
        blocks
    }

    /// Evaluate residuals, and optionally the analytic Jacobian.
    ///
    /// Correspondences that cannot be transferred (a point crossing the plane
    /// at infinity under the current parameters) leave their rows at zero
    /// rather than shortening the vector, so the residual length is a function
    /// of the problem alone. That is what lets the finite-difference test in
    /// this module compare like with like.
    fn evaluate(&self, x: &DVector<Scalar>, want_jac: bool) -> Option<Eval> {
        let f = x[0];
        if !(f > 0.0) || !f.is_finite() {
            return None;
        }
        let dist = self.distortion(x);
        let mut residuals = DVector::zeros(self.n_residuals());
        let mut jac = want_jac.then(|| DMatrix::zeros(self.n_residuals(), self.n_params()));
        let mut contributing = 0usize;

        for (pi, pair) in self.pairs.iter().enumerate() {
            let rmat = pair.rotation.matrix();
            let t = match self.opts.lever_arm {
                // Ji et al.: rotating about the wrist injects exactly this.
                Some(l) => (Mat3::identity() - rmat) * l,
                None => Vec3::zeros(),
            };
            let rho = self.depth_index(pi).map_or(0.0, |i| x[i]);
            let mut m = rmat;
            {
                let mut c2 = m.column_mut(2);
                c2 += rho * t;
            }
            let Some(m_inv) = m.try_inverse() else {
                continue;
            };

            for (k, &(q1, q2)) in pair.matches.iter().enumerate() {
                let row = 4 * (self.offsets[pi] + k);
                let (Some(fwd), Some(bwd)) =
                    (transfer(q1, f, &dist, &m), transfer(q2, f, &dist, &m_inv))
                else {
                    continue;
                };
                contributing += 1;
                let rf = fwd.pred - q2;
                let rb = bwd.pred - q1;
                residuals[row] = rf.x;
                residuals[row + 1] = rf.y;
                residuals[row + 2] = rb.x;
                residuals[row + 3] = rb.y;

                let Some(j) = jac.as_mut() else { continue };
                j[(row, 0)] = fwd.d_f.x;
                j[(row + 1, 0)] = fwd.d_f.y;
                j[(row + 2, 0)] = bwd.d_f.x;
                j[(row + 3, 0)] = bwd.d_f.y;
                if self.opts.model_distortion {
                    for c in 0..2 {
                        j[(row, 1 + c)] = fwd.d_k[c].x;
                        j[(row + 1, 1 + c)] = fwd.d_k[c].y;
                        j[(row + 2, 1 + c)] = bwd.d_k[c].x;
                        j[(row + 3, 1 + c)] = bwd.d_k[c].y;
                    }
                }
                if let Some(ri) = self.depth_index(pi) {
                    // Forward: dv/drho = t. Backward: d(M^-1)/drho * n2h
                    // = -M^-1 (t e3^T) M^-1 n2h = -(M^-1 t) * w.z.
                    let dv_f = fwd.d_v * t;
                    let dv_b = bwd.d_v * (-(m_inv * t) * bwd.v.z);
                    j[(row, ri)] = dv_f.x;
                    j[(row + 1, ri)] = dv_f.y;
                    j[(row + 2, ri)] = dv_b.x;
                    j[(row + 3, ri)] = dv_b.y;
                }
            }
        }

        if self.opts.model_distortion {
            let inv_sigma = 1.0 / self.opts.distortion_prior_sigma.max(1e-9);
            let base = 4 * self.total_matches;
            residuals[base] = x[1] * inv_sigma;
            residuals[base + 1] = x[2] * inv_sigma;
            if let Some(j) = jac.as_mut() {
                j[(base, 1)] = inv_sigma;
                j[(base + 1, 2)] = inv_sigma;
            }
        }

        Some(Eval {
            residuals,
            jacobian: jac,
            contributing,
        })
    }
}

struct Eval {
    residuals: DVector<Scalar>,
    jacobian: Option<DMatrix<Scalar>>,
    contributing: usize,
}

/// Huber weight and cost for a residual block of norm `norm`.
fn huber(norm: Scalar, delta: Scalar) -> (Scalar, Scalar) {
    if norm <= delta {
        (1.0, 0.5 * norm * norm)
    } else {
        (delta / norm, delta * (norm - 0.5 * delta))
    }
}

/// Robust cost `sum_blocks huber(||r_block||)`.
fn robust_cost(residuals: &DVector<Scalar>, blocks: &[Block], delta: Scalar) -> Scalar {
    blocks
        .iter()
        .map(|b| {
            let norm = residuals.rows(b.start, b.len).norm();
            let d = if b.is_data { delta } else { Scalar::INFINITY };
            huber(norm, d).1
        })
        .sum()
}

/// Levenberg-Marquardt refinement of focal length, distortion and per-pair
/// scene depth over all pairs jointly.
///
/// `initial_focal` comes from the linear stage. Returns `None` when the problem
/// is empty, the start is not a usable focal length, or the normal equations
/// are singular at every damping level tried.
///
/// **This is the single-start primitive, and the `(f, k1)` landscape is
/// multi-modal.** Prefer [`refine_multistart`] unless you specifically want one
/// basin; see its documentation for why.
#[must_use]
pub fn refine(
    pairs: &[PairObservation],
    initial_focal: Scalar,
    options: &RefineOptions,
) -> Option<RefineReport> {
    refine_from(pairs, initial_focal, 0.0, options)
}

/// [`refine`] from an explicit `(focal, k1)` start.
#[must_use]
pub fn refine_from(
    pairs: &[PairObservation],
    initial_focal: Scalar,
    initial_k1: Scalar,
    options: &RefineOptions,
) -> Option<RefineReport> {
    if pairs.is_empty() || !(initial_focal > 0.0) || !initial_focal.is_finite() {
        return None;
    }
    let problem = Problem::new(pairs, options);
    if problem.total_matches < problem.n_params() {
        return None;
    }
    let blocks = problem.blocks();
    let delta = options.huber_delta_px.max(1e-6);

    let mut x = problem.initial_with(initial_focal, initial_k1);
    let mut eval = problem.evaluate(&x, true)?;
    let initial_cost = robust_cost(&eval.residuals, &blocks, delta);
    let mut cost = initial_cost;

    let n = problem.n_params();
    let mut lambda = 1e-4;
    let mut iterations = 0usize;
    let mut converged = false;

    for _ in 0..options.max_iterations {
        iterations += 1;
        let Some(jac) = eval.jacobian.as_ref() else {
            break;
        };

        let mut hessian = DMatrix::<Scalar>::zeros(n, n);
        let mut gradient = DVector::<Scalar>::zeros(n);
        for b in &blocks {
            let r = eval.residuals.rows(b.start, b.len);
            let jb = jac.rows(b.start, b.len);
            let d = if b.is_data { delta } else { Scalar::INFINITY };
            let (w, _) = huber(r.norm(), d);
            hessian += w * (jb.transpose() * jb);
            gradient += w * (jb.transpose() * r);
        }

        let mut stepped = false;
        for _ in 0..12 {
            let mut damped = hessian.clone();
            for i in 0..n {
                // Marquardt scaling: damp proportional to the curvature already
                // present, so a focal length in pixels and an inverse depth in
                // 1/m are damped comparably despite the unit mismatch.
                let scale = damped[(i, i)].abs().max(1e-12);
                damped[(i, i)] += lambda * scale;
            }
            let Some(step) = damped.lu().solve(&(-&gradient)) else {
                lambda *= 8.0;
                continue;
            };
            let mut candidate = &x + &step;
            problem.clamp(&mut candidate, initial_focal);
            let Some(trial) = problem.evaluate(&candidate, true) else {
                lambda *= 8.0;
                continue;
            };
            let trial_cost = robust_cost(&trial.residuals, &blocks, delta);
            if trial_cost.is_finite() && trial_cost < cost {
                let improvement = (cost - trial_cost) / cost.max(1e-30);
                x = candidate;
                eval = trial;
                cost = trial_cost;
                lambda = (lambda * 0.3).max(1e-12);
                stepped = true;
                if improvement < COST_TOLERANCE {
                    converged = true;
                }
                break;
            }
            lambda *= 8.0;
        }
        if !stepped {
            // No damping level improved the cost: we are at a local minimum to
            // the precision the linearisation supports.
            converged = true;
            break;
        }
        if converged {
            break;
        }
    }

    // Covariance from the normal equations at the solution.
    //
    // The symmetric transfer residual states each correspondence's two
    // geometric constraints twice, once per direction, so both `J^T W J` and
    // the residual sum double-count. Halving the information and using
    // `2 * n_correspondences` degrees of freedom cancels the double count and
    // leaves the textbook `s^2 (J^T W J)^-1` for the focal block.
    let jac = eval.jacobian.as_ref()?;
    let mut hessian = DMatrix::<Scalar>::zeros(n, n);
    let mut sum_sq = 0.0;
    for b in &blocks {
        let r = eval.residuals.rows(b.start, b.len);
        let jb = jac.rows(b.start, b.len);
        let d = if b.is_data { delta } else { Scalar::INFINITY };
        let (w, _) = huber(r.norm(), d);
        hessian += w * (jb.transpose() * jb);
        if b.is_data {
            sum_sq += w * r.norm_squared();
        }
    }
    let dof = (2 * eval.contributing).saturating_sub(n).max(1) as Scalar;
    let s2 = sum_sq / dof;
    let focal_variance = hessian
        .clone()
        .try_inverse()
        .map(|inv| (s2 * inv[(0, 0)]).max(0.0))
        .unwrap_or(Scalar::INFINITY);

    let residual_rms_px = if eval.contributing > 0 {
        (sum_sq / (4.0 * eval.contributing as Scalar)).sqrt()
    } else {
        Scalar::INFINITY
    };

    let inverse_depths = (0..pairs.len())
        .filter_map(|p| problem.depth_index(p).map(|i| x[i]))
        .collect();

    Some(RefineReport {
        focal_px: x[0],
        distortion: problem.distortion(&x),
        inverse_depths,
        focal_variance,
        iterations,
        initial_cost,
        final_cost: cost,
        converged,
        residual_rms_px,
    })
}

/// Focal multipliers tried by [`refine_multistart`].
///
/// The linear stage is distortion-blind, so on a barrel lens its focal estimate
/// is biased by several percent in a direction that depends on the scene. These
/// bracket it.
const FOCAL_SEEDS: [Scalar; 3] = [0.85, 1.0, 1.2];

/// `k1` values tried by [`refine_multistart`]. Zero (no distortion), a typical
/// phone barrel coefficient, and a strong one.
const K1_SEEDS: [Scalar; 3] = [0.0, -0.15, -0.30];

/// [`refine`] from several starts, keeping the lowest-cost solution.
///
/// **Why this exists.** The joint `(f, k1)` cost surface for a rotating camera
/// is genuinely multi-modal: focal length and radial distortion both act
/// radially, so there is a second basin in which a too-large focal is traded
/// against a too-weak barrel coefficient. Levenberg-Marquardt is a local
/// method and settles into whichever basin it starts in, reporting
/// `converged: true` either way.
///
/// This was not theoretical. Single-start refinement on noise-free synthetic
/// barrel data (`k1 = -0.28`) converged to `k1 ~= -0.149` with the focal length
/// 3% high, at a residual of 1-7 px where the true optimum reaches 1e-4 px —
/// and whether it found the right basin flipped unpredictably with the rotation
/// magnitude and the Huber threshold. A calibration that is right or wrong
/// depending on how far the user happened to pan is not a calibration.
///
/// Nine starts on a problem this small costs single-digit milliseconds, and it
/// runs once per session during init.
///
/// The residual is the discriminator: the basins differ by four orders of
/// magnitude in final cost, so picking the best is unambiguous rather than a
/// coin flip between near-ties.
#[must_use]
pub fn refine_multistart(
    pairs: &[PairObservation],
    initial_focal: Scalar,
    options: &RefineOptions,
) -> Option<RefineReport> {
    if !options.model_distortion {
        // Without a distortion parameter there is no trade-off and no second
        // basin; one start is the whole search.
        return refine(pairs, initial_focal, options);
    }

    let mut best: Option<RefineReport> = None;
    for scale in FOCAL_SEEDS {
        for k1 in K1_SEEDS {
            let Some(report) = refine_from(pairs, initial_focal * scale, k1, options) else {
                continue;
            };
            if !report.focal_px.is_finite() || report.focal_px <= 0.0 {
                continue;
            }
            let better = match &best {
                None => true,
                Some(b) => report.final_cost < b.final_cost,
            };
            if better {
                best = Some(report);
            }
        }
    }
    // Fall back to the caller's own start if every seed was rejected, so this
    // can only ever be at least as good as calling `refine` directly.
    best.or_else(|| refine(pairs, initial_focal, options))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::SyntheticRig;
    use approx::assert_relative_eq;
    use wslam_core::DeterministicRng;

    fn pairs_from(rig: &SyntheticRig, angles: &[Vec3], seed: u64) -> Vec<PairObservation> {
        let mut rng = DeterministicRng::new("test-pairs", seed);
        angles
            .iter()
            .map(|phi| {
                let rotation = So3::exp(phi);
                PairObservation {
                    matches: rig.centred_pair(&rotation, 200, &mut rng),
                    rotation,
                }
            })
            .collect()
    }

    /// Central differences of the residual function, column by column.
    fn numeric_jacobian(problem: &Problem, x: &DVector<Scalar>) -> DMatrix<Scalar> {
        let mut out = DMatrix::zeros(problem.n_residuals(), problem.n_params());
        for i in 0..problem.n_params() {
            // Step sizes scaled to the parameter: f is ~10^3 px, k and rho are
            // O(0.1), so a single epsilon would be wrong for one of them.
            let eps = if i == 0 { 1e-4 } else { 1e-6 };
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[i] += eps;
            minus[i] -= eps;
            let rp = problem.evaluate(&plus, false).unwrap().residuals;
            let rm = problem.evaluate(&minus, false).unwrap().residuals;
            out.set_column(i, &((rp - rm) / (2.0 * eps)));
        }
        out
    }

    fn assert_jacobian_matches(problem: &Problem, x: &DVector<Scalar>) {
        let analytic = problem.evaluate(x, true).unwrap().jacobian.unwrap();
        let numeric = numeric_jacobian(problem, x);
        for c in 0..problem.n_params() {
            // Compare column-wise: the columns differ in magnitude by orders,
            // and a single global tolerance would let the small ones through.
            let scale = numeric.column(c).amax().max(1e-6);
            let err = (analytic.column(c) - numeric.column(c)).amax();
            assert!(
                err / scale < 2e-5,
                "column {c}: max error {err} against scale {scale}"
            );
        }
    }

    #[test]
    fn jacobian_matches_central_differences_pinhole() {
        let rig = SyntheticRig::pinhole(985.0, 1280, 720);
        let pairs = pairs_from(
            &rig,
            &[Vec3::new(0.0, 0.05, 0.0), Vec3::new(0.03, 0.0, 0.01)],
            1,
        );
        let opts = RefineOptions {
            model_distortion: false,
            lever_arm: None,
            ..Default::default()
        };
        let problem = Problem::new(&pairs, &opts);
        let x = problem.initial(940.0);
        assert_jacobian_matches(&problem, &x);
    }

    #[test]
    fn jacobian_matches_central_differences_with_distortion_and_lever_arm() {
        let mut rig = SyntheticRig::pinhole(985.0, 1280, 720);
        rig.distortion = RadialTangential::radial(-0.28, 0.09);
        rig.lever_arm = Some(Vec3::new(0.02, 0.05, -0.20));
        rig.depth = 1.5;
        let pairs = pairs_from(
            &rig,
            &[Vec3::new(0.0, 0.05, 0.0), Vec3::new(0.03, -0.02, 0.01)],
            2,
        );
        let opts = RefineOptions {
            model_distortion: true,
            lever_arm: rig.lever_arm,
            ..Default::default()
        };
        let problem = Problem::new(&pairs, &opts);
        // Off the truth in every parameter, so no derivative is evaluated at a
        // point where it happens to vanish.
        let mut x = problem.initial(940.0);
        x[1] = -0.19;
        x[2] = 0.04;
        x[3] = 0.55;
        x[4] = 0.71;
        assert_jacobian_matches(&problem, &x);
    }

    #[test]
    fn refine_is_a_fixed_point_at_the_truth() {
        // Noise-free data at the true parameters must already be a minimum:
        // if refinement moves, the model and the generator disagree.
        let rig = SyntheticRig::pinhole(985.0, 1280, 720);
        let pairs = pairs_from(
            &rig,
            &[Vec3::new(0.0, 0.05, 0.0), Vec3::new(0.02, -0.04, 0.0)],
            3,
        );
        let opts = RefineOptions {
            model_distortion: true,
            lever_arm: None,
            ..Default::default()
        };
        let report = refine(&pairs, 985.0, &opts).unwrap();
        assert!(
            report.residual_rms_px < 1e-9,
            "rms {}",
            report.residual_rms_px
        );
        assert_relative_eq!(report.focal_px, 985.0, max_relative = 1e-9);
        assert_relative_eq!(report.distortion.k1, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn refine_recovers_focal_and_k1_from_noise_free_barrel_data() {
        let mut rig = SyntheticRig::pinhole(985.0, 1280, 720);
        rig.distortion = RadialTangential::radial(-0.28, 0.0);
        let pairs = pairs_from(
            &rig,
            &[
                Vec3::new(0.0, 0.04, 0.0),
                Vec3::new(0.035, 0.0, 0.0),
                Vec3::new(0.02, -0.03, 0.01),
            ],
            4,
        );
        let opts = RefineOptions {
            model_distortion: true,
            lever_arm: None,
            ..Default::default()
        };
        // Start 12% off, as the distortion-blind linear stage would.
        let report = refine_multistart(&pairs, 1100.0, &opts).unwrap();
        assert_relative_eq!(report.focal_px, 985.0, max_relative = 2e-3);
        assert_relative_eq!(report.distortion.k1, -0.28, epsilon = 5e-3);
        assert!(report.residual_rms_px < 1e-3);
    }

    /// The reason [`refine_multistart`] exists, pinned as a test.
    ///
    /// Single-start LM on this exact problem settles into a second basin where
    /// a 3%-high focal is traded against a half-strength barrel coefficient,
    /// and reports `converged: true` while sitting at a residual four orders of
    /// magnitude worse than the true optimum. If this test ever starts passing
    /// the `assert!(single_start_is_wrong)` check trivially — because the
    /// landscape changed, or because someone made `refine` itself multi-start —
    /// then `refine_multistart` has become redundant and should be removed
    /// rather than left as cargo cult.
    #[test]
    fn single_start_refinement_lands_in_the_wrong_basin() {
        let mut rig = SyntheticRig::pinhole(985.0, 1280, 720);
        rig.distortion = RadialTangential::radial(-0.28, 0.0);
        let pairs = pairs_from(
            &rig,
            &[
                Vec3::new(0.0, 0.04, 0.0),
                Vec3::new(0.035, 0.0, 0.0),
                Vec3::new(0.02, -0.03, 0.01),
            ],
            4,
        );
        let opts = RefineOptions {
            model_distortion: true,
            lever_arm: None,
            ..Default::default()
        };

        let single = refine(&pairs, 1100.0, &opts).unwrap();
        let multi = refine_multistart(&pairs, 1100.0, &opts).unwrap();

        // The wrong basin looks convergent from the inside.
        assert!(single.converged);
        assert!(
            (single.focal_px - 985.0).abs() / 985.0 > 0.02,
            "single start unexpectedly found the truth: {}",
            single.focal_px
        );
        // And the residual is what gives it away, which is why the multi-start
        // selects on final cost rather than on any parameter heuristic.
        assert!(
            multi.final_cost < single.final_cost,
            "multi-start cost {} should beat single-start {}",
            multi.final_cost,
            single.final_cost
        );
        assert_relative_eq!(multi.focal_px, 985.0, max_relative = 2e-3);
    }

    #[test]
    fn refine_recovers_scene_depth_when_the_lever_arm_is_modelled() {
        let mut rig = SyntheticRig::pinhole(985.0, 1280, 720);
        rig.lever_arm = Some(Vec3::new(0.0, 0.0, -0.20));
        rig.depth = 2.0;
        rig.depth_spread = 0.0;
        let pairs = pairs_from(
            &rig,
            &[Vec3::new(0.0, 0.05, 0.0), Vec3::new(0.0, 0.08, 0.0)],
            5,
        );
        let opts = RefineOptions {
            model_distortion: false,
            lever_arm: rig.lever_arm,
            ..Default::default()
        };
        let report = refine(&pairs, 1010.0, &opts).unwrap();
        assert_relative_eq!(report.focal_px, 985.0, max_relative = 1e-4);
        for rho in &report.inverse_depths {
            assert_relative_eq!(*rho, 0.5, max_relative = 1e-3);
        }
    }

    #[test]
    fn refine_rejects_an_empty_or_impossible_problem() {
        let opts = RefineOptions::default();
        assert!(refine(&[], 985.0, &opts).is_none());
        let rig = SyntheticRig::pinhole(985.0, 1280, 720);
        let pairs = pairs_from(&rig, &[Vec3::new(0.0, 0.05, 0.0)], 6);
        assert!(refine(&pairs, -1.0, &opts).is_none());
        assert!(refine(&pairs, Scalar::NAN, &opts).is_none());
    }

    #[test]
    fn huber_downweights_beyond_the_transition() {
        let (w_in, c_in) = huber(1.0, 3.0);
        assert_relative_eq!(w_in, 1.0);
        assert_relative_eq!(c_in, 0.5);
        let (w_out, c_out) = huber(30.0, 3.0);
        assert_relative_eq!(w_out, 0.1);
        // Linear, not quadratic: a 30 px blunder must not dominate the sum.
        assert_relative_eq!(c_out, 3.0 * (30.0 - 1.5));
        assert!(c_out < 0.5 * 30.0 * 30.0);
    }
}
