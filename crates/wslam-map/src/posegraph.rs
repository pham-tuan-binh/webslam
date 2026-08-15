//! Pose-graph optimisation over SE(3) — L4b.
//!
//! spec.md §4 L4b: *"Loop closure + pose graph — second. Bounds global drift,
//! particularly yaw, which L1 and L3 only arrest locally."* spec.md §7 fixes
//! the shape of the thing just as firmly: *"We need a pose-graph optimizer, not
//! a solver framework. g2o and Ceres are general; our problem is not."* So this
//! module is Levenberg-Marquardt specialised to exactly one residual, with one
//! robust kernel and one sparsity pattern — not a factor graph library.
//!
//! # The problem
//!
//! A node holds `T_world_keyframe`. An edge `(i, j)` holds a measured relative
//! transform `Z_ij ≈ T_i^-1 · T_j` and an information matrix `Ω = Σ^-1`. The
//! residual is
//!
//! ```text
//! r_ij = log( Z_ij^-1 · T_i^-1 · T_j )        ∈ R^6, ordered [rho; phi]
//! ```
//!
//! and the objective is `½ Σ_k ρ(r_k^T Ω_k r_k)` with `ρ` the Huber kernel.
//!
//! # Conventions
//!
//! Perturbations are right-multiplied, `T ⊞ δ = T · exp(δ)`, and twists are
//! `[rho; phi]` — the convention `wslam_core::math` fixes once for the whole
//! workspace. The Jacobians below are therefore right Jacobians:
//!
//! ```text
//! ∂r/∂δ_j =  Jr^-1(r)
//! ∂r/∂δ_i = -Jr^-1(r) · Adj(T_j^-1 · T_i)
//! ```
//!
//! Both are analytic. `Jr^-1` for SE(3) needs Barfoot's `Q` block, which is the
//! only genuinely fiddly piece of algebra here and is pinned against central
//! finite differences in the tests. Finite differences never run at runtime.
//!
//! # Gauge
//!
//! An unfixed pose graph has a 6-dimensional null space — every solution is a
//! solution after a global rigid transform — and the normal equations are
//! singular. At least one node must therefore be held fixed. If the caller
//! fixes none, [`PoseGraph::optimize`] anchors the first node for the duration
//! of the solve and logs it rather than returning NaN.
//!
//! # Robustification
//!
//! spec.md §5 calls a false-positive loop closure *"irrecoverably"* corrupting
//! and *"worse than no loop closure at all"*. Geometric verification
//! ([`crate::Relocalizer::verify`]) is the first line of defence; the Huber
//! kernel here is the second, and it is what turns a closure that slips through
//! verification from a destroyed map into a locally distorted one.

#![warn(missing_docs)]

use std::collections::BTreeMap;

use wslam_core::math::{hat, split_twist, So3};
use wslam_core::{Mat3, Mat6, Se3, Vec3, Vec6};

use crate::KeyframeId;

// --------------------------------------------------------------------------
// Tuning constants
// --------------------------------------------------------------------------

/// Floor on the per-coordinate diagonal used for Marquardt scaling. Without it
/// a node that no surviving edge touches has a zero Hessian block and the
/// damped system is singular rather than merely uninformative.
const DAMP_FLOOR: f64 = 1e-6;

/// Damping bounds. Exceeding `LAMBDA_MAX` without an accepted step means no
/// descent direction exists to numerical precision, which is a minimum.
const LAMBDA_MIN: f64 = 1e-12;
const LAMBDA_MAX: f64 = 1e12;

/// A step smaller than this (metres / radians) cannot change the estimate, so
/// there is nothing left to do. This is what makes an already-converged graph
/// a fixed point instead of a damping-thrash.
const STEP_EPSILON: f64 = 1e-14;

/// Damping increases per outer iteration before giving up on that iteration.
const MAX_DAMPING_RETRIES: usize = 12;

/// Below this rotation angle Barfoot's `Q` is evaluated by Taylor series.
///
/// The closed-form coefficients cancel `O(θ)` terms down to `O(θ^5)`, so their
/// *relative* accuracy degrades as `θ^-4`; at `θ = 1e-3` that is already ~1e-2.
/// The cancellation is benign in the assembled matrix (the lost digits multiply
/// blocks of order `θ^4`) but the series is free, so we take it. `1e-2` is well
/// inside the regime where the two branches agree to ~1e-11.
const Q_SMALL_ANGLE: f64 = 1e-2;

/// Crossover between the two linear solvers, in predicted block operations per
/// node. Sparse Cholesky costs `Σ_j |col_j|^2` block multiplies, so this is a
/// budget on the *fill*, not on `N` — which is the honest metric: a 500-node
/// chain with a handful of loop closures factors with `|col_j| ≈ 2` regardless
/// of `N`, while a densely covisible graph blows up at any size. Above the
/// budget we fall back to preconditioned conjugate gradient, whose cost is
/// linear in the number of edges per iteration.
const CHOLESKY_BLOCK_OP_BUDGET: usize = 256;

/// PCG stops when the residual falls this far below the initial right-hand
/// side, or after `PCG_MAX_ITERATIONS`, whichever comes first. A truncated step
/// is fine — LM judges every step on the actual cost, so an inexact Newton
/// direction costs outer iterations, never correctness.
const PCG_RELATIVE_TOLERANCE: f64 = 1e-12;
const PCG_MAX_ITERATIONS: usize = 400;

// --------------------------------------------------------------------------
// SE(3) right Jacobian
// --------------------------------------------------------------------------

/// Barfoot's `Q` block — the top-right block of the SE(3) left Jacobian.
///
/// Barfoot, *State Estimation for Robotics* (2017), eq. 7.86, in the `[rho; phi]`
/// ordering this workspace uses.
fn se3_q(rho: &Vec3, phi: &Vec3) -> Mat3 {
    let theta_sq = phi.norm_squared();
    let rx = hat(rho);
    let px = hat(phi);

    let (c1, c2, c3) = if theta_sq < Q_SMALL_ANGLE * Q_SMALL_ANGLE {
        (
            1.0 / 6.0 - theta_sq / 120.0,
            1.0 / 24.0 - theta_sq / 720.0,
            1.0 / 120.0 - theta_sq / 2520.0,
        )
    } else {
        let theta = theta_sq.sqrt();
        let (s, c) = theta.sin_cos();
        let t4 = theta_sq * theta_sq;
        (
            (theta - s) / (theta_sq * theta),
            (theta_sq + 2.0 * c - 2.0) / (2.0 * t4),
            (2.0 * theta - 3.0 * s + theta * c) / (2.0 * t4 * theta),
        )
    };

    let a = px * rx + rx * px + px * rx * px;
    let b = px * px * rx + rx * px * px - 3.0 * (px * rx * px);
    let d = px * rx * px * px + px * px * rx * px;

    0.5 * rx + c1 * a + c2 * b + c3 * d
}

/// Inverse of the SE(3) **left** Jacobian at `xi = [rho; phi]`.
fn se3_left_jacobian_inv(xi: &Vec6) -> Mat6 {
    let (rho, phi) = split_twist(xi);
    let jl_inv = So3::left_jacobian_inv(&phi);
    let top_right = -(jl_inv * se3_q(&rho, &phi) * jl_inv);

    let mut m = Mat6::zeros();
    m.fixed_view_mut::<3, 3>(0, 0).copy_from(&jl_inv);
    m.fixed_view_mut::<3, 3>(0, 3).copy_from(&top_right);
    m.fixed_view_mut::<3, 3>(3, 3).copy_from(&jl_inv);
    m
}

/// Inverse of the SE(3) **right** Jacobian at `xi = [rho; phi]`.
///
/// Defined by `log(exp(xi) · exp(δ)) = xi + Jr^-1(xi) δ + O(δ^2)`, which is
/// exactly the derivative a right-perturbation residual needs. Uses
/// `Jr(xi) = Jl(-xi)`.
#[must_use]
pub fn se3_right_jacobian_inv(xi: &Vec6) -> Mat6 {
    se3_left_jacobian_inv(&(-xi))
}

// --------------------------------------------------------------------------
// Residual
// --------------------------------------------------------------------------

/// The pose-graph residual `log(Z^-1 · T_i^-1 · T_j)`, in `[rho; phi]` order.
///
/// Public because the orchestration layer wants the same quantity to
/// chi-squared-gate a loop candidate before it is ever added as an edge.
#[must_use]
pub fn edge_residual(t_i: &Se3, t_j: &Se3, measurement: &Se3) -> Vec6 {
    measurement
        .inverse()
        .compose(&t_i.inverse().compose(t_j))
        .log()
}

/// Residual plus its two analytic right Jacobians, `(r, ∂r/∂δ_i, ∂r/∂δ_j)`.
fn edge_jacobians(t_i: &Se3, t_j: &Se3, measurement: &Se3) -> (Vec6, Mat6, Mat6) {
    let t_ij = t_i.inverse().compose(t_j);
    let r = measurement.inverse().compose(&t_ij).log();
    let j_inv = se3_right_jacobian_inv(&r);
    // Perturbing i by δ pushes exp(-Adj(T_ij^-1) δ) through to the right of the
    // error term, hence the adjoint and the sign.
    let j_i = -(j_inv * t_ij.inverse().adjoint());
    (r, j_i, j_inv)
}

/// Huber `(ρ(e), IRLS weight)` for a squared Mahalanobis error `e = r^T Ω r`.
///
/// `delta` is a threshold on `sqrt(e)`, i.e. on the Mahalanobis *distance*, so
/// it is dimensionless and comparable against a chi-squared quantile.
fn huber(e_sq: f64, delta: Option<f64>) -> (f64, f64) {
    match delta {
        Some(d) if e_sq > d * d => {
            let e = e_sq.sqrt();
            (2.0 * d * e - d * d, d / e)
        }
        _ => (e_sq, 1.0),
    }
}

/// Use only the symmetric part of a caller-supplied information matrix. The
/// antisymmetric part contributes nothing to `r^T Ω r` but would make `J^T Ω J`
/// non-symmetric, which silently breaks Cholesky.
fn symmetrised(m: &Mat6) -> Mat6 {
    0.5 * (m + m.transpose())
}

// --------------------------------------------------------------------------
// Public types
// --------------------------------------------------------------------------

/// A relative-pose constraint between two keyframes.
pub struct Edge {
    /// Source keyframe.
    pub from: KeyframeId,
    /// Target keyframe.
    pub to: KeyframeId,
    /// Measured `T_from_to`.
    pub measurement: Se3,
    /// Information matrix `Σ^-1`, `[translation; rotation]` block order.
    pub information: Mat6,
}

impl std::fmt::Debug for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Edge")
            .field("from", &self.from.0)
            .field("to", &self.to.0)
            .field("measurement", &self.measurement)
            .finish_non_exhaustive()
    }
}

/// Solver knobs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverConfig {
    /// Maximum outer Levenberg-Marquardt iterations.
    pub max_iterations: usize,
    /// Relative cost-decrease below which the solve is declared converged.
    pub tolerance: f64,
    /// Initial LM damping. Increased on a rejected step, decreased on accept.
    pub lambda: f64,
    /// Huber threshold on the Mahalanobis distance `sqrt(r^T Ω r)`.
    /// Non-positive or non-finite disables robustification entirely.
    pub huber_delta: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            max_iterations: 30,
            tolerance: 1e-9,
            lambda: 1e-4,
            // sqrt of the 95th percentile of chi^2 with 6 degrees of freedom
            // (12.592): an edge consistent with its own stated covariance is
            // inside this 95% of the time, so nothing but genuine outliers gets
            // down-weighted.
            huber_delta: 3.548_5,
        }
    }
}

impl SolverConfig {
    /// The Huber threshold, or `None` when robustification is disabled.
    fn robust(&self) -> Option<f64> {
        (self.huber_delta.is_finite() && self.huber_delta > 0.0).then_some(self.huber_delta)
    }
}

/// What the solve did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverReport {
    /// Outer iterations performed.
    pub iterations: usize,
    /// `½ Σ ρ(r^T Ω r)` before any step.
    pub initial_cost: f64,
    /// The same quantity after the last accepted step.
    pub final_cost: f64,
    /// `true` when a stopping criterion fired, `false` when the iteration or
    /// damping budget ran out while progress was still being made.
    pub converged: bool,
}

struct Node {
    id: u64,
    pose: Se3,
    fixed: bool,
}

/// Pose graph over SE(3), optimised by Levenberg-Marquardt.
///
/// See the module docs for the residual, the Jacobians and the gauge rule.
pub struct PoseGraph {
    nodes: Vec<Node>,
    /// `KeyframeId.0 -> index into nodes`.
    index: BTreeMap<u64, usize>,
    edges: Vec<Edge>,
}

impl Default for PoseGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl PoseGraph {
    /// An empty graph.
    #[must_use]
    pub fn new() -> Self {
        PoseGraph {
            nodes: Vec::new(),
            index: BTreeMap::new(),
            edges: Vec::new(),
        }
    }

    /// Insert a node, or update the pose and fixed flag of an existing one.
    pub fn add_node(&mut self, id: KeyframeId, pose: Se3, fixed: bool) {
        let raw = id.0;
        match self.index.get(&raw) {
            Some(&k) => {
                self.nodes[k].pose = pose;
                self.nodes[k].fixed = fixed;
            }
            None => {
                self.index.insert(raw, self.nodes.len());
                self.nodes.push(Node {
                    id: raw,
                    pose,
                    fixed,
                });
            }
        }
    }

    /// Add a relative-pose constraint `measurement ≈ T_from^-1 · T_to`.
    ///
    /// Endpoints need not exist yet; they are resolved at [`Self::optimize`]
    /// time and an edge that still names an unknown keyframe then is skipped.
    /// A self-edge (`from == to`) is retained for the debug surface but never
    /// linearised: its two Jacobians cancel exactly, so it constrains nothing
    /// and only adds a constant to the cost.
    pub fn add_edge(
        &mut self,
        from: KeyframeId,
        to: KeyframeId,
        measurement: Se3,
        information: Mat6,
    ) {
        if from.0 == to.0 {
            log::warn!(
                "pose graph: self-edge on keyframe {} constrains nothing; ignoring it in the solve",
                from.0
            );
        }
        self.edges.push(Edge {
            from,
            to,
            measurement,
            information,
        });
    }

    /// Current estimate for a keyframe, if it is in the graph.
    #[must_use]
    pub fn pose(&self, id: KeyframeId) -> Option<Se3> {
        self.index.get(&id.0).map(|&k| self.nodes[k].pose)
    }

    /// Overwrite a node's estimate. The epoch-merge path: when a whole
    /// coordinate frame is re-expressed through a Sim(3), every node born in
    /// it moves at once. Returns `false` if the keyframe is not in the graph.
    pub fn set_pose(&mut self, id: KeyframeId, pose: Se3) -> bool {
        match self.index.get(&id.0) {
            Some(&k) => {
                self.nodes[k].pose = pose;
                true
            }
            None => false,
        }
    }

    /// Rescale the translational part of every edge between the given nodes.
    ///
    /// A similarity with scale `s` maps a relative measurement
    /// `T_ij = T_i⁻¹∘T_j` to one with the same rotation and `s·t_ij`: the
    /// rotations cancel the similarity's, the baseline does not. Edges with
    /// only one endpoint in `ids` are left alone — those cross the frame
    /// boundary and are the caller's problem to re-derive or drop.
    pub fn rescale_edges_within(&mut self, ids: &std::collections::HashSet<KeyframeId>, s: f64) {
        for edge in &mut self.edges {
            if ids.contains(&edge.from) && ids.contains(&edge.to) {
                edge.measurement = Se3::new(
                    edge.measurement.rotation(),
                    edge.measurement.translation() * s,
                );
            }
        }
    }

    /// All edges, in insertion order — including ones the solver skips, because
    /// the debug surface (spec.md §3) wants to draw rejected candidates too.
    #[must_use]
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// `true` when the graph has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Every node as `(id, pose, fixed)`, in insertion order.
    pub fn nodes(&self) -> impl Iterator<Item = (KeyframeId, Se3, bool)> + '_ {
        self.nodes
            .iter()
            .map(|n| (KeyframeId(n.id), n.pose, n.fixed))
    }

    /// Optimise, and write the result back into the nodes.
    ///
    /// Levenberg-Marquardt with Huber-weighted IRLS. The reported costs are
    /// `½ Σ ρ(r^T Ω r)` under the *same* kernel throughout, so
    /// `final_cost <= initial_cost` always holds: a step is only accepted when
    /// it lowers the robust cost, not the linearised model's prediction of it.
    pub fn optimize(&mut self, config: &SolverConfig) -> SolverReport {
        let robust = config.robust();

        // Resolve edge endpoints once. `None` means "skip": unknown keyframe or
        // a self-edge.
        let pairs: Vec<Option<(usize, usize)>> = self
            .edges
            .iter()
            .map(|e| {
                let i = *self.index.get(&e.from.0)?;
                let j = *self.index.get(&e.to.0)?;
                (i != j).then_some((i, j))
            })
            .collect();

        let mut poses: Vec<Se3> = self.nodes.iter().map(|n| n.pose).collect();
        let initial_cost = total_cost(&poses, &self.edges, &pairs, robust);

        if self.nodes.is_empty() {
            return SolverReport {
                iterations: 0,
                initial_cost,
                final_cost: initial_cost,
                converged: true,
            };
        }

        // Gauge (spec.md §4 L4b): six unobservable directions unless something
        // is pinned. Anchoring is local to this solve; the caller's `fixed`
        // flags are not rewritten behind its back.
        let mut fixed: Vec<bool> = self.nodes.iter().map(|n| n.fixed).collect();
        if !fixed.iter().any(|&f| f) {
            fixed[0] = true;
            log::warn!(
                "pose graph has no fixed node; anchoring keyframe {} for this solve to remove \
                 the 6-DoF gauge freedom",
                self.nodes[0].id
            );
        }

        let mut free = vec![None; self.nodes.len()];
        let mut n_free = 0usize;
        for (k, &f) in fixed.iter().enumerate() {
            if !f {
                free[k] = Some(n_free);
                n_free += 1;
            }
        }
        if n_free == 0 {
            return SolverReport {
                iterations: 0,
                initial_cost,
                final_cost: initial_cost,
                converged: true,
            };
        }

        let mut cost = initial_cost;
        let mut lambda = config.lambda.max(LAMBDA_MIN);
        let mut iterations = 0usize;
        let mut converged = false;
        let mut trial: Vec<Se3> = poses.clone();

        'outer: while iterations < config.max_iterations {
            iterations += 1;
            let (h, g, _) = build_system(&poses, &self.edges, &pairs, &free, n_free, robust);
            let rhs: Vec<Vec6> = g.iter().map(|v| -v).collect();
            let use_cholesky = h.predicted_block_ops() <= CHOLESKY_BLOCK_OP_BUDGET * n_free;

            // A step is accepted only when it lowers the real robust cost, so
            // the reported cost sequence is monotone by construction.
            let mut relative_gain = None;
            let mut any_solve_succeeded = false;

            for _ in 0..MAX_DAMPING_RETRIES {
                let damped = h.damped_diagonal(lambda);
                let step = if use_cholesky {
                    h.solve_cholesky(&damped, &rhs)
                } else {
                    h.solve_pcg(&damped, &rhs)
                };
                let step = step.filter(|dx| dx.iter().all(|b| b.iter().all(|v| v.is_finite())));

                let Some(dx) = step else {
                    lambda = (lambda * 10.0).min(LAMBDA_MAX);
                    continue;
                };
                any_solve_succeeded = true;

                let step_size = dx.iter().map(|b| b.amax()).fold(0.0f64, f64::max);
                if step_size < STEP_EPSILON {
                    // Nothing left to move: an already-optimal graph is a fixed
                    // point rather than a damping thrash.
                    converged = true;
                    break 'outer;
                }

                trial.copy_from_slice(&poses);
                for (k, slot) in free.iter().enumerate() {
                    if let Some(b) = *slot {
                        trial[k] = poses[k].plus(&dx[b]).normalized();
                    }
                }
                let trial_cost = total_cost(&trial, &self.edges, &pairs, robust);

                if trial_cost < cost {
                    relative_gain = Some((cost - trial_cost) / cost.max(f64::MIN_POSITIVE));
                    poses.copy_from_slice(&trial);
                    cost = trial_cost;
                    lambda = (lambda * 0.1).max(LAMBDA_MIN);
                    break;
                }

                lambda = (lambda * 10.0).min(LAMBDA_MAX);
                if lambda >= LAMBDA_MAX {
                    break;
                }
            }

            match relative_gain {
                Some(gain) if gain >= config.tolerance => {}
                Some(_) => {
                    converged = true;
                    break;
                }
                None => {
                    // No step improved the cost. If the damped system factorised
                    // then this is a minimum to numerical precision; if it never
                    // did, the problem is degenerate and we say so.
                    converged = any_solve_succeeded;
                    break;
                }
            }
        }

        for (node, pose) in self.nodes.iter_mut().zip(&poses) {
            node.pose = *pose;
        }

        log::debug!(
            "pose graph: {} nodes, {} edges, cost {:.6e} -> {:.6e} in {} iterations (converged={})",
            self.nodes.len(),
            self.edges.len(),
            initial_cost,
            cost,
            iterations,
            converged
        );

        SolverReport {
            iterations,
            initial_cost,
            final_cost: cost,
            converged,
        }
    }
}

// --------------------------------------------------------------------------
// Assembly
// --------------------------------------------------------------------------

fn total_cost(
    poses: &[Se3],
    edges: &[Edge],
    pairs: &[Option<(usize, usize)>],
    robust: Option<f64>,
) -> f64 {
    let mut cost = 0.0;
    for (edge, pair) in edges.iter().zip(pairs) {
        let Some((i, j)) = *pair else { continue };
        let r = edge_residual(&poses[i], &poses[j], &edge.measurement);
        let e_sq = r.dot(&(symmetrised(&edge.information) * r));
        cost += 0.5 * huber(e_sq, robust).0;
    }
    cost
}

/// Assemble `H = Σ w J^T Ω J` and `g = Σ w J^T Ω r` over the free nodes.
fn build_system(
    poses: &[Se3],
    edges: &[Edge],
    pairs: &[Option<(usize, usize)>],
    free: &[Option<usize>],
    n_free: usize,
    robust: Option<f64>,
) -> (BlockMatrix, Vec<Vec6>, f64) {
    let mut h = BlockMatrix::zeros(n_free);
    let mut g = vec![Vec6::zeros(); n_free];
    let mut cost = 0.0;

    for (edge, pair) in edges.iter().zip(pairs) {
        let Some((i, j)) = *pair else { continue };
        let omega = symmetrised(&edge.information);
        let (r, j_i, j_j) = edge_jacobians(&poses[i], &poses[j], &edge.measurement);
        let omega_r = omega * r;
        let e_sq = r.dot(&omega_r);
        let (rho, w) = huber(e_sq, robust);
        cost += 0.5 * rho;

        let (a, b) = (free[i], free[j]);
        let jit_omega = j_i.transpose() * omega;
        let jjt_omega = j_j.transpose() * omega;

        if let Some(a) = a {
            h.add_diag(a, &(w * (jit_omega * j_i)));
            g[a] += w * (j_i.transpose() * omega_r);
        }
        if let Some(b) = b {
            h.add_diag(b, &(w * (jjt_omega * j_j)));
            g[b] += w * (j_j.transpose() * omega_r);
        }
        if let (Some(a), Some(b)) = (a, b) {
            // Only the lower triangle is stored, so orient the block by index.
            if a < b {
                h.add_lower(b, a, &(w * (jjt_omega * j_i)));
            } else {
                h.add_lower(a, b, &(w * (jit_omega * j_j)));
            }
        }
    }

    (h, g, cost)
}

// --------------------------------------------------------------------------
// Block-sparse symmetric matrix and the two linear solvers
// --------------------------------------------------------------------------

/// Symmetric block matrix with 6x6 blocks, lower triangle only.
struct BlockMatrix {
    n: usize,
    diag: Vec<Mat6>,
    /// `cols[j][i]` is the block at row `i`, column `j`, with `i > j`.
    cols: Vec<BTreeMap<usize, Mat6>>,
}

impl BlockMatrix {
    fn zeros(n: usize) -> Self {
        BlockMatrix {
            n,
            diag: vec![Mat6::zeros(); n],
            cols: vec![BTreeMap::new(); n],
        }
    }

    fn add_diag(&mut self, j: usize, block: &Mat6) {
        self.diag[j] += block;
    }

    fn add_lower(&mut self, i: usize, j: usize, block: &Mat6) {
        debug_assert!(i > j);
        *self.cols[j].entry(i).or_insert_with(Mat6::zeros) += block;
    }

    /// Marquardt-scaled damping: `H_kk + λ · max(H_kk, floor)` per coordinate.
    /// Scaling by the diagonal rather than by the identity matters here because
    /// the translation and rotation blocks of `Ω` are in different units.
    fn damped_diagonal(&self, lambda: f64) -> Vec<Mat6> {
        self.diag
            .iter()
            .map(|d| {
                let mut m = *d;
                for k in 0..6 {
                    m[(k, k)] += lambda * d[(k, k)].max(DAMP_FLOOR);
                }
                m
            })
            .collect()
    }

    /// Symbolic factorisation: exact block sparsity pattern of `L` under the
    /// natural (insertion) ordering.
    ///
    /// A node's fill-in is absorbed by its parent in the elimination tree — the
    /// smallest row index in its column — and propagates upward from there. For
    /// a keyframe chain plus a few loop closures this leaves `|col_j|` at two
    /// or three regardless of `N`, which is why sparse Cholesky is the right
    /// default for our graphs.
    fn symbolic_pattern(&self) -> Vec<Vec<usize>> {
        let mut pat: Vec<Vec<usize>> = self
            .cols
            .iter()
            .map(|c| c.keys().copied().collect())
            .collect();
        for j in 0..self.n {
            let Some(&parent) = pat[j].first() else {
                continue;
            };
            let promoted: Vec<usize> = pat[j].iter().copied().filter(|&i| i != parent).collect();
            let target = &mut pat[parent];
            for i in promoted {
                if let Err(at) = target.binary_search(&i) {
                    target.insert(at, i);
                }
            }
        }
        pat
    }

    /// Predicted block multiplies for a Cholesky factorisation, `Σ_j |col_j|^2`.
    fn predicted_block_ops(&self) -> usize {
        self.symbolic_pattern()
            .iter()
            .map(|c| c.len().saturating_mul(c.len()))
            .sum()
    }

    /// Right-looking block Cholesky, then block forward/backward substitution.
    ///
    /// Fill-in is discovered during the numeric pass rather than replayed from
    /// the symbolic one: columns are eliminated in increasing order and a
    /// column only ever writes into later columns, so a pattern grown on the
    /// fly is complete by the time it is needed. Returns `None` if a diagonal
    /// block is not positive definite, which LM answers by raising `λ`.
    fn solve_cholesky(&self, damped: &[Mat6], rhs: &[Vec6]) -> Option<Vec<Vec6>> {
        let n = self.n;
        let mut wdiag: Vec<Mat6> = damped.to_vec();
        let mut wcol: Vec<BTreeMap<usize, Mat6>> = self.cols.clone();

        // `l_col[j]` = the below-diagonal blocks of column j; `l_inv[j]` = Ljj^-1.
        let mut l_col: Vec<Vec<(usize, Mat6)>> = vec![Vec::new(); n];
        let mut l_inv: Vec<Mat6> = vec![Mat6::zeros(); n];

        for j in 0..n {
            let ljj = wdiag[j].cholesky()?.l();
            let ljj_inv = ljj.try_inverse()?;
            let ljj_inv_t = ljj_inv.transpose();
            l_inv[j] = ljj_inv;

            let col: Vec<(usize, Mat6)> = wcol[j]
                .iter()
                .map(|(&i, block)| (i, block * ljj_inv_t))
                .collect();

            for a in 0..col.len() {
                let (i, li) = col[a];
                let li_t = li.transpose();
                wdiag[i] -= li * li_t;
                // BTreeMap keys ascend, so every k here is > i.
                for &(k, lk) in &col[a + 1..] {
                    *wcol[i].entry(k).or_insert_with(Mat6::zeros) -= lk * li_t;
                }
            }
            l_col[j] = col;
        }

        // L y = rhs
        let mut x = rhs.to_vec();
        for j in 0..n {
            x[j] = l_inv[j] * x[j];
            for &(i, lij) in &l_col[j] {
                let update = lij * x[j];
                x[i] -= update;
            }
        }
        // L^T x = y
        for j in (0..n).rev() {
            for &(i, lij) in &l_col[j] {
                let update = lij.transpose() * x[i];
                x[j] -= update;
            }
            x[j] = l_inv[j].transpose() * x[j];
        }
        Some(x)
    }

    /// `y = (offdiag(H) + damped) x`, using both triangles of each stored block.
    fn multiply(&self, damped: &[Mat6], x: &[Vec6]) -> Vec<Vec6> {
        let mut y: Vec<Vec6> = (0..self.n).map(|i| damped[i] * x[i]).collect();
        for (j, col) in self.cols.iter().enumerate() {
            for (&i, block) in col {
                let lower = block * x[j];
                let upper = block.transpose() * x[i];
                y[i] += lower;
                y[j] += upper;
            }
        }
        y
    }

    /// Preconditioned conjugate gradient with a block-Jacobi preconditioner.
    ///
    /// The fallback when predicted fill makes Cholesky the wrong shape. Each
    /// iteration costs one sparse matvec, so it stays linear in the edge count
    /// no matter how tangled the covisibility graph gets.
    fn solve_pcg(&self, damped: &[Mat6], rhs: &[Vec6]) -> Option<Vec<Vec6>> {
        let n = self.n;
        let mut m_inv = Vec::with_capacity(n);
        for d in damped {
            m_inv.push(d.cholesky()?.inverse());
        }

        let mut x = vec![Vec6::zeros(); n];
        let mut r = rhs.to_vec();
        let b_norm = r.iter().map(|v| v.norm_squared()).sum::<f64>().sqrt();
        if b_norm == 0.0 {
            return Some(x);
        }
        let target = PCG_RELATIVE_TOLERANCE * b_norm;

        let mut z: Vec<Vec6> = (0..n).map(|i| m_inv[i] * r[i]).collect();
        let mut p = z.clone();
        let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a.dot(b)).sum();

        for _ in 0..(6 * n).min(PCG_MAX_ITERATIONS) {
            let ap = self.multiply(damped, &p);
            let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a.dot(b)).sum();
            if pap <= 0.0 || !pap.is_finite() {
                // Lost positive definiteness; hand back whatever we have and let
                // LM judge it on the actual cost.
                break;
            }
            let alpha = rz / pap;
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            if r.iter().map(|v| v.norm_squared()).sum::<f64>().sqrt() <= target {
                break;
            }
            for i in 0..n {
                z[i] = m_inv[i] * r[i];
            }
            let rz_next: f64 = r.iter().zip(&z).map(|(a, b)| a.dot(b)).sum();
            let beta = rz_next / rz;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rz = rz_next;
        }
        Some(x)
    }
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use wslam_core::DeterministicRng;

    fn kf(i: u64) -> KeyframeId {
        KeyframeId(i)
    }

    fn se3(tx: f64, ty: f64, tz: f64, rx: f64, ry: f64, rz: f64) -> Se3 {
        Se3::new(So3::exp(&Vec3::new(rx, ry, rz)), Vec3::new(tx, ty, tz))
    }

    fn yaw(angle: f64) -> Se3 {
        Se3::from_rotation(So3::exp(&Vec3::new(0.0, 0.0, angle)))
    }

    fn info(w: f64) -> Mat6 {
        Mat6::identity() * w
    }

    /// Max over nodes of the SE(3) distance between estimate and truth.
    fn max_pose_error(graph: &PoseGraph, truth: &[Se3]) -> f64 {
        truth
            .iter()
            .enumerate()
            .map(|(i, t)| graph.pose(kf(i as u64)).unwrap().minus(t).amax())
            .fold(0.0f64, f64::max)
    }

    /// RMS over nodes of the SE(3) distance between estimate and truth.
    fn rms_pose_error(graph: &PoseGraph, truth: &[Se3]) -> f64 {
        let sum: f64 = truth
            .iter()
            .enumerate()
            .map(|(i, t)| graph.pose(kf(i as u64)).unwrap().minus(t).norm_squared())
            .sum();
        (sum / truth.len() as f64).sqrt()
    }

    fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
        assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (tol {tol})");
    }

    /// A closed square loop in the plane: N poses, heading tangent to the path,
    /// returning exactly to the start.
    fn square_loop(n: usize, side: f64) -> Vec<Se3> {
        assert!(n.is_multiple_of(4));
        let per_side = n / 4;
        let step = side / per_side as f64;
        let mut poses = Vec::with_capacity(n);
        let mut t = Vec3::zeros();
        for k in 0..n {
            let leg = k / per_side;
            let heading = std::f64::consts::FRAC_PI_2 * leg as f64;
            poses.push(Se3::new(So3::exp(&Vec3::new(0.0, 0.0, heading)), t));
            let dir = Vec3::new(heading.cos(), heading.sin(), 0.0);
            t += dir * step;
        }
        poses
    }

    /// Odometry + one closure, all measurements exact w.r.t. `truth`.
    fn exact_loop_graph(truth: &[Se3], odom_info: f64, loop_info: f64) -> PoseGraph {
        let n = truth.len();
        let mut g = PoseGraph::new();
        for (i, t) in truth.iter().enumerate() {
            g.add_node(kf(i as u64), *t, i == 0);
        }
        for i in 0..n - 1 {
            g.add_edge(
                kf(i as u64),
                kf(i as u64 + 1),
                truth[i].inverse().compose(&truth[i + 1]),
                info(odom_info),
            );
        }
        g.add_edge(
            kf(n as u64 - 1),
            kf(0),
            truth[n - 1].inverse().compose(&truth[0]),
            info(loop_info),
        );
        g
    }

    // ---- Lie algebra -----------------------------------------------------

    #[test]
    fn se3_right_jacobian_inv_matches_central_differences() {
        // Defining identity: log(exp(xi) exp(d)) = xi + Jr^-1(xi) d + O(d^2).
        let h = 1e-6;
        for xi in [
            Vec6::new(0.4, -1.2, 0.9, 0.3, -0.15, 0.55),
            Vec6::new(-2.0, 0.5, 3.0, 1.1, 0.7, -0.4),
            Vec6::new(0.1, 0.2, -0.05, 1e-4, -2e-4, 5e-5),
            Vec6::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            // Straddle Q_SMALL_ANGLE with a large rho, where a wrong Taylor
            // branch would show up and a small rho would hide it.
            Vec6::new(1.5, -2.0, 0.8, 0.0, 0.0, Q_SMALL_ANGLE * 0.99),
            Vec6::new(1.5, -2.0, 0.8, 0.0, 0.0, Q_SMALL_ANGLE * 1.01),
        ] {
            let analytic = se3_right_jacobian_inv(&xi);
            let base = Se3::exp(&xi);
            for k in 0..6 {
                let mut d = Vec6::zeros();
                d[k] = h;
                let plus = base.compose(&Se3::exp(&d)).log();
                let minus = base.compose(&Se3::exp(&(-d))).log();
                let numeric = (plus - minus) / (2.0 * h);
                let err = (numeric - analytic.column(k)).amax();
                assert!(err < 1e-7, "xi={xi:?} column {k} error {err}");
            }
        }
    }

    #[test]
    fn se3_left_jacobian_matches_the_adjoint_integral() {
        // Independent reference for Barfoot's Q: for any Lie group,
        // `Jl(xi) = ∫_0^1 Adj(exp(s·xi)) ds`. Integrating that numerically
        // derives the whole 6x6 left Jacobian from `Se3::exp` and `Se3::adjoint`
        // alone, so it shares no algebra with `se3_q` — including the Taylor
        // and closed-form branches, which the sweep straddles.
        let integral = |xi: &Vec6| {
            let steps = 2000usize; // composite Simpson
            let mut acc = Mat6::zeros();
            for k in 0..=steps {
                let s = k as f64 / steps as f64;
                let w = if k == 0 || k == steps {
                    1.0
                } else if k % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                acc += w * Se3::exp(&(xi * s)).adjoint();
            }
            acc / (3.0 * steps as f64)
        };

        let axis = Vec3::new(0.3, -0.5, 0.81).normalize();
        for &theta in &[
            0.0,
            1e-6,
            Q_SMALL_ANGLE * 0.5,
            Q_SMALL_ANGLE * 0.999,
            Q_SMALL_ANGLE * 1.001,
            0.1,
            0.9,
            2.5,
        ] {
            let phi = axis * theta;
            let xi = Vec6::new(0.7, -1.3, 0.2, phi.x, phi.y, phi.z);
            let analytic = se3_left_jacobian_inv(&xi)
                .try_inverse()
                .expect("Jl is invertible away from 2*pi");
            let err = (analytic - integral(&xi)).amax();
            assert!(err < 1e-9, "theta={theta}: |Jl - integral| = {err}");
        }
    }

    #[test]
    fn analytic_edge_jacobians_match_central_differences() {
        let mut rng = DeterministicRng::new("posegraph-jacobian", 0xC0FFEE);
        let h = 1e-6;
        for trial in 0..40 {
            let mut sample = |scale: f64| {
                Vec6::new(
                    rng.uniform_range(-scale, scale),
                    rng.uniform_range(-scale, scale),
                    rng.uniform_range(-scale, scale),
                    rng.uniform_range(-0.6, 0.6),
                    rng.uniform_range(-0.6, 0.6),
                    rng.uniform_range(-0.6, 0.6),
                )
            };
            let t_i = Se3::exp(&sample(2.0));
            let t_j = Se3::exp(&sample(2.0));
            // A measurement close-but-not-equal to the true relative pose, so
            // the residual is O(0.1) — big enough that the Q block matters,
            // small enough to stay far from the log's pi singularity.
            let z = t_i
                .inverse()
                .compose(&t_j)
                .compose(&Se3::exp(&(sample(0.15) * 0.3)));

            let (_, j_i, j_j) = edge_jacobians(&t_i, &t_j, &z);
            for k in 0..6 {
                let mut d = Vec6::zeros();
                d[k] = h;
                let num_i = (edge_residual(&t_i.plus(&d), &t_j, &z)
                    - edge_residual(&t_i.plus(&(-d)), &t_j, &z))
                    / (2.0 * h);
                let num_j = (edge_residual(&t_i, &t_j.plus(&d), &z)
                    - edge_residual(&t_i, &t_j.plus(&(-d)), &z))
                    / (2.0 * h);
                assert!(
                    (num_i - j_i.column(k)).amax() < 1e-6,
                    "trial {trial} d/ddi column {k}: {} vs {}",
                    num_i,
                    j_i.column(k)
                );
                assert!(
                    (num_j - j_j.column(k)).amax() < 1e-6,
                    "trial {trial} d/ddj column {k}: {} vs {}",
                    num_j,
                    j_j.column(k)
                );
            }
        }
    }

    #[test]
    fn residual_vanishes_exactly_on_a_consistent_measurement() {
        let t_i = se3(1.0, -2.0, 0.5, 0.2, -0.1, 0.4);
        let t_j = se3(-0.3, 1.1, 2.0, -0.5, 0.3, 0.1);
        let z = t_i.inverse().compose(&t_j);
        assert!(edge_residual(&t_i, &t_j, &z).amax() < 1e-14);
    }

    // ---- Fixed points and closed forms -----------------------------------

    #[test]
    fn perfect_chain_is_a_fixed_point() {
        let truth: Vec<Se3> = (0..12)
            .map(|k| {
                let f = k as f64;
                se3(
                    0.4 * f,
                    0.1 * f * f,
                    -0.05 * f,
                    0.02 * f,
                    -0.01 * f,
                    0.03 * f,
                )
            })
            .collect();

        let mut g = PoseGraph::new();
        for (i, t) in truth.iter().enumerate() {
            g.add_node(kf(i as u64), *t, i == 0);
        }
        for i in 0..truth.len() - 1 {
            g.add_edge(
                kf(i as u64),
                kf(i as u64 + 1),
                truth[i].inverse().compose(&truth[i + 1]),
                info(1.0),
            );
        }

        let report = g.optimize(&SolverConfig::default());
        assert!(report.initial_cost < 1e-24, "cost {}", report.initial_cost);
        assert!(report.converged);
        assert!(report.final_cost <= report.initial_cost);
        assert!(
            max_pose_error(&g, &truth) < 1e-12,
            "the optimizer moved an already-optimal graph by {}",
            max_pose_error(&g, &truth)
        );
    }

    #[test]
    fn two_conflicting_edges_land_on_the_information_weighted_mean() {
        // With identity rotations and collinear translations the SE(3)
        // residual is exactly linear: r = t - a. So the minimiser of
        // w1|t-a|^2 + w2|t-b|^2 is the closed form (w1 a + w2 b)/(w1 + w2).
        let (a, b, w1, w2) = (1.0, 2.0, 3.0, 7.0);
        let mut g = PoseGraph::new();
        g.add_node(kf(0), Se3::identity(), true);
        g.add_node(
            kf(1),
            Se3::from_translation(Vec3::new(0.0, 0.0, 0.0)),
            false,
        );
        g.add_edge(
            kf(0),
            kf(1),
            Se3::from_translation(Vec3::new(a, 0.0, 0.0)),
            info(w1),
        );
        g.add_edge(
            kf(0),
            kf(1),
            Se3::from_translation(Vec3::new(b, 0.0, 0.0)),
            info(w2),
        );

        // Robustification off: this test is about the quadratic optimum.
        let report = g.optimize(&SolverConfig {
            huber_delta: 0.0,
            ..Default::default()
        });
        let expected = (w1 * a + w2 * b) / (w1 + w2);
        let got = g.pose(kf(1)).unwrap().translation().x;
        assert_close(got, expected, 1e-10, "weighted mean");

        // And the residual cost at that point is the closed-form value too.
        let e1 = w1 * (expected - a).powi(2);
        let e2 = w2 * (expected - b).powi(2);
        assert_close(report.final_cost, 0.5 * (e1 + e2), 1e-10, "final cost");
    }

    #[test]
    fn loop_closure_recovers_the_known_trajectory() {
        // Measurements exact, initial guess dead-reckoned with drift. The global
        // minimum is therefore the ground truth itself, so this is a recovery
        // test with a closed-form answer rather than a "cost went down" test.
        let truth = square_loop(16, 4.0);
        let mut g = exact_loop_graph(&truth, 1.0, 1.0);

        let drift = se3(0.06, -0.05, 0.02, 0.008, -0.006, 0.04);
        let mut guess = truth[0];
        for i in 1..truth.len() {
            let rel = truth[i - 1].inverse().compose(&truth[i]);
            guess = guess.compose(&rel).compose(&drift);
            g.add_node(kf(i as u64), guess, false);
        }
        let before = max_pose_error(&g, &truth);
        assert!(
            before > 0.5,
            "the drifted guess should be badly wrong: {before}"
        );

        let report = g.optimize(&SolverConfig::default());
        assert!(report.initial_cost > 1.0, "cost {}", report.initial_cost);
        assert!(
            report.final_cost < report.initial_cost * 1e-12,
            "{} -> {}",
            report.initial_cost,
            report.final_cost
        );
        assert!(
            max_pose_error(&g, &truth) < 1e-8,
            "recovered error {}",
            max_pose_error(&g, &truth)
        );
    }

    #[test]
    fn loop_closure_bounds_drift_from_corrupted_odometry() {
        // The spec.md §4 L4b claim, tested directly: odometry is corrupted by a
        // known per-step drift so dead reckoning accumulates a large loop-close
        // gap; a single verified closure with high information has to bound it.
        let truth = square_loop(16, 4.0);
        let n = truth.len();
        let drift = se3(0.02, -0.01, 0.005, 0.002, -0.001, 0.01);

        let mut g = PoseGraph::new();
        g.add_node(kf(0), truth[0], true);
        let mut guess = truth[0];
        let mut odom = Vec::new();
        for i in 1..n {
            let z = truth[i - 1].inverse().compose(&truth[i]).compose(&drift);
            odom.push(z);
            guess = guess.compose(&z);
            g.add_node(kf(i as u64), guess, false);
        }
        for (i, z) in odom.iter().enumerate() {
            g.add_edge(kf(i as u64), kf(i as u64 + 1), *z, info(1.0));
        }

        let last_id = kf(n as u64 - 1);
        let open_end = g.pose(last_id).unwrap().minus(&truth[n - 1]).amax();
        let open_rms = rms_pose_error(&g, &truth);

        // The closure is a geometrically verified measurement, so it is trusted
        // far more than any single odometry step.
        g.add_edge(
            last_id,
            kf(0),
            truth[n - 1].inverse().compose(&truth[0]),
            info(1e4),
        );

        let report = g.optimize(&SolverConfig::default());
        let closed_end = g.pose(last_id).unwrap().minus(&truth[n - 1]).amax();
        let closed_rms = rms_pose_error(&g, &truth);

        assert!(
            report.final_cost < report.initial_cost * 1e-2,
            "cost {} -> {}",
            report.initial_cost,
            report.final_cost
        );
        // End-of-trajectory drift is what "bounds global drift" means: open-loop
        // it grows without limit, closed it is pinned to the closure accuracy.
        assert!(
            closed_end < open_end / 10.0,
            "closure failed to bound end-of-loop drift: {open_end} -> {closed_end}"
        );
        // Mid-loop nodes cannot reach ground truth — every odometry measurement
        // is genuinely biased, so the MAP estimate spreads the disagreement
        // around the loop. Halving the RMS is the correct amount of help, not a
        // weak result.
        assert!(
            closed_rms < open_rms / 2.0,
            "trajectory RMS barely improved: {open_rms} -> {closed_rms}"
        );

        // The loop itself must actually be closed.
        let gap = g
            .pose(last_id)
            .unwrap()
            .compose(&truth[n - 1].inverse().compose(&truth[0]))
            .minus(&g.pose(kf(0)).unwrap())
            .amax();
        assert!(gap < 5e-3, "loop remains open by {gap}");
    }

    #[test]
    fn pure_yaw_drift_is_removed_by_closure() {
        // spec.md §4 L4b names yaw: L1 is drift-free in roll/pitch and L3 only
        // arrests yaw locally, so yaw is the axis the pose graph has to own.
        let truth = square_loop(16, 4.0);
        let n = truth.len();
        let per_step_yaw = 0.01;
        let mut g = PoseGraph::new();
        g.add_node(kf(0), truth[0], true);

        let mut guess = truth[0];
        let mut odom = Vec::new();
        for i in 1..n {
            let z = truth[i - 1]
                .inverse()
                .compose(&truth[i])
                .compose(&yaw(per_step_yaw));
            odom.push(z);
            guess = guess.compose(&z);
            g.add_node(kf(i as u64), guess, false);
        }
        for (i, z) in odom.iter().enumerate() {
            g.add_edge(kf(i as u64), kf(i as u64 + 1), *z, info(1.0));
        }

        let open_yaw = g
            .pose(kf(n as u64 - 1))
            .unwrap()
            .rotation()
            .minus(&truth[n - 1].rotation())
            .z
            .abs();
        // Dead reckoning accumulates one drift per step.
        assert_close(
            open_yaw,
            per_step_yaw * (n - 1) as f64,
            1e-6,
            "open-loop yaw",
        );

        g.add_edge(
            kf(n as u64 - 1),
            kf(0),
            truth[n - 1].inverse().compose(&truth[0]),
            info(1e6),
        );
        g.optimize(&SolverConfig::default());

        let closed_yaw = g
            .pose(kf(n as u64 - 1))
            .unwrap()
            .rotation()
            .minus(&truth[n - 1].rotation())
            .z
            .abs();
        assert!(
            closed_yaw < open_yaw / 20.0,
            "yaw drift survived closure: {open_yaw} -> {closed_yaw}"
        );
    }

    #[test]
    fn optimum_is_invariant_to_the_global_gauge() {
        // Transform every node and the anchor by one rigid G; the optimised
        // relative geometry must be identical. A solver that leaked a world-frame
        // assumption would fail this.
        let truth = square_loop(12, 3.0);
        let build = |g_world: Se3| {
            let mut g = exact_loop_graph(&truth, 1.0, 1.0);
            let mut guess = truth[0];
            let drift = se3(0.02, 0.01, -0.01, 0.003, 0.002, 0.01);
            for i in 0..truth.len() {
                if i > 0 {
                    let rel = truth[i - 1].inverse().compose(&truth[i]);
                    guess = guess.compose(&rel).compose(&drift);
                }
                g.add_node(kf(i as u64), g_world.compose(&guess), i == 0);
            }
            g.optimize(&SolverConfig::default());
            g
        };

        let plain = build(Se3::identity());
        let shifted = build(se3(5.0, -3.0, 7.0, 0.3, -0.9, 1.2));

        for i in 1..truth.len() {
            let a = plain
                .pose(kf(0))
                .unwrap()
                .inverse()
                .compose(&plain.pose(kf(i as u64)).unwrap());
            let b = shifted
                .pose(kf(0))
                .unwrap()
                .inverse()
                .compose(&shifted.pose(kf(i as u64)).unwrap());
            assert!(a.minus(&b).amax() < 1e-8, "node {i} gauge-dependent");
        }
    }

    // ---- Robustness ------------------------------------------------------

    #[test]
    fn huber_bounds_the_damage_from_a_false_positive_loop_closure() {
        // spec.md §5: a false-positive closure corrupts the map "irrecoverably"
        // and is "worse than no loop closure at all". Geometric verification is
        // the first line of defence; this kernel is the second.
        //
        // The property being tested is *bounded influence*, which is stronger
        // and much more informative than any single error threshold: Huber caps
        // an edge's gradient contribution at `delta * sqrt(Ω)` regardless of how
        // wrong it is, so the damage saturates. Least squares has no such cap
        // and the damage grows with the blunder — which is precisely what
        // "irrecoverably" means. We therefore run the same graph with a 3 m and
        // a 30 m blunder and compare how the two solvers scale.
        let truth = square_loop(24, 6.0);
        let n = truth.len();
        // sigma = 1 cm / 10 mrad, so the Huber threshold sits in real sigma
        // units and a metre-scale blunder is hundreds of sigma out. With a toy
        // `information` of 1 the kernel could never fire at all.
        let edge_info = info(1e4);

        let build = |huber_delta: f64, blunder: f64| {
            let mut g = PoseGraph::new();
            let mut guess = truth[0];
            let drift = se3(0.002, -0.001, 0.0005, 0.0002, -0.00015, 0.001);
            for i in 0..n {
                if i > 0 {
                    guess = guess
                        .compose(&truth[i - 1].inverse().compose(&truth[i]))
                        .compose(&drift);
                }
                g.add_node(kf(i as u64), guess, i == 0);
            }
            // Many good constraints: sequential odometry plus every skip-2 pair.
            for i in 0..n - 1 {
                g.add_edge(
                    kf(i as u64),
                    kf(i as u64 + 1),
                    truth[i].inverse().compose(&truth[i + 1]),
                    edge_info,
                );
            }
            for i in 0..n - 2 {
                g.add_edge(
                    kf(i as u64),
                    kf(i as u64 + 2),
                    truth[i].inverse().compose(&truth[i + 2]),
                    edge_info,
                );
            }
            g.add_edge(
                kf(n as u64 - 1),
                kf(0),
                truth[n - 1].inverse().compose(&truth[0]),
                edge_info,
            );
            // The survivor of geometric verification: a closure between two
            // nodes that are genuinely metres and half a radian apart.
            let bogus = truth[3].inverse().compose(&truth[15]).compose(&se3(
                blunder,
                -blunder * 0.66,
                blunder * 0.33,
                0.0,
                0.0,
                0.5,
            ));
            g.add_edge(kf(3), kf(15), bogus, edge_info);

            let report = g.optimize(&SolverConfig {
                max_iterations: 200,
                huber_delta,
                ..Default::default()
            });
            (max_pose_error(&g, &truth), report)
        };

        let delta = SolverConfig::default().huber_delta;
        let (robust_small, small_report) = build(delta, 3.0);
        let (robust_large, large_report) = build(delta, 30.0);
        let (naive_small, _) = build(0.0, 3.0);
        let (naive_large, _) = build(0.0, 30.0);

        assert!(small_report.converged && large_report.converged);

        // Bounded influence: a 10x worse blunder does essentially no extra
        // damage. The plateau sits near `delta / sqrt(Ω)` = 3.55 / 100 = 3.5 cm
        // of pull per adjacent constraint, spread around the loop.
        assert!(
            robust_large < 1.5 * robust_small,
            "Huber influence was not bounded: 3 m blunder -> {robust_small}, \
             30 m blunder -> {robust_large}"
        );
        assert!(robust_large < 0.35, "robust error {robust_large}");

        // Unbounded influence: least squares damage tracks the blunder.
        assert!(
            naive_large > 5.0 * naive_small,
            "without robustification the damage should scale with the blunder: \
             {naive_small} -> {naive_large}"
        );
        // And the head-to-head at the same blunder, which is the comparison the
        // spec's "worse than no loop closure at all" is about.
        assert!(
            naive_large > 30.0 * robust_large,
            "no robustification was not materially worse ({naive_large} vs {robust_large}); \
             the comparison is the test"
        );
        assert!(naive_small > 5.0 * robust_small);
    }

    #[test]
    fn huber_leaves_inlier_only_graphs_alone() {
        // The kernel must be inert when nothing is an outlier, otherwise it is
        // silently biasing every ordinary solve.
        let truth = square_loop(12, 3.0);
        let solve = |huber_delta: f64| {
            let mut g = exact_loop_graph(&truth, 1.0, 1.0);
            let drift = se3(0.01, -0.005, 0.002, 0.001, 0.0, 0.004);
            let mut guess = truth[0];
            for i in 1..truth.len() {
                guess = guess
                    .compose(&truth[i - 1].inverse().compose(&truth[i]))
                    .compose(&drift);
                g.add_node(kf(i as u64), guess, false);
            }
            g.optimize(&SolverConfig {
                huber_delta,
                ..Default::default()
            });
            g
        };
        let a = solve(SolverConfig::default().huber_delta);
        let b = solve(0.0);
        for i in 0..truth.len() {
            let d = a
                .pose(kf(i as u64))
                .unwrap()
                .minus(&b.pose(kf(i as u64)).unwrap())
                .amax();
            assert!(d < 1e-7, "node {i} differs by {d} with and without Huber");
        }
    }

    // ---- Degenerate cases the spec names ---------------------------------

    #[test]
    fn unfixed_graph_is_anchored_and_stays_finite() {
        // 6-DoF null space; without an anchor the normal equations are singular.
        let truth = square_loop(8, 2.0);
        let mut g = PoseGraph::new();
        let drift = se3(0.05, -0.03, 0.02, 0.005, 0.004, 0.03);
        let mut guess = truth[0];
        for i in 0..truth.len() {
            if i > 0 {
                guess = guess
                    .compose(&truth[i - 1].inverse().compose(&truth[i]))
                    .compose(&drift);
            }
            g.add_node(kf(i as u64), guess, false); // nothing fixed
        }
        for i in 0..truth.len() - 1 {
            g.add_edge(
                kf(i as u64),
                kf(i as u64 + 1),
                truth[i].inverse().compose(&truth[i + 1]),
                info(1.0),
            );
        }
        g.add_edge(
            kf(truth.len() as u64 - 1),
            kf(0),
            truth[truth.len() - 1].inverse().compose(&truth[0]),
            info(1.0),
        );

        let anchor_before = g.pose(kf(0)).unwrap();
        let report = g.optimize(&SolverConfig::default());
        assert!(report.initial_cost.is_finite() && report.final_cost.is_finite());
        assert!(report.final_cost < report.initial_cost * 1e-10);
        for (_, pose, _) in g.nodes() {
            assert!(pose.translation().iter().all(|v| v.is_finite()));
            assert!(pose.rotation().log().iter().all(|v| v.is_finite()));
        }
        // The anchored node is node 0 and it did not move.
        assert!(g.pose(kf(0)).unwrap().minus(&anchor_before).amax() < 1e-14);
        // Relative geometry still matches truth even though the world frame is
        // whatever the drifted node 0 happened to be.
        let rel = g
            .pose(kf(0))
            .unwrap()
            .inverse()
            .compose(&g.pose(kf(4)).unwrap());
        let rel_truth = truth[0].inverse().compose(&truth[4]);
        assert!(rel.minus(&rel_truth).amax() < 1e-8);
    }

    #[test]
    fn all_nodes_fixed_is_a_no_op() {
        let mut g = PoseGraph::new();
        g.add_node(kf(0), Se3::identity(), true);
        g.add_node(kf(1), Se3::from_translation(Vec3::new(5.0, 0.0, 0.0)), true);
        g.add_edge(
            kf(0),
            kf(1),
            Se3::from_translation(Vec3::new(1.0, 0.0, 0.0)),
            info(1.0),
        );
        let report = g.optimize(&SolverConfig::default());
        assert_eq!(report.iterations, 0);
        assert!(report.converged);
        // Cost is real (the edge is badly violated) but nothing can move.
        assert!(report.final_cost > 0.0);
        assert_close(
            g.pose(kf(1)).unwrap().translation().x,
            5.0,
            1e-15,
            "fixed node moved",
        );
    }

    #[test]
    fn empty_and_singleton_graphs_do_not_panic() {
        let mut empty = PoseGraph::new();
        let r = empty.optimize(&SolverConfig::default());
        assert_eq!(r.iterations, 0);
        assert!(r.converged);
        assert!(empty.is_empty());

        let mut one = PoseGraph::new();
        one.add_node(kf(7), se3(1.0, 2.0, 3.0, 0.1, 0.2, 0.3), false);
        let r = one.optimize(&SolverConfig::default());
        assert!(r.converged);
        assert_eq!(r.initial_cost, 0.0);
        assert!(one.pose(kf(7)).is_some());
        assert!(one.pose(kf(8)).is_none());
    }

    #[test]
    fn dangling_and_self_edges_are_skipped_not_fatal() {
        let mut g = PoseGraph::new();
        g.add_node(kf(0), Se3::identity(), true);
        g.add_node(
            kf(1),
            Se3::from_translation(Vec3::new(0.5, 0.0, 0.0)),
            false,
        );
        // Endpoint 99 never added.
        g.add_edge(
            kf(1),
            kf(99),
            Se3::from_translation(Vec3::new(1.0, 0.0, 0.0)),
            info(1.0),
        );
        // Self-edge: Jacobians cancel exactly, so it constrains nothing.
        g.add_edge(
            kf(1),
            kf(1),
            Se3::from_translation(Vec3::new(9.0, 0.0, 0.0)),
            info(1.0),
        );
        g.add_edge(
            kf(0),
            kf(1),
            Se3::from_translation(Vec3::new(2.0, 0.0, 0.0)),
            info(1.0),
        );

        let report = g.optimize(&SolverConfig::default());
        assert_eq!(
            g.edge_count(),
            3,
            "edges() keeps skipped edges for the debug surface"
        );
        assert!(report.final_cost < 1e-18, "cost {}", report.final_cost);
        assert_close(
            g.pose(kf(1)).unwrap().translation().x,
            2.0,
            1e-9,
            "solved x",
        );
    }

    #[test]
    fn re_adding_a_node_updates_it() {
        let mut g = PoseGraph::new();
        g.add_node(kf(4), Se3::identity(), false);
        g.add_node(kf(4), Se3::from_translation(Vec3::new(1.0, 2.0, 3.0)), true);
        assert_eq!(g.node_count(), 1);
        assert_close(
            g.pose(kf(4)).unwrap().translation().y,
            2.0,
            0.0,
            "updated pose",
        );
        assert!(g.nodes().next().unwrap().2, "updated fixed flag");
    }

    #[test]
    fn zero_information_edge_contributes_nothing() {
        let mut g = PoseGraph::new();
        g.add_node(kf(0), Se3::identity(), true);
        g.add_node(
            kf(1),
            Se3::from_translation(Vec3::new(4.0, 0.0, 0.0)),
            false,
        );
        g.add_edge(
            kf(0),
            kf(1),
            Se3::from_translation(Vec3::new(1.0, 0.0, 0.0)),
            Mat6::zeros(),
        );
        let report = g.optimize(&SolverConfig::default());
        assert_eq!(report.initial_cost, 0.0);
        assert_close(
            g.pose(kf(1)).unwrap().translation().x,
            4.0,
            1e-12,
            "unmoved",
        );
    }

    #[test]
    fn cost_never_increases() {
        let truth = square_loop(20, 5.0);
        let mut rng = DeterministicRng::new("posegraph-monotone", 4242);
        let mut g = exact_loop_graph(&truth, 1.0, 1.0);
        for (i, t) in truth.iter().enumerate().skip(1) {
            let noise = Vec6::from_fn(|k, _| {
                if k < 3 {
                    rng.normal_with(0.0, 0.2)
                } else {
                    rng.normal_with(0.0, 0.05)
                }
            });
            g.add_node(kf(i as u64), t.plus(&noise), false);
        }
        for delta in [0.0, 0.5, 3.5485, f64::INFINITY] {
            let mut h = PoseGraph::new();
            for (id, pose, fixed) in g.nodes() {
                h.add_node(id, pose, fixed);
            }
            for e in g.edges() {
                h.add_edge(
                    KeyframeId(e.from.0),
                    KeyframeId(e.to.0),
                    e.measurement,
                    e.information,
                );
            }
            let report = h.optimize(&SolverConfig {
                huber_delta: delta,
                ..Default::default()
            });
            assert!(
                report.final_cost <= report.initial_cost,
                "delta {delta}: {} -> {}",
                report.initial_cost,
                report.final_cost
            );
            assert!(report.final_cost.is_finite());
        }
    }

    // ---- Linear algebra internals ----------------------------------------

    #[test]
    fn sparse_cholesky_and_pcg_agree_on_the_same_system() {
        // Two independent solvers on one synthetic SPD block system: if they
        // agree to 1e-9 neither is silently wrong about the sparsity pattern.
        let mut rng = DeterministicRng::new("posegraph-linalg", 77);
        let n = 9;
        let mut h = BlockMatrix::zeros(n);
        for j in 0..n {
            let a = Mat6::from_fn(|_, _| rng.uniform_range(-1.0, 1.0));
            h.add_diag(j, &(a.transpose() * a + Mat6::identity() * 6.0));
        }
        // A chain plus two long-range blocks, i.e. the pose-graph shape.
        for j in 0..n - 1 {
            let b = Mat6::from_fn(|_, _| rng.uniform_range(-0.3, 0.3));
            h.add_lower(j + 1, j, &b);
        }
        h.add_lower(
            n - 1,
            0,
            &Mat6::from_fn(|_, _| rng.uniform_range(-0.2, 0.2)),
        );
        h.add_lower(
            n - 2,
            2,
            &Mat6::from_fn(|_, _| rng.uniform_range(-0.2, 0.2)),
        );

        let rhs: Vec<Vec6> = (0..n)
            .map(|_| Vec6::from_fn(|_, _| rng.uniform_range(-1.0, 1.0)))
            .collect();
        let damped = h.damped_diagonal(0.0);

        let chol = h.solve_cholesky(&damped, &rhs).expect("SPD");
        let pcg = h.solve_pcg(&damped, &rhs).expect("SPD");

        for i in 0..n {
            assert!(
                (chol[i] - pcg[i]).amax() < 1e-9,
                "block {i}: {} vs {}",
                chol[i],
                pcg[i]
            );
        }
        // And the Cholesky solution really solves the system.
        let residual = h.multiply(&damped, &chol);
        for i in 0..n {
            assert!((residual[i] - rhs[i]).amax() < 1e-9, "residual block {i}");
        }
    }

    #[test]
    fn symbolic_fill_stays_sparse_for_chain_plus_closures() {
        // The claim behind CHOLESKY_BLOCK_OP_BUDGET. A bare chain factors with
        // no fill at all; each long-range closure adds at most one extra
        // frontal column over its span, so fill is O(n·L), not O(n^2).
        let n = 200;
        let chain = |extra: usize| {
            let mut h = BlockMatrix::zeros(n);
            for j in 0..n {
                h.add_diag(j, &Mat6::identity());
            }
            for j in 0..n - 1 {
                h.add_lower(j + 1, j, &Mat6::identity());
            }
            for k in 0..extra {
                h.add_lower(n - 1 - k, k * 3, &Mat6::identity());
            }
            h
        };

        let bare = chain(0);
        let fill_bare: usize = bare.symbolic_pattern().iter().map(Vec::len).sum();
        assert_eq!(fill_bare, n - 1, "a chain must not fill in at all");

        let looped = chain(8);
        let fill: usize = looped.symbolic_pattern().iter().map(Vec::len).sum();
        let dense = n * (n - 1) / 2;
        assert!(fill < dense / 10, "fill {fill} vs dense {dense} for n={n}");
        assert!(
            looped.predicted_block_ops() <= CHOLESKY_BLOCK_OP_BUDGET * n,
            "a keyframe chain with 8 closures must stay on the Cholesky side"
        );
    }

    #[test]
    fn dense_graph_falls_back_to_pcg_and_still_solves() {
        // Every node connected to every other: predicted fill is cubic in the
        // block count, so the crossover must pick PCG, and PCG must still
        // deliver a step.
        let n = 40;
        let mut h = BlockMatrix::zeros(n);
        let mut rng = DeterministicRng::new("posegraph-dense", 5);
        for j in 0..n {
            h.add_diag(j, &(Mat6::identity() * (40.0 + rng.uniform())));
        }
        for j in 0..n {
            for i in j + 1..n {
                h.add_lower(i, j, &(Mat6::identity() * rng.uniform_range(-0.4, 0.4)));
            }
        }
        assert!(
            h.predicted_block_ops() > CHOLESKY_BLOCK_OP_BUDGET * n,
            "expected the dense case to exceed the fill budget"
        );
        let rhs: Vec<Vec6> = (0..n)
            .map(|_| Vec6::from_fn(|_, _| rng.uniform_range(-1.0, 1.0)))
            .collect();
        let damped = h.damped_diagonal(0.0);
        let x = h.solve_pcg(&damped, &rhs).expect("SPD");
        let back = h.multiply(&damped, &x);
        for i in 0..n {
            assert!((back[i] - rhs[i]).amax() < 1e-8, "block {i}");
        }
    }

    // ---- Scale -----------------------------------------------------------

    #[test]
    fn five_hundred_nodes_optimise_quickly() {
        // Five laps of a circle with a small vertical offset per lap, i.e. a
        // trajectory that genuinely revisits. Loop closures land between
        // successive passes over the same place, which is the graph shape
        // place recognition actually produces — not uniformly random long-range
        // edges, which would be an adversarial fill-in case rather than a
        // representative one.
        let per_lap = 100usize;
        let laps = 5usize;
        let n = per_lap * laps;
        let truth: Vec<Se3> = (0..n)
            .map(|k| {
                let lap = (k / per_lap) as f64;
                let a = std::f64::consts::TAU * (k % per_lap) as f64 / per_lap as f64;
                Se3::new(
                    // Heading tangent to the circle, plus a little roll/pitch.
                    So3::exp(&Vec3::new(0.02 * a.sin(), 0.015 * a.cos(), a)),
                    Vec3::new(3.0 * a.cos(), 3.0 * a.sin(), 0.05 * lap),
                )
            })
            .collect();

        let mut rng = DeterministicRng::new("posegraph-scale", 20_260_801);
        let mut g = PoseGraph::new();
        g.add_node(kf(0), truth[0], true);
        let mut guess = truth[0];
        for i in 1..n {
            let step = Vec6::from_fn(|k, _| {
                if k < 3 {
                    rng.normal_with(0.0, 2e-3)
                } else {
                    rng.normal_with(0.0, 1e-3)
                }
            });
            guess = guess
                .compose(&truth[i - 1].inverse().compose(&truth[i]))
                .plus(&step);
            g.add_node(kf(i as u64), guess, false);
        }
        for i in 0..n - 1 {
            g.add_edge(
                kf(i as u64),
                kf(i as u64 + 1),
                truth[i].inverse().compose(&truth[i + 1]),
                info(1.0),
            );
        }
        for lap in 1..laps {
            for slot in (0..per_lap).step_by(20) {
                let a = ((lap - 1) * per_lap + slot) as u64;
                let b = (lap * per_lap + slot) as u64;
                g.add_edge(
                    kf(a),
                    kf(b),
                    truth[a as usize].inverse().compose(&truth[b as usize]),
                    info(1.0),
                );
            }
        }
        let open_loop_error = max_pose_error(&g, &truth);
        assert!(
            open_loop_error > 0.05,
            "drift {open_loop_error} is too small to be a test"
        );

        // `Instant` here is a test-harness stopwatch. Nothing inside the solver
        // reads a clock — spec.md §6 forbids it on any estimation path.
        let start = std::time::Instant::now();
        let report = g.optimize(&SolverConfig::default());
        let elapsed = start.elapsed();

        assert!(
            report.final_cost < report.initial_cost * 1e-10,
            "cost {} -> {}",
            report.initial_cost,
            report.final_cost
        );
        // Measurements are exact, so the global minimum is the ground truth.
        let recovered = max_pose_error(&g, &truth);
        assert!(recovered < 1e-6, "recovered error {recovered}");
        // `cargo test` builds unoptimised and nalgebra pays dearly for that;
        // release is roughly 40x faster, so this is well under a second where
        // it matters.
        assert!(
            elapsed.as_secs_f64() < 5.0,
            "{n} nodes took {elapsed:?} for {} iterations",
            report.iterations
        );
        eprintln!(
            "{n} nodes / {} edges: {elapsed:?}, {} iterations",
            g.edge_count(),
            report.iterations
        );
    }
}
