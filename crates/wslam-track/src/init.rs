//! Two-view bootstrap: the first pose and the first landmarks, up to scale.
//!
//! Monocular initialisation has to survive two scene types that want different
//! models. A planar scene (a table, a wall, a floor — most of the time someone
//! points a phone at something) makes the essential matrix rank deficient: the
//! epipolar constraint has a three-dimensional solution space and the recovered
//! `E` is arbitrary within it. A general scene makes the homography wrong,
//! because no single plane explains the correspondences.
//!
//! So estimate both, score both on the same data, and let the ratio decide —
//! the ORB-SLAM heuristic (Mur-Artal & Tardós, spec.md §5). The scoring is a
//! chi-squared-weighted inlier count, which rewards a model for fitting well as
//! well as for fitting often.
//!
//! **Scale is not observable here and is not claimed.** The returned pose has a
//! unit-norm translation and the landmarks are in units of that baseline
//! (spec.md §4 L3, "Pose up to scale"; §2, "Monocular metric scale is
//! unobservable"). Metric scale arrives later, from L5.
//!
//! All 2D inputs are **undistorted** pixel coordinates; see the module docs of
//! [`crate::motion_ba`].

use wslam_core::{CameraIntrinsics, DeterministicRng, Mat3, Scalar, Se3, So3, Vec2, Vec3};

use crate::motion_ba::pinhole_only;
use crate::triangulate::{triangulate_two_view, TriangulationConfig};

/// Chi-squared 95% quantile with two degrees of freedom — the gate for a
/// point-to-point transfer error, and the constant both models are scored
/// against so the two scores are commensurable.
const CHI2_2DOF_95: Scalar = 5.991;
/// Chi-squared 95% quantile with one degree of freedom — the gate for a
/// point-to-*line* (epipolar) error, which only constrains one direction.
const CHI2_1DOF_95: Scalar = 3.841;

/// Parallax floor applied while *disambiguating* the decomposition, as opposed
/// to while deciding which points become landmarks.
///
/// Cheirality — "is the point in front of both cameras" — is the question that
/// separates the four essential-matrix candidates, and it is meaningful for any
/// point that is not at infinity. This floor exists only to keep the DLT out of
/// its rank-deficient corner; [`InitConfig::min_parallax_rad`], two orders of
/// magnitude larger, is the gate that decides what is worth triangulating.
const CHEIRALITY_PARALLAX_RAD: Scalar = 0.02 * std::f64::consts::PI / 180.0;

/// Which model won the ratio test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitModel {
    /// Planar scene: the homography branch.
    Homography,
    /// General scene: the essential branch.
    Essential,
}

/// Tuning for [`initialize_two_view`].
#[derive(Debug, Clone, Copy)]
pub struct InitConfig {
    /// Assumed per-axis pixel measurement noise. The model gates are chi-squared
    /// quantiles scaled by this, not a raw pixel distance, so that the
    /// homography (two degrees of freedom) and the essential matrix (one) are
    /// held to statistically equivalent standards.
    pub pixel_sigma: Scalar,
    /// RANSAC iteration cap for both models.
    pub ransac_iterations: usize,
    /// Median parallax below which the pair is refused outright.
    pub min_parallax_rad: Scalar,
    /// Minimum inliers the winning model must have.
    pub min_inliers: usize,
    /// Minimum landmarks the winning `(R, t)` must triangulate.
    pub min_triangulated: usize,
    /// Maximum reprojection error for a landmark, in pixels.
    pub max_reprojection_px: Scalar,
    /// `S_H / (S_H + S_E)` above which the homography branch is taken.
    /// 0.45 is the ORB-SLAM value: biased towards the homography, because
    /// mis-modelling a planar scene as general is the more damaging error.
    pub homography_ratio: Scalar,
    /// A second `(R, t)` candidate explaining more than this fraction of the
    /// winner's landmarks means the decomposition is ambiguous, and an
    /// ambiguous bootstrap poisons the whole session.
    pub ambiguity_ratio: Scalar,
    /// Fraction of the winning model's inliers that the chosen `(R, t)` must
    /// place in front of **both** cameras. ORB-SLAM's `nMinGood`; 0.9 is its
    /// value. A correct decomposition explains nearly all of the
    /// correspondences that already agreed with the model, so falling short is
    /// evidence that the configuration was degenerate.
    pub min_cheiral_fraction: Scalar,
}

impl Default for InitConfig {
    fn default() -> Self {
        InitConfig {
            pixel_sigma: 1.0,
            ransac_iterations: 400,
            min_parallax_rad: 1.0_f64.to_radians(),
            min_inliers: 15,
            min_triangulated: 12,
            max_reprojection_px: 4.0,
            homography_ratio: 0.45,
            ambiguity_ratio: 0.75,
            min_cheiral_fraction: 0.9,
        }
    }
}

/// A landmark produced by the bootstrap.
#[derive(Debug, Clone, Copy)]
pub struct InitLandmark {
    /// Index into the input correspondence slice.
    pub match_index: usize,
    /// Position in the first camera's frame, in units of the baseline.
    pub position: Vec3,
    /// 3x3 position covariance in the same units.
    pub covariance: Mat3,
    /// Angle subtended at the landmark by the two viewpoints.
    pub parallax_rad: Scalar,
}

/// Outcome of a successful bootstrap.
#[derive(Debug, Clone)]
pub struct TwoViewInit {
    /// `T_world_camera` of the **second** view, with the first view placed at
    /// the world origin. The translation is unit norm: its *direction* is
    /// observable, its magnitude is not.
    pub pose: Se3,
    /// Which model was used.
    pub model: InitModel,
    /// Accepted landmarks.
    pub landmarks: Vec<InitLandmark>,
    /// Inlier flags of the winning model, in input order.
    pub inliers: Vec<bool>,
    /// Median parallax over every inlier the chosen `(R, t)` places in front of
    /// both cameras — **not** over [`TwoViewInit::landmarks`], which have all
    /// already cleared [`InitConfig::min_parallax_rad`] and whose median is
    /// therefore incapable of reporting a low-parallax pair. This is the
    /// parallax of the view *pair*.
    pub median_parallax_rad: Scalar,
    /// Homography score.
    pub score_homography: Scalar,
    /// Essential score.
    pub score_essential: Scalar,
    /// `S_H / (S_H + S_E)`, the quantity compared against
    /// [`InitConfig::homography_ratio`].
    pub homography_ratio: Scalar,
}

/// A robustly fitted two-view model.
#[derive(Debug, Clone)]
pub struct ModelFit {
    /// The model, expressed in **normalised image coordinates**: a homography
    /// mapping `x1 -> x2`, or an essential matrix satisfying `x2^T E x1 = 0`.
    pub model: Mat3,
    /// Per-correspondence inlier flags.
    pub inliers: Vec<bool>,
    /// ORB-SLAM-style score, summed over both transfer directions.
    pub score: Scalar,
    /// Number of `true` entries in [`ModelFit::inliers`].
    pub inlier_count: usize,
}

/// A rotation, translation direction and plane normal from a homography.
#[derive(Debug, Clone, Copy)]
pub struct HomographyDecomposition {
    /// Rotation of `T_camera2_camera1`.
    pub rotation: So3,
    /// Unit translation of `T_camera2_camera1`; the true magnitude is folded
    /// into the plane distance and is not recoverable from the homography alone.
    pub translation: Vec3,
    /// Plane normal in the first camera's frame.
    pub plane_normal: Vec3,
}

/// Isotropic Hartley normalisation: centroid to the origin, mean distance to
/// `sqrt(2)`.
///
/// Not optional. The DLT design matrix mixes terms of order `x^2` with terms of
/// order 1, and without this the smallest singular vector is dominated by
/// whichever coordinate happens to be largest.
///
/// Returns the transformed points and the `T` with `x' = T x`.
#[must_use]
pub fn hartley_normalize(points: &[Vec2]) -> Option<(Vec<Vec2>, Mat3)> {
    if points.is_empty() {
        return None;
    }
    let n = points.len() as Scalar;
    let centroid = points.iter().sum::<Vec2>() / n;
    let mean_dist = points.iter().map(|p| (p - centroid).norm()).sum::<Scalar>() / n;
    if !(mean_dist.is_finite() && mean_dist > 1e-12) {
        return None; // all points coincident
    }
    let s = (2.0 as Scalar).sqrt() / mean_dist;
    let t = Mat3::new(
        s,
        0.0,
        -s * centroid.x,
        0.0,
        s,
        -s * centroid.y,
        0.0,
        0.0,
        1.0,
    );
    Some((points.iter().map(|p| (p - centroid) * s).collect(), t))
}

/// Four-point DLT homography, `x2 ~ H x1`.
///
/// The input coordinate frame is the caller's business — pass pixels and get a
/// pixel homography, pass normalised image coordinates and get one that
/// decomposes directly into `(R, t, n)`.
#[must_use]
pub fn estimate_homography(correspondences: &[(Vec2, Vec2)]) -> Option<Mat3> {
    if correspondences.len() < 4 {
        return None;
    }
    let src: Vec<Vec2> = correspondences.iter().map(|c| c.0).collect();
    let dst: Vec<Vec2> = correspondences.iter().map(|c| c.1).collect();
    let (a, t1) = hartley_normalize(&src)?;
    let (b, t2) = hartley_normalize(&dst)?;

    let mut rows = Vec::with_capacity(2 * a.len());
    for (p, q) in a.iter().zip(b.iter()) {
        rows.push([-p.x, -p.y, -1.0, 0.0, 0.0, 0.0, q.x * p.x, q.x * p.y, q.x]);
        rows.push([0.0, 0.0, 0.0, -p.x, -p.y, -1.0, q.y * p.x, q.y * p.y, q.y]);
    }
    let h = reshape3(&null_vector_9(&rows)?);
    let denormalized = t2.try_inverse()? * h * t1;
    let scale = denormalized[(2, 2)];
    // Fix the overall sign and scale so downstream comparisons are stable; a
    // homography with h33 == 0 sends the first camera centre to infinity, which
    // no valid two-view geometry does.
    if scale.abs() < 1e-12 {
        return None;
    }
    Some(denormalized / scale)
}

/// Eight-point essential matrix from **normalised image coordinates**.
///
/// Hartley normalisation, the linear null-space solve, then the **singularity
/// constraint** — `E` is forced to rank two in the normalised frame, which is
/// where the least-squares problem was posed, and which is legitimate because
/// `det(T2^T E' T1) = det(T2) det(E') det(T1)`, so rank deficiency is the one
/// property that survives denormalisation intact.
///
/// ## Why the singular values are *not* equalised
///
/// A true essential matrix has two equal non-zero singular values, and the
/// Frobenius-nearest such matrix is `U diag(s, s, 0) V^T` with
/// `s = (σ1 + σ2)/2` (Hartley & Zisserman, Result 9.19). Applying that
/// projection here makes the result measurably *worse*, and the reason is a
/// scale mismatch rather than a bug in the formula:
///
/// On a clean 140-point scene with 0.4 px noise the linear solution already has
/// `σ2/σ1` in `[0.991, 0.999]`. Equalising is therefore a perturbation of a few
/// parts per thousand of `‖E‖` — but the epipolar residuals it is competing
/// with are ~0.5 px out of a focal length of 520, i.e. one part per thousand of
/// the same quantity. Frobenius-nearest is nearest in the wrong metric: it is
/// blind to which directions in `E` the correspondences actually constrain, and
/// it spends the whole noise budget. Measured over six scene draws the
/// projection moves the mean epipolar residual from 0.45 px to between 0.5 and
/// 2.2 px and drops the inlier count at the chi-squared gate from 140/140 to as
/// low as 38/140.
///
/// Nothing downstream wants it either: [`decompose_essential`] reads only `U`
/// and `V` from the SVD, and neither is changed by rescaling singular values.
/// The calibrated constraint belongs in a non-linear refinement that minimises
/// the same reprojection error the score does, not in a Frobenius projection —
/// which is also what ORB-SLAM's `ReconstructF` does (rank two, then straight
/// into the decomposition).
///
/// So: rank two, always; two equal singular values only when the data says so.
#[must_use]
pub fn estimate_essential_eight_point(correspondences: &[(Vec2, Vec2)]) -> Option<Mat3> {
    if correspondences.len() < 8 {
        return None;
    }
    let src: Vec<Vec2> = correspondences.iter().map(|c| c.0).collect();
    let dst: Vec<Vec2> = correspondences.iter().map(|c| c.1).collect();
    let (a, t1) = hartley_normalize(&src)?;
    let (b, t2) = hartley_normalize(&dst)?;

    let mut rows = Vec::with_capacity(a.len());
    for (p, q) in a.iter().zip(b.iter()) {
        rows.push([
            q.x * p.x,
            q.x * p.y,
            q.x,
            q.y * p.x,
            q.y * p.y,
            q.y,
            p.x,
            p.y,
            1.0,
        ]);
    }
    let e_norm = reshape3(&null_vector_9(&rows)?);

    let mut svd = e_norm.svd(true, true);
    svd.sort_by_singular_values();
    let (u, v_t) = (svd.u?, svd.v_t?);
    let (s1, s2) = (svd.singular_values[0], svd.singular_values[1]);
    if !(s1.is_finite() && s1 > 1e-14) {
        return None;
    }
    let singular = u * Mat3::from_diagonal(&Vec3::new(s1, s2, 0.0)) * v_t;
    let e = t2.transpose() * singular * t1;
    e.iter().all(|v| v.is_finite()).then_some(e)
}

/// Robust homography fit over pixel correspondences.
///
/// Returns the homography in **normalised image coordinates**, so it is ready
/// for [`decompose_homography`]. Minimal sets come from `rng` (spec.md §6).
#[must_use]
pub fn estimate_homography_ransac(
    matches: &[(Vec2, Vec2)],
    k: &CameraIntrinsics,
    pixel_sigma: Scalar,
    iterations: usize,
    rng: &mut DeterministicRng,
) -> Option<ModelFit> {
    let normalized = normalize_matches(matches, k);
    ransac(
        &normalized,
        4,
        iterations,
        rng,
        k,
        pixel_sigma,
        estimate_homography,
        score_homography,
    )
}

/// Robust essential-matrix fit over pixel correspondences.
///
/// Minimal set is eight, so the iteration budget matters far more than it does
/// for the homography: at a 60% inlier ratio, 99% confidence needs ~270 samples.
#[must_use]
pub fn estimate_essential_ransac(
    matches: &[(Vec2, Vec2)],
    k: &CameraIntrinsics,
    pixel_sigma: Scalar,
    iterations: usize,
    rng: &mut DeterministicRng,
) -> Option<ModelFit> {
    let normalized = normalize_matches(matches, k);
    ransac(
        &normalized,
        8,
        iterations,
        rng,
        k,
        pixel_sigma,
        estimate_essential_eight_point,
        score_essential,
    )
}

/// The four `(R, t)` candidates consistent with an essential matrix.
///
/// `E = [t]_x R` with `(R, t) = T_camera2_camera1`, so the returned rotations
/// and translations map a point in the first camera's frame into the second's.
/// Two rotations times two translation signs; only cheirality separates them.
#[must_use]
pub fn decompose_essential(e: &Mat3) -> Vec<(So3, Vec3)> {
    let mut svd = e.svd(true, true);
    svd.sort_by_singular_values();
    let (Some(mut u), Some(mut v_t)) = (svd.u, svd.v_t) else {
        return Vec::new();
    };
    // Force both factors into SO(3) so the products below are rotations rather
    // than reflections. Flipping the third column of U only flips the sign of
    // t, which is already enumerated.
    if u.determinant() < 0.0 {
        let mut c = u.column_mut(2);
        c *= -1.0;
    }
    if v_t.determinant() < 0.0 {
        let mut r = v_t.row_mut(2);
        r *= -1.0;
    }
    let w = Mat3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let t = u.column(2).into_owned();
    if t.norm() < 1e-12 {
        return Vec::new();
    }
    let t = t.normalize();
    [
        (u * w * v_t, t),
        (u * w * v_t, -t),
        (u * w.transpose() * v_t, t),
        (u * w.transpose() * v_t, -t),
    ]
    .iter()
    .map(|(r, t)| (rotation_from_orthonormal(r), *t))
    .collect()
}

/// The (up to) eight `(R, t, n)` candidates consistent with a homography.
///
/// Faugeras' SVD decomposition in the form ORB-SLAM uses. Returns empty for the
/// degenerate case in which the homography's singular values are not distinct —
/// most importantly a **pure rotation**, where `H` is a rotation matrix, all
/// three singular values are 1, and there is no translation direction to find.
#[must_use]
pub fn decompose_homography(h: &Mat3) -> Vec<HomographyDecomposition> {
    let mut svd = h.svd(true, true);
    svd.sort_by_singular_values();
    let (Some(u), Some(v_t)) = (svd.u, svd.v_t) else {
        return Vec::new();
    };
    let v = v_t.transpose();
    let sign = u.determinant() * v_t.determinant();
    let (d1, d2, d3) = (
        svd.singular_values[0],
        svd.singular_values[1],
        svd.singular_values[2],
    );
    // Distinct singular values are required: equality means the plane normal is
    // unrecoverable. `1 + 1e-7` separates a genuine plane from a pure rotation
    // (whose ratios sit at 1 to within f64 rounding) without rejecting a small
    // but real baseline.
    if d3 <= 1e-12 || d1 / d2 < 1.0 + 1e-7 || d2 / d3 < 1.0 + 1e-7 {
        return Vec::new();
    }

    let span = d1 * d1 - d3 * d3;
    let aux1 = ((d1 * d1 - d2 * d2) / span).sqrt();
    let aux3 = ((d2 * d2 - d3 * d3) / span).sqrt();
    let x1 = [aux1, aux1, -aux1, -aux1];
    let x3 = [aux3, -aux3, aux3, -aux3];
    let cross = ((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)).sqrt();

    let mut out = Vec::with_capacity(8);

    // d' = +d2: the plane is in front, four sign combinations.
    let s_theta = cross / ((d1 + d3) * d2);
    let c_theta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
    for (i, sign_theta) in [s_theta, -s_theta, -s_theta, s_theta].iter().enumerate() {
        let rp = Mat3::new(
            c_theta,
            0.0,
            -sign_theta,
            0.0,
            1.0,
            0.0,
            *sign_theta,
            0.0,
            c_theta,
        );
        let tp = Vec3::new(x1[i], 0.0, -x3[i]) * (d1 - d3);
        out.push(build_decomposition(
            sign, &u, &rp, &v_t, &v, &tp, x1[i], x3[i],
        ));
    }

    // d' = -d2: the reflected family.
    let s_phi = cross / ((d1 - d3) * d2);
    let c_phi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
    for (i, sign_phi) in [s_phi, -s_phi, -s_phi, s_phi].iter().enumerate() {
        let rp = Mat3::new(
            c_phi, 0.0, *sign_phi, 0.0, -1.0, 0.0, *sign_phi, 0.0, -c_phi,
        );
        let tp = Vec3::new(x1[i], 0.0, x3[i]) * (d1 + d3);
        out.push(build_decomposition(
            sign, &u, &rp, &v_t, &v, &tp, x1[i], x3[i],
        ));
    }
    out.retain(|d| d.translation.iter().all(|c| c.is_finite()));
    out
}

/// Two-view bootstrap.
///
/// Fits both models, picks by the ORB-SLAM ratio, decomposes the winner,
/// disambiguates by cheirality, and triangulates.
///
/// Returns `None` when the pair cannot support an initialisation — too few
/// correspondences, no model with enough inliers, an ambiguous decomposition,
/// or **insufficient parallax**, which is the common case in practice because
/// the user has not started moving yet. Returning a pose here would be worse
/// than returning nothing: every landmark behind it would be garbage and the
/// map would be built on it.
#[must_use]
pub fn initialize_two_view(
    matches: &[(Vec2, Vec2)],
    k: &CameraIntrinsics,
    config: &InitConfig,
    rng: &mut DeterministicRng,
) -> Option<TwoViewInit> {
    if matches.len() < 8 || matches.len() < config.min_inliers {
        return None;
    }
    let kp = pinhole_only(k);
    let iterations = config.ransac_iterations.max(1);

    let homography = estimate_homography_ransac(
        matches,
        &kp,
        config.pixel_sigma,
        iterations,
        &mut rng.fork("init-homography", 0),
    );
    let essential = estimate_essential_ransac(
        matches,
        &kp,
        config.pixel_sigma,
        iterations,
        &mut rng.fork("init-essential", 1),
    );

    let score_h = homography.as_ref().map_or(0.0, |f| f.score);
    let score_e = essential.as_ref().map_or(0.0, |f| f.score);
    let total = score_h + score_e;
    if total <= 0.0 {
        return None;
    }
    let ratio = score_h / total;

    let (model, fit) = if ratio > config.homography_ratio {
        (InitModel::Homography, homography.as_ref()?)
    } else {
        (InitModel::Essential, essential.as_ref()?)
    };
    if fit.inlier_count < config.min_inliers {
        return None;
    }

    // Candidate poses: T_world_camera2 with the first camera at the origin.
    let candidates: Vec<Se3> = match model {
        InitModel::Homography => decompose_homography(&fit.model)
            .into_iter()
            .map(|d| pose_from_relative(&d.rotation, &d.translation))
            .collect(),
        InitModel::Essential => decompose_essential(&fit.model)
            .into_iter()
            .map(|(r, t)| pose_from_relative(&r, &t))
            .collect(),
    };
    if candidates.is_empty() {
        return None;
    }

    let tri_config = TriangulationConfig {
        min_parallax_rad: config.min_parallax_rad,
        max_reprojection_px: config.max_reprojection_px,
        pixel_sigma: config.pixel_sigma,
        // Depths are in units of a unit baseline, so the horizon is a ratio.
        max_depth: 1.0e4,
    };

    // Disambiguation is *cheirality first*. The four (or eight) candidates
    // differ in which of them puts the scene in front of both cameras, and that
    // question has no dependence on how much parallax any individual point
    // subtends. Ranking the candidates by their *parallax-gated* landmark count
    // — as an earlier version of this function did — makes the choice depend on
    // a quantity that is fabricated whenever the configuration is degenerate,
    // and so picks the wrong branch exactly when it matters most.
    //
    // Concretely, on a pure rotation with a hair of detector noise the essential
    // matrix is arbitrary: with `t = 0` the constraint `x2^T [t]_x R x1 = 0`
    // holds for *every* `t`, because `x2` is parallel to `R x1` and
    // `a^T [t]_x a = 0`. The eight-point solution is then unconstrained in the
    // epipole. One spurious candidate concentrated 15 of 70 inliers at short
    // fake depths and won a parallax-gated vote, while the candidate that
    // actually satisfied cheirality for 55 of 70 lost, because all 55 of its
    // points sat below one degree. The bootstrap reported a map made of noise.
    //
    // The parallax count is kept as the *second* key, and it earns its place on
    // a plane. A planar homography admits two physically admissible
    // decompositions (Faugeras & Lustman) and both reproject exactly, so
    // cheirality alone cannot separate them — on the fixture in this module's
    // tests both explain all 140 inliers. What separates them is that the dual
    // solution reconstructs the plane about six times further away in units of
    // the baseline, putting every one of its points below the parallax gate.
    // Preferring the candidate whose depths are actually observable from this
    // baseline is the honest tie-break: the alternative is committing to a
    // reconstruction that, by its own geometry, has no depth information in it.
    let cheirality_config = TriangulationConfig {
        // Only wide enough to keep the DLT out of its rank-deficient corner;
        // `max_depth` does most of the work of excluding points at infinity.
        min_parallax_rad: CHEIRALITY_PARALLAX_RAD.min(config.min_parallax_rad),
        ..tri_config
    };
    // One pass per candidate: the two configs differ only in the parallax
    // floor, so the strict set is the cheiral set filtered on `parallax_rad`.
    let mut ranked: Vec<(Se3, Vec<InitLandmark>)> = candidates
        .into_iter()
        .map(|pose| {
            let mut cheiral =
                triangulate_all(matches, &fit.inliers, &pose, &kp, &cheirality_config);
            cheiral.sort_by(|a, b| a.parallax_rad.total_cmp(&b.parallax_rad));
            (pose, cheiral)
        })
        .collect();
    let strict_count = |c: &[InitLandmark]| {
        c.iter()
            .filter(|l| l.parallax_rad >= config.min_parallax_rad)
            .count()
    };
    ranked.sort_by(|a, b| {
        b.1.len()
            .cmp(&a.1.len())
            .then(strict_count(&b.1).cmp(&strict_count(&a.1)))
    });
    let (pose, cheiral) = ranked.remove(0);
    let best_cheiral = cheiral.len();
    let best_strict = strict_count(&cheiral);

    // An ambiguous decomposition means neither key separated the candidates;
    // committing to either one is then a coin flip.
    if ranked.iter().any(|(_, other)| {
        other.len() as Scalar > config.ambiguity_ratio * best_cheiral as Scalar
            && strict_count(other) as Scalar > config.ambiguity_ratio * best_strict as Scalar
    }) {
        return None;
    }
    // ORB-SLAM's `nMinGood` test: a real two-view geometry explains almost all
    // of its own epipolar inliers, because those correspondences already agree
    // with the model. A decomposition that leaves a tenth of them behind a
    // camera is not the right one, and usually means the configuration was
    // degenerate and the model therefore arbitrary.
    if (best_cheiral as Scalar) < config.min_cheiral_fraction * fit.inlier_count as Scalar {
        return None;
    }
    if best_cheiral == 0 {
        return None;
    }

    // The parallax gate, over the *cheiral* set.
    //
    // This has to be measured before the per-landmark parallax filter, not
    // after. Measuring it after — which is what this function used to do — takes
    // the median of a set from which every point below `min_parallax_rad` has
    // already been removed, so the test `median < min_parallax_rad` is
    // arithmetically incapable of failing. It was dead code, and it was the only
    // thing standing between a pure rotation and a fabricated map: on the
    // rotation-only fixture in `tracker`, 22 of 61 cheiral points cleared one
    // degree (noise, and a handful of drifted tracks), their median was 6.2
    // degrees, and the bootstrap accepted. The median over all 61 is far below
    // the gate, which is the true statement about that view pair.
    //
    // ORB-SLAM measures the same quantity the same way, over the good points
    // rather than over the surviving ones.
    let median_parallax_rad = cheiral[cheiral.len() / 2].parallax_rad;
    if median_parallax_rad < config.min_parallax_rad {
        return None;
    }

    let landmarks: Vec<InitLandmark> = cheiral
        .into_iter()
        .filter(|l| l.parallax_rad >= config.min_parallax_rad)
        .collect();
    if landmarks.len() < config.min_triangulated {
        return None;
    }

    Some(TwoViewInit {
        pose,
        model,
        landmarks,
        inliers: fit.inliers.clone(),
        median_parallax_rad,
        score_homography: score_h,
        score_essential: score_e,
        homography_ratio: ratio,
    })
}

// ---------------------------------------------------------------------------
// internals
// ---------------------------------------------------------------------------

fn normalize_matches(matches: &[(Vec2, Vec2)], k: &CameraIntrinsics) -> Vec<(Vec2, Vec2)> {
    let kp = pinhole_only(k);
    matches
        .iter()
        .map(|(a, b)| (kp.unproject_normalized(*a), kp.unproject_normalized(*b)))
        .collect()
}

/// `T_world_camera2` from a relative `(R, t) = T_camera2_camera1`, with the
/// first camera at the world origin.
fn pose_from_relative(rotation: &So3, translation: &Vec3) -> Se3 {
    Se3::new(*rotation, *translation).inverse()
}

/// Shared RANSAC driver for the two models. The estimator and the scorer are
/// the only things that differ.
#[allow(clippy::too_many_arguments)]
fn ransac(
    normalized: &[(Vec2, Vec2)],
    sample_size: usize,
    iterations: usize,
    rng: &mut DeterministicRng,
    k: &CameraIntrinsics,
    pixel_sigma: Scalar,
    estimate: impl Fn(&[(Vec2, Vec2)]) -> Option<Mat3>,
    score: impl Fn(&Mat3, &[(Vec2, Vec2)], &CameraIntrinsics, Scalar, &mut [bool]) -> Scalar,
) -> Option<ModelFit> {
    let n = normalized.len();
    if n < sample_size {
        return None;
    }
    let mut indices = Vec::with_capacity(sample_size);
    let mut sample = Vec::with_capacity(sample_size);
    let mut scratch = vec![false; n];
    let mut best: Option<ModelFit> = None;
    let mut needed = iterations;

    for iteration in 0..iterations {
        if iteration >= needed {
            break;
        }
        rng.sample_distinct(n, sample_size, &mut indices);
        if indices.len() < sample_size {
            return None;
        }
        sample.clear();
        sample.extend(indices.iter().map(|&i| normalized[i]));
        let Some(model) = estimate(&sample) else {
            continue;
        };
        let s = score(&model, normalized, k, pixel_sigma, &mut scratch);
        if best.as_ref().is_none_or(|b| s > b.score) {
            let inlier_count = scratch.iter().filter(|&&b| b).count();
            best = Some(ModelFit {
                model,
                inliers: scratch.clone(),
                score: s,
                inlier_count,
            });
            // Adaptive cap. This matters far more for the eight-point model than
            // for a three-point PnP: the sample size sits in the exponent, so a
            // clean scene that would otherwise burn the whole budget exits after
            // a handful of samples.
            let w = inlier_count as Scalar / n as Scalar;
            needed = crate::pnp::adaptive_iterations(w, sample_size as u32, 0.99).min(iterations);
        }
    }

    // Local optimisation. The minimal-sample model is only ever a hypothesis,
    // and the eight-point solution in particular is badly conditioned: eight
    // correspondences pin down eight of the nine entries of `E`, but an
    // essential matrix has five degrees of freedom, so projecting the linear
    // answer onto the essential manifold moves it a long way. The consequence
    // is a consensus set that is real but *partial* — the hypothesis explains
    // sixty per cent of a clean scene and RANSAC has no reason to look further.
    //
    // Refitting on the consensus and re-classifying is the LO-RANSAC fix
    // (Chum et al.), and it has to iterate: each refit is estimated from more
    // and better-conditioned data, which recovers more inliers, which improves
    // the next refit. A single round leaves most of that on the table. Same
    // structure as `pnp::solve_pnp_ransac`'s three refinement rounds.
    const LOCAL_OPTIMIZATION_ROUNDS: usize = 8;
    let mut fit = best?;
    for _ in 0..LOCAL_OPTIMIZATION_ROUNDS {
        let consensus: Vec<(Vec2, Vec2)> = normalized
            .iter()
            .zip(&fit.inliers)
            .filter_map(|(m, &ok)| ok.then_some(*m))
            .collect();
        if consensus.len() <= sample_size {
            break;
        }
        let Some(model) = estimate(&consensus) else {
            break;
        };
        let s = score(&model, normalized, k, pixel_sigma, &mut scratch);
        // Strictly better only: `>=` would let the loop oscillate between two
        // equal-scoring models and burn the budget without improving anything.
        if s <= fit.score {
            break;
        }
        fit = ModelFit {
            model,
            inliers: scratch.clone(),
            score: s,
            inlier_count: scratch.iter().filter(|&&b| b).count(),
        };
    }
    Some(fit)
}

/// Symmetric transfer error, scored the ORB-SLAM way: each direction earns
/// `chi2_95 - chi2` when it fits, and a correspondence failing either direction
/// is an outlier.
fn score_homography(
    h: &Mat3,
    normalized: &[(Vec2, Vec2)],
    k: &CameraIntrinsics,
    sigma: Scalar,
    inliers: &mut [bool],
) -> Scalar {
    let Some(h_inv) = h.try_inverse() else {
        inliers.fill(false);
        return 0.0;
    };
    let inv_sigma2 = 1.0 / (sigma * sigma);
    let mut score = 0.0;
    for (i, (x1, x2)) in normalized.iter().enumerate() {
        let (Some(f), Some(b)) = (transfer(h, x1), transfer(&h_inv, x2)) else {
            inliers[i] = false;
            continue;
        };
        // Errors are converted to pixels before gating so the threshold means
        // what it says; for a pinhole this conversion is exact.
        let chi_f = pixel_norm_sq(k, f - x2) * inv_sigma2;
        let chi_b = pixel_norm_sq(k, b - x1) * inv_sigma2;
        if chi_f > CHI2_2DOF_95 || chi_b > CHI2_2DOF_95 {
            inliers[i] = false;
        } else {
            inliers[i] = true;
            score += (CHI2_2DOF_95 - chi_f) + (CHI2_2DOF_95 - chi_b);
        }
    }
    score
}

/// Symmetric epipolar (point-to-line) distance, gated at the one-degree-of-
/// freedom quantile but rewarded on the same two-degree-of-freedom scale as the
/// homography, so `S_H` and `S_E` are directly comparable.
fn score_essential(
    e: &Mat3,
    normalized: &[(Vec2, Vec2)],
    k: &CameraIntrinsics,
    sigma: Scalar,
    inliers: &mut [bool],
) -> Scalar {
    // An epipolar distance in normalised coordinates has no separate x and y
    // component to scale, so it converts to pixels through a single focal
    // length. sqrt(fx*fy) is the right compromise; phone sensors are
    // square-pixel, so fx and fy differ by rounding at most.
    let focal = (k.fx * k.fy).sqrt();
    let inv_sigma2 = 1.0 / (sigma * sigma);
    let mut score = 0.0;
    for (i, (x1, x2)) in normalized.iter().enumerate() {
        let a = Vec3::new(x1.x, x1.y, 1.0);
        let b = Vec3::new(x2.x, x2.y, 1.0);
        let l2 = e * a;
        let l1 = e.transpose() * b;
        let num = b.dot(&l2);
        let den2 = l2.x * l2.x + l2.y * l2.y;
        let den1 = l1.x * l1.x + l1.y * l1.y;
        if den1 <= 1e-24 || den2 <= 1e-24 {
            inliers[i] = false;
            continue;
        }
        let chi_f = num * num / den2 * focal * focal * inv_sigma2;
        let chi_b = num * num / den1 * focal * focal * inv_sigma2;
        if chi_f > CHI2_1DOF_95 || chi_b > CHI2_1DOF_95 {
            inliers[i] = false;
        } else {
            inliers[i] = true;
            score += (CHI2_2DOF_95 - chi_f) + (CHI2_2DOF_95 - chi_b);
        }
    }
    score
}

fn transfer(h: &Mat3, x: &Vec2) -> Option<Vec2> {
    let v = h * Vec3::new(x.x, x.y, 1.0);
    (v.z.abs() > 1e-12).then(|| Vec2::new(v.x / v.z, v.y / v.z))
}

fn pixel_norm_sq(k: &CameraIntrinsics, d: Vec2) -> Scalar {
    let x = d.x * k.fx;
    let y = d.y * k.fy;
    x * x + y * y
}

fn triangulate_all(
    matches: &[(Vec2, Vec2)],
    inliers: &[bool],
    pose2: &Se3,
    k: &CameraIntrinsics,
    config: &TriangulationConfig,
) -> Vec<InitLandmark> {
    let first = Se3::identity();
    let mut out = Vec::new();
    for (i, (a, b)) in matches.iter().enumerate() {
        if !inliers.get(i).copied().unwrap_or(false) {
            continue;
        }
        if let Ok(p) = triangulate_two_view(&first, *a, pose2, *b, k, config) {
            out.push(InitLandmark {
                match_index: i,
                position: p.position,
                covariance: p.covariance,
                parallax_rad: p.parallax_rad,
            });
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn build_decomposition(
    sign: Scalar,
    u: &Mat3,
    rp: &Mat3,
    v_t: &Mat3,
    v: &Mat3,
    tp: &Vec3,
    x1: Scalar,
    x3: Scalar,
) -> HomographyDecomposition {
    let r = (u * rp * v_t) * sign;
    let t = u * tp;
    let mut n = v * Vec3::new(x1, 0.0, x3);
    if n.z < 0.0 {
        n = -n;
    }
    HomographyDecomposition {
        rotation: rotation_from_orthonormal(&r),
        translation: if t.norm() > 1e-12 {
            t.normalize()
        } else {
            Vec3::new(Scalar::NAN, 0.0, 0.0)
        },
        plane_normal: n,
    }
}

/// Wrap a matrix that is orthonormal *by construction* — a product of SVD
/// factors — as an [`So3`], skipping the iterative projection in
/// `So3::from_matrix`, which has nothing to correct here.
fn rotation_from_orthonormal(m: &Mat3) -> So3 {
    So3::from_quaternion(nalgebra::UnitQuaternion::from_rotation_matrix(
        &nalgebra::Rotation3::from_matrix_unchecked(*m),
    ))
}

fn reshape3(v: &[Scalar; 9]) -> Mat3 {
    Mat3::new(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8])
}

/// Null vector of a stack of 9-wide constraint rows.
///
/// Zero-pads to at least nine rows: `nalgebra` returns the thin SVD, so a
/// system with exactly eight constraints would otherwise never expose the
/// singular vector we are after.
fn null_vector_9(rows: &[[Scalar; 9]]) -> Option<[Scalar; 9]> {
    let m = rows.len().max(9);
    let mut a = nalgebra::DMatrix::<Scalar>::zeros(m, 9);
    for (i, row) in rows.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            a[(i, j)] = *v;
        }
    }
    let svd = a.svd(false, true);
    let v_t = svd.v_t?;
    let idx = svd
        .singular_values
        .iter()
        .enumerate()
        .min_by(|x, y| x.1.partial_cmp(y.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)?;
    let mut out = [0.0; 9];
    for (j, o) in out.iter_mut().enumerate() {
        *o = v_t[(idx, j)];
    }
    out.iter().all(|v| v.is_finite()).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use wslam_core::math::hat;

    fn intrinsics() -> CameraIntrinsics {
        CameraIntrinsics::from_focal(520.0, 640, 480)
    }

    /// `T_camera2_camera1` for the synthetic pairs below.
    fn relative() -> (So3, Vec3) {
        (
            So3::exp(&Vec3::new(0.03, -0.09, 0.02)),
            Vec3::new(-0.45, 0.06, 0.08),
        )
    }

    fn project(k: &CameraIntrinsics, p_cam: &Vec3) -> Option<Vec2> {
        crate::motion_ba::project_pinhole(&pinhole_only(k), p_cam)
    }

    /// Correspondences of a general (non-planar) scene, with optional noise.
    fn general_scene(n: usize, sigma: Scalar, seed: u64) -> Vec<(Vec2, Vec2)> {
        let k = intrinsics();
        let (r, t) = relative();
        let mut rng = DeterministicRng::new("scene", seed);
        let mut out = Vec::new();
        while out.len() < n {
            let p = Vec3::new(
                rng.uniform_range(-1.6, 1.6),
                rng.uniform_range(-1.2, 1.2),
                rng.uniform_range(2.5, 8.0),
            );
            let p2 = r.act(&p) + t;
            let (Some(a), Some(b)) = (project(&k, &p), project(&k, &p2)) else {
                continue;
            };
            if !(k.contains(a, 6.0) && k.contains(b, 6.0)) {
                continue;
            }
            out.push((
                a + Vec2::new(rng.normal() * sigma, rng.normal() * sigma),
                b + Vec2::new(rng.normal() * sigma, rng.normal() * sigma),
            ));
        }
        out
    }

    /// Correspondences of a slanted plane.
    fn planar_scene(n: usize, sigma: Scalar, seed: u64) -> Vec<(Vec2, Vec2)> {
        let k = intrinsics();
        let (r, t) = relative();
        let mut rng = DeterministicRng::new("plane", seed);
        let mut out = Vec::new();
        while out.len() < n {
            let (x, y) = (rng.uniform_range(-1.6, 1.6), rng.uniform_range(-1.2, 1.2));
            // z = 5 + 0.4x - 0.25y : a plane tilted away from fronto-parallel so
            // the homography is not a pure scaling.
            let p = Vec3::new(x, y, 5.0 + 0.4 * x - 0.25 * y);
            let p2 = r.act(&p) + t;
            let (Some(a), Some(b)) = (project(&k, &p), project(&k, &p2)) else {
                continue;
            };
            if !(k.contains(a, 6.0) && k.contains(b, 6.0)) {
                continue;
            }
            out.push((
                a + Vec2::new(rng.normal() * sigma, rng.normal() * sigma),
                b + Vec2::new(rng.normal() * sigma, rng.normal() * sigma),
            ));
        }
        out
    }

    #[test]
    fn hartley_normalisation_hits_its_targets() {
        let pts: Vec<Vec2> = (0..30)
            .map(|i| {
                let f = i as Scalar;
                Vec2::new(300.0 + 90.0 * f.sin(), 220.0 + 60.0 * (f * 0.7).cos())
            })
            .collect();
        let (out, t) = hartley_normalize(&pts).unwrap();
        let centroid = out.iter().sum::<Vec2>() / out.len() as Scalar;
        let mean = out.iter().map(|p| p.norm()).sum::<Scalar>() / out.len() as Scalar;
        assert_relative_eq!(centroid.norm(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(mean, (2.0 as Scalar).sqrt(), epsilon = 1e-12);
        // The reported transform really is the one that was applied.
        let v = t * Vec3::new(pts[3].x, pts[3].y, 1.0);
        assert_relative_eq!(Vec2::new(v.x / v.z, v.y / v.z), out[3], epsilon = 1e-12);
    }

    #[test]
    fn hartley_normalisation_rejects_coincident_points() {
        assert!(hartley_normalize(&[]).is_none());
        assert!(hartley_normalize(&[Vec2::new(1.0, 2.0); 5]).is_none());
    }

    #[test]
    fn homography_dlt_recovers_a_planted_transform() {
        let truth = Mat3::new(1.2, 0.15, 30.0, -0.1, 0.95, -12.0, 2e-4, -1e-4, 1.0);
        let mut rng = DeterministicRng::new("h", 1);
        let pts: Vec<(Vec2, Vec2)> = (0..25)
            .map(|_| {
                let p = Vec2::new(
                    rng.uniform_range(-200.0, 200.0),
                    rng.uniform_range(-150.0, 150.0),
                );
                let v = truth * Vec3::new(p.x, p.y, 1.0);
                (p, Vec2::new(v.x / v.z, v.y / v.z))
            })
            .collect();
        let h = estimate_homography(&pts).unwrap();
        assert_relative_eq!(h, truth, epsilon = 1e-9);
    }

    #[test]
    fn homography_needs_four_points() {
        let pts = general_scene(3, 0.0, 2);
        assert!(estimate_homography(&pts).is_none());
    }

    #[test]
    fn essential_matrix_satisfies_the_epipolar_constraint_it_was_fitted_to() {
        let k = intrinsics();
        let matches = general_scene(40, 0.0, 3);
        let normalized = normalize_matches(&matches, &k);
        let e = estimate_essential_eight_point(&normalized).unwrap();
        for (a, b) in &normalized {
            let residual = Vec3::new(b.x, b.y, 1.0).dot(&(e * Vec3::new(a.x, a.y, 1.0)));
            assert!(residual.abs() < 1e-10, "epipolar residual {residual}");
        }
        // Two equal singular values and one zero: that is what makes it
        // essential rather than merely fundamental.
        let mut svd = e.svd(false, false);
        svd.sort_by_singular_values();
        let s = svd.singular_values;
        assert_relative_eq!(s[0], s[1], epsilon = 1e-9);
        assert!(s[2] < 1e-9, "third singular value {}", s[2]);
    }

    #[test]
    fn essential_matrix_matches_the_planted_geometry() {
        let k = intrinsics();
        let (r, t) = relative();
        let truth = hat(&t) * r.matrix();
        let matches = general_scene(50, 0.0, 4);
        let e = estimate_essential_eight_point(&normalize_matches(&matches, &k)).unwrap();
        // Up to sign and scale.
        let a = e / e.norm();
        let b = truth / truth.norm();
        let diff = (a - b).norm().min((a + b).norm());
        assert!(diff < 1e-9, "essential differs by {diff}");
    }

    #[test]
    fn essential_needs_eight_points() {
        let k = intrinsics();
        let matches = general_scene(7, 0.0, 5);
        assert!(estimate_essential_eight_point(&normalize_matches(&matches, &k)).is_none());
    }

    #[test]
    fn essential_decomposition_contains_the_truth() {
        let (r, t) = relative();
        let e = hat(&t.normalize()) * r.matrix();
        let solutions = decompose_essential(&e);
        assert_eq!(solutions.len(), 4);
        let best = solutions
            .iter()
            .map(|(rr, tt)| rr.minus(&r).norm() + (tt - t.normalize()).norm())
            .fold(Scalar::INFINITY, f64::min);
        assert!(best < 1e-9, "closest decomposition is {best} away");
    }

    #[test]
    fn homography_decomposition_contains_the_truth() {
        // H = R - t n^T / d for a plane with normal n at distance d.
        let (r, t) = relative();
        let n = Vec3::new(0.15, -0.1, -1.0).normalize();
        let d = 5.0;
        let h = r.matrix() - (t * n.transpose()) / d;
        let solutions = decompose_homography(&h);
        assert_eq!(solutions.len(), 8);
        let best = solutions
            .iter()
            .map(|s| s.rotation.minus(&r).norm() + (s.translation - t.normalize()).norm())
            .fold(Scalar::INFINITY, f64::min);
        assert!(best < 1e-8, "closest decomposition is {best} away");
    }

    #[test]
    fn a_pure_rotation_homography_has_no_translation_to_decompose() {
        let h = So3::exp(&Vec3::new(0.1, -0.2, 0.05)).matrix();
        assert!(
            decompose_homography(&h).is_empty(),
            "a rotation homography must not yield a baseline"
        );
    }

    /// Run the bootstrap over several scene draws and return, per draw, the
    /// chosen model, the rotation error and the cosine between the estimated
    /// and true translation *directions*.
    ///
    /// Sweeping rather than trusting one draw: a single seed that happens to
    /// pass tells you nothing about whether the next one will, and a threshold
    /// tuned to one draw is a threshold tuned to noise.
    fn sweep(planar: bool, sigma: Scalar, seeds: u64) -> Vec<(InitModel, Scalar, Scalar, usize)> {
        let k = intrinsics();
        let (r, t) = relative();
        // T_world_camera2 is the inverse of the relative T_camera2_camera1.
        let truth = Se3::new(r, t).inverse();
        (0..seeds)
            .map(|s| {
                let matches = if planar {
                    planar_scene(140, sigma, 3000 + s)
                } else {
                    general_scene(140, sigma, 5000 + s)
                };
                let mut rng = DeterministicRng::new("init", 7000 + s);
                let init = initialize_two_view(&matches, &k, &InitConfig::default(), &mut rng)
                    .unwrap_or_else(|| panic!("planar={planar} seed={s}: refused to initialise"));
                let rot = init.pose.rotation().minus(&truth.rotation()).norm();
                let cos = init
                    .pose
                    .translation()
                    .normalize()
                    .dot(&truth.translation().normalize());
                assert_relative_eq!(init.pose.translation().norm(), 1.0, epsilon = 1e-9);
                (init.model, rot, cos, init.landmarks.len())
            })
            .collect()
    }

    fn median_of(mut v: Vec<Scalar>) -> Scalar {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    #[test]
    fn general_scene_picks_the_essential_branch() {
        let results = sweep(false, 0.4, 6);
        for (model, rot, cos, landmarks) in &results {
            assert_eq!(*model, InitModel::Essential);
            // Loose per-draw bounds; the median below is the accuracy claim.
            // The epipole is the weakest quantity in a two-view bootstrap with a
            // 0.45-unit baseline against a scene 2.5-8 units away, and the
            // eight-point solution is linear, not maximum-likelihood.
            assert!(*rot < 0.03, "rotation error {rot} rad");
            assert!(*cos > 0.99, "translation direction cos {cos}");
            assert!(*landmarks > 100, "{landmarks} landmarks");
        }
        let rot = median_of(results.iter().map(|r| r.1).collect());
        let cos = median_of(results.iter().map(|r| r.2).collect());
        assert!(rot < 0.012, "median rotation error {rot} rad");
        assert!(cos > 0.997, "median translation direction cos {cos}");
    }

    #[test]
    fn planar_scene_picks_the_homography_branch() {
        // The margin here is structurally narrow and that is not a bug: on a
        // plane the epipolar constraint has a three-dimensional solution space,
        // so the essential matrix fits the data about as well as the homography
        // does and S_H/(S_H+S_E) sits near 0.5. ORB-SLAM's 0.45 threshold is
        // chosen for exactly that, and it is why the test asserts the branch
        // over several draws rather than once.
        let results = sweep(true, 0.4, 6);
        for (model, rot, cos, landmarks) in &results {
            assert_eq!(*model, InitModel::Homography);
            assert!(*rot < 0.02, "rotation error {rot} rad");
            assert!(*cos > 0.999, "translation direction cos {cos}");
            assert!(*landmarks > 100, "{landmarks} landmarks");
        }
        let rot = median_of(results.iter().map(|r| r.1).collect());
        assert!(rot < 0.006, "median rotation error {rot} rad");
    }

    #[test]
    fn the_returned_translation_is_unit_norm() {
        let k = intrinsics();
        let matches = general_scene(120, 0.3, 12);
        let mut rng = DeterministicRng::new("init", 5);
        let init = initialize_two_view(&matches, &k, &InitConfig::default(), &mut rng).unwrap();
        assert_relative_eq!(init.pose.translation().norm(), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn pure_rotation_refuses_to_initialise() {
        // spec.md §6 L3 names this as the tier-2 survival case. There is no
        // baseline, so there is no pose and no map to build.
        let k = intrinsics();
        let r = So3::exp(&Vec3::new(0.0, 0.09, 0.0));
        let mut rng = DeterministicRng::new("rot", 6);
        let mut matches = Vec::new();
        while matches.len() < 120 {
            let p = Vec3::new(
                rng.uniform_range(-1.6, 1.6),
                rng.uniform_range(-1.2, 1.2),
                rng.uniform_range(2.5, 8.0),
            );
            let (Some(a), Some(b)) = (project(&k, &p), project(&k, &r.act(&p))) else {
                continue;
            };
            if !(k.contains(a, 6.0) && k.contains(b, 6.0)) {
                continue;
            }
            matches.push((a, b));
        }
        let mut rng = DeterministicRng::new("init", 7);
        assert!(initialize_two_view(&matches, &k, &InitConfig::default(), &mut rng).is_none());
    }

    #[test]
    fn a_baseline_too_short_for_parallax_refuses_to_initialise() {
        // Real translation, but 2 mm of it against a scene 5 m away.
        let k = intrinsics();
        let t = Vec3::new(-0.002, 0.0, 0.0);
        let mut rng = DeterministicRng::new("tiny", 8);
        let mut matches = Vec::new();
        while matches.len() < 120 {
            let p = Vec3::new(
                rng.uniform_range(-1.6, 1.6),
                rng.uniform_range(-1.2, 1.2),
                rng.uniform_range(4.0, 8.0),
            );
            let (Some(a), Some(b)) = (project(&k, &p), project(&k, &(p + t))) else {
                continue;
            };
            if !(k.contains(a, 6.0) && k.contains(b, 6.0)) {
                continue;
            }
            matches.push((a, b));
        }
        let mut rng = DeterministicRng::new("init", 9);
        assert!(initialize_two_view(&matches, &k, &InitConfig::default(), &mut rng).is_none());
    }

    #[test]
    fn initialisation_is_reproducible_for_a_given_seed() {
        let k = intrinsics();
        let matches = general_scene(120, 0.5, 13);
        let run = |seed: u64| {
            let mut rng = DeterministicRng::new("init", seed);
            initialize_two_view(&matches, &k, &InitConfig::default(), &mut rng).unwrap()
        };
        let a = run(31337);
        let b = run(31337);
        assert_eq!(a.pose.log(), b.pose.log());
        assert_eq!(a.inliers, b.inliers);
        assert_eq!(a.landmarks.len(), b.landmarks.len());
        assert_eq!(a.score_essential.to_bits(), b.score_essential.to_bits());
    }

    #[test]
    fn survives_a_third_of_the_matches_being_wrong() {
        let k = intrinsics();
        let (r, t) = relative();
        let mut matches = general_scene(150, 0.3, 14);
        let mut rng = DeterministicRng::new("corrupt", 15);
        let mut corrupted = 0;
        for (i, m) in matches.iter_mut().enumerate() {
            if i % 3 == 0 {
                m.1 = Vec2::new(rng.uniform_range(0.0, 640.0), rng.uniform_range(0.0, 480.0));
                corrupted += 1;
            }
        }
        assert_eq!(corrupted, 50);
        let cfg = InitConfig {
            ransac_iterations: 3000,
            ..InitConfig::default()
        };
        let mut rng = DeterministicRng::new("init", 16);
        let init = initialize_two_view(&matches, &k, &cfg, &mut rng).unwrap();
        let truth = Se3::new(r, t).inverse();
        assert!(init.pose.rotation().minus(&truth.rotation()).norm() < 0.02);
        let cos = init
            .pose
            .translation()
            .normalize()
            .dot(&truth.translation().normalize());
        assert!(cos > 0.998, "translation direction cos {cos}");
        // The wrong matches must not have been accepted as landmarks.
        assert!(init.landmarks.iter().all(|l| l.match_index % 3 != 0));
    }

    #[test]
    fn too_few_correspondences_is_refused() {
        let k = intrinsics();
        let mut rng = DeterministicRng::new("init", 17);
        let matches = general_scene(7, 0.0, 18);
        assert!(initialize_two_view(&matches, &k, &InitConfig::default(), &mut rng).is_none());
    }

    #[test]
    fn landmarks_reproject_onto_both_views() {
        let k = intrinsics();
        let matches = general_scene(120, 0.0, 19);
        let mut rng = DeterministicRng::new("init", 20);
        let init = initialize_two_view(&matches, &k, &InitConfig::default(), &mut rng).unwrap();
        let kp = pinhole_only(&k);
        for lm in &init.landmarks {
            let (a, b) = matches[lm.match_index];
            let pa = project(&kp, &lm.position).unwrap();
            let pb = project(&kp, &init.pose.inverse().act(&lm.position)).unwrap();
            assert!((pa - a).norm() < 1.0, "view 1 error {}", (pa - a).norm());
            assert!((pb - b).norm() < 1.0, "view 2 error {}", (pb - b).norm());
            assert!(lm.position.z > 0.0, "landmark behind the first camera");
            assert!(lm.covariance[(2, 2)] > 0.0);
        }
    }
}
