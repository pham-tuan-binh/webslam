//! Pyramidal Lucas-Kanade, inverse compositional.
//!
//! Inverse compositional (Baker & Matthews 2004) rather than the forwards
//! additive formulation: the Hessian is built from the *template* gradients, so
//! it is computed once per feature instead of once per iteration. For a
//! translation-only warp on a 9x9 patch that is 81 gradient evaluations and one
//! 2x2 inverse per feature per level, and the per-iteration cost collapses to a
//! single pass of bilinear samples. On a phone that difference is the frame
//! budget.
//!
//! The warp is pure translation. An affine warp would track through scale and
//! shear, but it needs a larger window to stay conditioned, and at 30-60 Hz the
//! inter-frame affine deformation of a small patch is below the noise floor.
//! spec.md §4 L3 buys robustness to large motion with the *pyramid*, not with
//! warp parameters.
//!
//! Every accepted track is checked forward-backward. A track that does not
//! return to where it started is not a track, and letting one through poisons
//! the PnP inlier set with a correspondence that is geometrically consistent
//! with nothing.

use crate::pyramid::{from_level, to_level, Pyramid};
use wslam_core::{GrayImage, Scalar, Vec2};

/// Why a track ended where it did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KltStatus {
    /// Converged inside the image with an acceptable residual.
    Converged,
    /// Hit the iteration cap while still moving.
    NotConverged,
    /// Walked outside the usable image area.
    OutOfBounds,
    /// The template patch has no corner: the structure tensor is rank
    /// deficient, so the displacement is unobservable along one direction (the
    /// aperture problem).
    IllConditioned,
    /// Converged, but onto something that does not look like the template.
    HighError,
    /// Converged, but tracking back from the result does not return to the
    /// start.
    InconsistentBackward,
}

/// Result of tracking one point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KltTrack {
    /// Position in the next frame, full-resolution pixels. Meaningful only when
    /// [`KltTrack::ok`].
    pub px: Vec2,
    /// Outcome.
    pub status: KltStatus,
    /// Mean absolute photometric residual over the finest patch solved, in
    /// intensity units.
    pub error: Scalar,
    /// Forward-backward round-trip distance in pixels. `INFINITY` when the
    /// backward pass itself failed; `0` when the check was disabled.
    pub fb_error: Scalar,
}

impl KltTrack {
    /// Whether the track should be used.
    #[inline]
    #[must_use]
    pub fn ok(&self) -> bool {
        matches!(self.status, KltStatus::Converged)
    }

    fn failed(status: KltStatus, px: Vec2, error: Scalar) -> Self {
        KltTrack {
            px,
            status,
            error,
            fb_error: Scalar::INFINITY,
        }
    }
}

/// Flow tuning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KltConfig {
    /// Half-width of the tracking patch; the patch is `(2r+1) x (2r+1)`.
    pub window_radius: u32,
    /// Iteration cap per pyramid level.
    pub max_iterations: u32,
    /// Convergence threshold on the update, in level pixels.
    pub epsilon: Scalar,
    /// Reject when the mean absolute residual exceeds this (0-255 units).
    pub max_error: Scalar,
    /// Reject when the smaller eigenvalue of the per-pixel-averaged structure
    /// tensor falls below this.
    pub min_eigenvalue: Scalar,
    /// Forward-backward tolerance in full-resolution pixels. `None` disables
    /// the check, which is only ever right in a test.
    pub forward_backward: Option<Scalar>,
    /// Extra margin, in level pixels, kept between the patch and the image
    /// edge.
    pub border: Scalar,
    /// Coarsest level to use. Clamped to the shallower of the two pyramids.
    pub max_level: u32,
}

impl Default for KltConfig {
    fn default() -> Self {
        KltConfig {
            window_radius: 5,
            max_iterations: 24,
            epsilon: 0.01,
            max_error: 28.0,
            min_eigenvalue: 1e-3,
            forward_backward: Some(1.0),
            border: 1.0,
            max_level: u32::MAX,
        }
    }
}

/// Track `points` from `prev` into `next`.
///
/// `guesses`, when supplied, are full-resolution predictions of where each
/// point will land — the tracker feeds it the L1 rotation prior's reprojection.
/// A good guess mainly saves coarse-level iterations; it cannot rescue a
/// displacement the pyramid does not span.
#[must_use]
pub fn track(
    prev: &Pyramid,
    next: &Pyramid,
    points: &[Vec2],
    guesses: Option<&[Vec2]>,
    config: &KltConfig,
) -> Vec<KltTrack> {
    points
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let guess = guesses.and_then(|g| g.get(i).copied()).unwrap_or(p);
            track_one(prev, next, p, guess, config)
        })
        .collect()
}

/// Track a single point, including the forward-backward check.
#[must_use]
pub fn track_one(
    prev: &Pyramid,
    next: &Pyramid,
    point: Vec2,
    guess: Vec2,
    config: &KltConfig,
) -> KltTrack {
    let mut fwd = track_directed(prev, next, point, guess, config);
    let Some(tolerance) = config.forward_backward else {
        fwd.fb_error = 0.0;
        return fwd;
    };
    if !fwd.ok() {
        return fwd;
    }
    // Seed the backward pass with the *prediction*, mirrored — not with the
    // forward result's displacement, and not with `point`.
    //
    // Seeding at `fwd.px` (zero displacement) would force the backward pass to
    // span the whole motion unaided, so a prior that lets the forward pass
    // handle a jump wider than the pyramid would be thrown away by a backward
    // pass that cannot. Mirroring keeps the two directions symmetric.
    //
    // This is not a tautology even when the prediction is exact: the backward
    // pass starting at `point` still has to *stay* there, which it only does if
    // the two patches genuinely correspond. A mis-track lands on content that
    // does not match, and the first inverse-compositional step walks away from
    // it.
    let back_seed = fwd.px - (guess - point);
    let back = track_directed(next, prev, fwd.px, back_seed, config);
    let fb = if back.ok() {
        (back.px - point).norm()
    } else {
        Scalar::INFINITY
    };
    fwd.fb_error = fb;
    if fb > tolerance {
        fwd.status = KltStatus::InconsistentBackward;
    }
    fwd
}

/// One direction of the flow, coarse to fine, with no consistency check.
///
/// Public because the forward-backward check is a policy the tracker may want
/// to apply itself (it already knows which features it trusts), and because a
/// test needs to observe the unchecked result to show the check earns its keep.
#[must_use]
pub fn track_directed(
    prev: &Pyramid,
    next: &Pyramid,
    point: Vec2,
    guess: Vec2,
    config: &KltConfig,
) -> KltTrack {
    let top = prev.top_level().min(next.top_level()).min(config.max_level);

    let mut current = guess;
    let mut error = Scalar::INFINITY;
    let mut solved_any = false;

    for level in (0..=top).rev() {
        let (Some(t_level), Some(n_level)) = (prev.level(level), next.level(level)) else {
            continue;
        };
        let t_px = to_level(point, level);
        let g_px = to_level(current, level);

        // A template patch that does not fit contributes clamped-border pixels
        // in place of image content; skipping the level is strictly better than
        // solving against fabricated data.
        if !patch_fits(&t_level.image, t_px, config) {
            continue;
        }
        match solve_level(&t_level.image, &n_level.image, t_px, g_px, config) {
            Ok(outcome) => {
                current = from_level(outcome.px, level);
                error = outcome.error;
                solved_any = true;
                if outcome.out_of_bounds {
                    return KltTrack::failed(KltStatus::OutOfBounds, current, error);
                }
                if !outcome.converged && level == 0 {
                    return KltTrack::failed(KltStatus::NotConverged, current, error);
                }
            }
            Err(status) => {
                // Ill-conditioning at a coarse level is common and harmless —
                // the level simply has no structure. At level 0 it is fatal.
                if level == 0 {
                    return KltTrack::failed(status, current, error);
                }
            }
        }
    }

    if !solved_any {
        return KltTrack::failed(KltStatus::IllConditioned, current, error);
    }
    if error > config.max_error {
        return KltTrack::failed(KltStatus::HighError, current, error);
    }
    KltTrack {
        px: current,
        status: KltStatus::Converged,
        error,
        fb_error: 0.0,
    }
}

struct LevelOutcome {
    px: Vec2,
    error: Scalar,
    converged: bool,
    out_of_bounds: bool,
}

fn patch_fits(image: &GrayImage, px: Vec2, config: &KltConfig) -> bool {
    let m = config.window_radius as Scalar + config.border + 1.0;
    px.x >= m
        && px.y >= m
        && px.x <= image.width() as Scalar - 1.0 - m
        && px.y <= image.height() as Scalar - 1.0 - m
}

/// Inverse-compositional solve at one level, translation warp.
fn solve_level(
    template: &GrayImage,
    target: &GrayImage,
    t_px: Vec2,
    mut g_px: Vec2,
    config: &KltConfig,
) -> Result<LevelOutcome, KltStatus> {
    let r = config.window_radius.max(1) as i32;
    let side = (2 * r + 1) as usize;
    let n = side * side;

    let mut patch = Vec::with_capacity(n);
    let mut grad = Vec::with_capacity(n);
    let mut h = nalgebra::Matrix2::<Scalar>::zeros();
    for dy in -r..=r {
        for dx in -r..=r {
            let (px, py) = (t_px.x + dx as Scalar, t_px.y + dy as Scalar);
            let g = template.gradient_bilinear(px, py);
            patch.push(template.sample_bilinear(px, py));
            h += g * g.transpose();
            grad.push(g);
        }
    }

    // Averaged so the threshold means the same thing at any window size, and
    // matches the units of `corners::shi_tomasi_response_map`.
    let inv_n = 1.0 / n as Scalar;
    let (a, b, c) = (h[(0, 0)] * inv_n, h[(0, 1)] * inv_n, h[(1, 1)] * inv_n);
    let lambda_min = 0.5 * ((a + c) - ((a - c) * (a - c) + 4.0 * b * b).max(0.0).sqrt());
    if lambda_min < config.min_eigenvalue {
        return Err(KltStatus::IllConditioned);
    }
    let Some(h_inv) = h.try_inverse() else {
        return Err(KltStatus::IllConditioned);
    };

    let mut converged = false;
    for _ in 0..config.max_iterations {
        if !patch_fits(target, g_px, config) {
            return Ok(LevelOutcome {
                px: g_px,
                error: Scalar::INFINITY,
                converged: false,
                out_of_bounds: true,
            });
        }
        let mut rhs = Vec2::zeros();
        let mut i = 0usize;
        for dy in -r..=r {
            for dx in -r..=r {
                let v = target.sample_bilinear(g_px.x + dx as Scalar, g_px.y + dy as Scalar);
                rhs += grad[i] * (v - patch[i]);
                i += 1;
            }
        }
        // Delta = H^-1 sum grad_T (I(W) - T); the inverse-compositional update
        // composes the *inverse* warp, hence the subtraction.
        let delta = h_inv * rhs;
        g_px -= delta;
        if delta.norm() < config.epsilon {
            converged = true;
            break;
        }
    }

    if !patch_fits(target, g_px, config) {
        return Ok(LevelOutcome {
            px: g_px,
            error: Scalar::INFINITY,
            converged: false,
            out_of_bounds: true,
        });
    }

    let mut abs = 0.0;
    let mut i = 0usize;
    for dy in -r..=r {
        for dx in -r..=r {
            let v = target.sample_bilinear(g_px.x + dx as Scalar, g_px.y + dy as Scalar);
            abs += (v - patch[i]).abs();
            i += 1;
        }
    }

    Ok(LevelOutcome {
        px: g_px,
        error: abs * inv_n,
        converged,
        out_of_bounds: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyramid::{PyramidConfig, PyramidFilter};
    use wslam_core::CameraIntrinsics;

    /// Band-limited, aperiodic, and evaluated in continuous coordinates so a
    /// translated copy carries the *exact* displacement rather than a resampled
    /// approximation of one. Testing sub-pixel accuracy against a bilinearly
    /// resampled reference measures the resampler, not the tracker.
    fn texture(x: Scalar, y: Scalar) -> Scalar {
        128.0
            + 52.0 * (0.029 * x + 0.017 * y).sin()
            + 34.0 * (0.101 * x - 0.063 * y).cos()
            + 20.0 * (0.181 * x).sin() * (0.149 * y).cos()
            + 12.0 * (0.331 * x + 0.257 * y).sin()
    }

    fn render(w: u32, h: u32, f: impl Fn(Scalar, Scalar) -> Scalar) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        let data = img.data_mut();
        for y in 0..h {
            for x in 0..w {
                data[(y * w + x) as usize] =
                    f(x as Scalar, y as Scalar).round().clamp(0.0, 255.0) as u8;
            }
        }
        img
    }

    fn pyr(img: &GrayImage, levels: u32) -> Pyramid {
        let k = CameraIntrinsics::from_focal(img.width() as Scalar, img.width(), img.height());
        Pyramid::build(
            img,
            &k,
            &PyramidConfig {
                levels,
                filter: PyramidFilter::Binomial,
            },
        )
    }

    /// `next` is `prev` shifted by `(dx, dy)`: a point at `p` in `prev` appears
    /// at `p + (dx, dy)` in `next`.
    fn shifted_pair(w: u32, h: u32, dx: Scalar, dy: Scalar, levels: u32) -> (Pyramid, Pyramid) {
        let a = render(w, h, texture);
        let b = render(w, h, |x, y| texture(x - dx, y - dy));
        (pyr(&a, levels), pyr(&b, levels))
    }

    #[test]
    fn recovers_an_integer_translation() {
        let (a, b) = shifted_pair(256, 256, 3.0, -2.0, 3);
        let p = Vec2::new(120.0, 130.0);
        let t = track_one(&a, &b, p, p, &KltConfig::default());
        assert!(t.ok(), "{t:?}");
        assert!((t.px - Vec2::new(123.0, 128.0)).norm() < 0.05, "{:?}", t.px);
    }

    #[test]
    fn recovers_a_subpixel_translation_to_a_tenth_of_a_pixel() {
        let (a, b) = shifted_pair(256, 256, 3.4, 1.6, 3);
        let cfg = KltConfig::default();
        let mut worst: Scalar = 0.0;
        for &(x, y) in &[
            (80.0, 90.0),
            (128.0, 128.0),
            (170.0, 70.0),
            (100.0, 180.0),
            (60.0, 160.0),
        ] {
            let p = Vec2::new(x, y);
            let t = track_one(&a, &b, p, p, &cfg);
            assert!(t.ok(), "at {p:?}: {t:?}");
            let err = (t.px - (p + Vec2::new(3.4, 1.6))).norm();
            worst = worst.max(err);
        }
        assert!(worst < 0.1, "worst sub-pixel error {worst} px");
    }

    #[test]
    fn tracks_in_the_reported_direction() {
        // A sign flip in the inverse-compositional update produces a tracker
        // that converges beautifully onto exactly the wrong displacement.
        let (a, b) = shifted_pair(256, 256, 5.0, 0.0, 3);
        let p = Vec2::new(128.0, 128.0);
        let t = track_one(&a, &b, p, p, &KltConfig::default());
        assert!(t.ok());
        assert!(t.px.x > p.x, "expected +x motion, got {:?}", t.px);
    }

    #[test]
    fn pyramid_earns_its_place_on_a_large_displacement() {
        // 17 px is far outside a 5-px window's basin of attraction at full
        // resolution, and well inside it at level 3 (17 / 8 ~ 2.1 px).
        let (a4, b4) = shifted_pair(320, 320, 17.0, -11.0, 4);
        let p = Vec2::new(160.0, 170.0);
        let truth = p + Vec2::new(17.0, -11.0);

        let single = KltConfig {
            max_level: 0,
            ..KltConfig::default()
        };
        let one = track_directed(&a4, &b4, p, p, &single);
        let single_err = (one.px - truth).norm();
        assert!(
            !one.ok() || single_err > 2.0,
            "single level should not solve a 17 px jump, got {one:?} (err {single_err})"
        );

        let many = track_one(&a4, &b4, p, p, &KltConfig::default());
        assert!(many.ok(), "{many:?}");
        assert!(
            (many.px - truth).norm() < 0.25,
            "multi-level error {} px",
            (many.px - truth).norm()
        );
    }

    #[test]
    fn a_prediction_rescues_the_single_level_case() {
        // The same 17 px jump, solved at one level given the L1-style prior.
        // Confirms the failure above is basin size, not a broken solver.
        let (a, b) = shifted_pair(320, 320, 17.0, -11.0, 1);
        let p = Vec2::new(160.0, 170.0);
        let truth = p + Vec2::new(17.0, -11.0);
        let guess = truth + Vec2::new(0.7, -0.6);
        let t = track_one(&a, &b, p, guess, &KltConfig::default());
        assert!(t.ok(), "{t:?}");
        assert!((t.px - truth).norm() < 0.15, "{:?}", t.px);
        // The backward pass had to re-solve a 0.9 px offset from the mirrored
        // prediction, so the round trip is a real measurement here.
        assert!(t.fb_error < 0.15, "fb {}", t.fb_error);
    }

    #[test]
    fn forward_backward_passes_a_true_translation() {
        let (a, b) = shifted_pair(256, 256, 2.5, -1.25, 3);
        let p = Vec2::new(140.0, 110.0);
        let t = track_one(&a, &b, p, p, &KltConfig::default());
        assert!(t.ok());
        assert!(t.fb_error < 0.05, "fb {} px", t.fb_error);
    }

    #[test]
    fn forward_backward_rejects_a_track_into_a_blank_region() {
        // Occlusion, which is what forward-backward exists for: half the frame
        // goes blank (a hand, a wall, a pocket). A feature whose destination is
        // inside the blank half slides to the occlusion boundary and *converges
        // there* with a plausible update — the forward pass has no way to know
        // it is looking at an edge instead of its own patch. Only the round trip
        // does.
        let a = render(256, 256, texture);
        let occluded = render(
            256,
            256,
            |x, y| if x > 110.0 { 128.0 } else { texture(x, y) },
        );
        let (pa, pb) = (pyr(&a, 3), pyr(&occluded, 3));

        // Disable the photometric gate so forward-backward is the only thing
        // left that can reject the track.
        let unchecked = KltConfig {
            max_error: Scalar::INFINITY,
            forward_backward: None,
            ..KltConfig::default()
        };
        let checked = KltConfig {
            forward_backward: Some(1.0),
            ..unchecked
        };

        let inside_blank = Vec2::new(128.0, 128.0);
        let without = track_one(&pa, &pb, inside_blank, inside_blank, &unchecked);
        assert!(
            without.ok(),
            "fixture is wrong: the forward pass must look healthy, got {without:?}"
        );
        let with = track_one(&pa, &pb, inside_blank, inside_blank, &checked);
        assert_eq!(with.status, KltStatus::InconsistentBackward, "{with:?}");
        assert!(!with.ok());

        // Control, same frame pair: a feature that stays in the visible half is
        // still accepted, so the check is discriminating and not just strict.
        let visible = Vec2::new(100.0, 90.0);
        let good = track_one(&pa, &pb, visible, visible, &checked);
        assert!(good.ok(), "{good:?}");
        assert!((good.px - visible).norm() < 0.01);
    }

    #[test]
    fn a_flat_template_is_rejected_as_ill_conditioned() {
        let flat = render(128, 128, |_, _| 100.0);
        let (a, b) = (pyr(&flat, 3), pyr(&flat, 3));
        let t = track_one(
            &a,
            &b,
            Vec2::new(64.0, 64.0),
            Vec2::new(64.0, 64.0),
            &KltConfig::default(),
        );
        assert_eq!(t.status, KltStatus::IllConditioned);
    }

    #[test]
    fn a_one_dimensional_template_is_rejected_aperture_problem() {
        // Vertical stripes: displacement along y is unobservable, and a tracker
        // that reports one is inventing it.
        let stripes = render(128, 128, |x, _| 128.0 + 100.0 * (0.2 * x).sin());
        let (a, b) = (pyr(&stripes, 2), pyr(&stripes, 2));
        let t = track_one(
            &a,
            &b,
            Vec2::new(64.0, 64.0),
            Vec2::new(64.0, 64.0),
            &KltConfig::default(),
        );
        assert_eq!(t.status, KltStatus::IllConditioned);
    }

    #[test]
    fn a_track_leaving_the_frame_is_out_of_bounds_not_a_pose() {
        let (a, b) = shifted_pair(160, 160, 0.0, 0.0, 2);
        let p = Vec2::new(80.0, 80.0);
        // Aim the prediction outside the image; the solver must not clamp its
        // way to a plausible-looking answer.
        let t = track_one(&a, &b, p, Vec2::new(300.0, 80.0), &KltConfig::default());
        assert_eq!(t.status, KltStatus::OutOfBounds);
    }

    #[test]
    fn high_photometric_error_is_rejected() {
        let a = render(256, 256, texture);
        // Same geometry, inverted contrast: the flow solve is happy, the patch
        // is not the same patch.
        let b = render(256, 256, |x, y| 255.0 - texture(x, y));
        let (pa, pb) = (pyr(&a, 3), pyr(&b, 3));
        let p = Vec2::new(128.0, 128.0);
        let t = track_one(&pa, &pb, p, p, &KltConfig::default());
        assert!(!t.ok(), "{t:?}");
    }

    #[test]
    fn batch_tracking_matches_point_by_point() {
        let (a, b) = shifted_pair(256, 256, 2.0, 3.0, 3);
        let pts: Vec<Vec2> = (0..8)
            .map(|i| Vec2::new(70.0 + 15.0 * i as Scalar, 90.0 + 9.0 * i as Scalar))
            .collect();
        let cfg = KltConfig::default();
        let batch = track(&a, &b, &pts, None, &cfg);
        assert_eq!(batch.len(), pts.len());
        for (i, p) in pts.iter().enumerate() {
            let single = track_one(&a, &b, *p, *p, &cfg);
            assert_eq!(batch[i].px.x.to_bits(), single.px.x.to_bits());
            assert_eq!(batch[i].status, single.status);
        }
    }

    #[test]
    fn guesses_are_applied_per_point() {
        let (a, b) = shifted_pair(320, 320, 15.0, 0.0, 3);
        let pts = vec![Vec2::new(150.0, 150.0), Vec2::new(180.0, 200.0)];
        let guesses = vec![Vec2::new(165.0, 150.0), Vec2::new(195.0, 200.0)];
        let out = track(&a, &b, &pts, Some(&guesses), &KltConfig::default());
        for (i, t) in out.iter().enumerate() {
            assert!(t.ok(), "{t:?}");
            assert!((t.px - (pts[i] + Vec2::new(15.0, 0.0))).norm() < 0.2);
        }
    }

    #[test]
    fn identical_inputs_give_bit_identical_output() {
        let (a, b) = shifted_pair(192, 192, 1.7, -0.9, 3);
        let pts: Vec<Vec2> = (0..12)
            .map(|i| Vec2::new(60.0 + 8.0 * i as Scalar, 70.0 + 5.0 * i as Scalar))
            .collect();
        let cfg = KltConfig::default();
        let x = track(&a, &b, &pts, None, &cfg);
        let y = track(&a, &b, &pts, None, &cfg);
        for (p, q) in x.iter().zip(y.iter()) {
            assert_eq!(p.px.x.to_bits(), q.px.x.to_bits());
            assert_eq!(p.px.y.to_bits(), q.px.y.to_bits());
            assert_eq!(p.error.to_bits(), q.error.to_bits());
        }
    }
}
