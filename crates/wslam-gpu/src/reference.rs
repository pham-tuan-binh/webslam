//! CPU reference for every WGSL kernel in this crate.
//!
//! spec.md §6 L3 says of the tracker port: *"Any divergence is a port bug, not
//! an algorithm result."* That is only a checkable claim if there is something
//! to diverge *from*. This module is that something: a straight-line
//! implementation of each kernel, written to mirror the shader's arithmetic
//! **including its summation order**, because float addition is not associative
//! and "same algorithm, different order" produces differences large enough to
//! hide a real bug behind.
//!
//! Precision differs deliberately: the reference accumulates in `f64` where the
//! GPU has only `f32`. The equivalence tests therefore state a tolerance rather
//! than claiming bit-exactness. Matching the *order* keeps that tolerance near
//! the f32 epsilon, so a genuine port bug still stands out by orders of
//! magnitude.
//!
//! Two pieces of the corner path are shared rather than duplicated:
//! [`select_corners`] runs on the host in both the GPU and reference pipelines,
//! because ranking and spacing a few hundred candidates is not kernel work.
//! What the GPU actually computes — the response image and the per-cell
//! winner — is duplicated here in full and compared directly.

use wslam_core::GrayImage;

use crate::{FlowConfig, FlowResult};

/// Box radius of the Shi-Tomasi structure tensor. Must equal `RADIUS` in
/// `corners.wgsl`.
pub const CORNER_RADIUS: i32 = 1;

/// Pixels at the border where the response is undefined: the structure-tensor
/// box plus the one pixel the central difference reaches beyond it.
pub const CORNER_MARGIN: i32 = CORNER_RADIUS + 1;

/// Grid cell size in pixels for the bucketed non-max suppression.
///
/// Chosen so a 640x480 frame yields 1200 buckets: enough spatial spread for a
/// well-conditioned PnP, and a 9.6 kB readback against 300 kB for the image.
pub const CORNER_CELL_PX: u32 = 16;

/// Workgroup width of `klt.wgsl`. The reference partitions the tracking window
/// across this many virtual lanes so its summation order matches the GPU's.
pub const KLT_WORKGROUP: usize = 64;

/// Minimum `lambda_min / window_area` for the Lucas-Kanade system to be
/// considered solvable.
///
/// Normalised by window area so it does not have to be retuned when
/// [`FlowConfig::window`] changes. A flat patch scores exactly zero; a patch
/// with a one-grey-level-per-pixel gradient in both directions scores ~0.25.
pub const MIN_STRUCTURE_EIGENVALUE: f64 = 1e-3;

/// A single-channel `f64` image — one pyramid level of the CPU reference.
///
/// The GPU stores levels as `f32`; this is the same data at higher precision,
/// which is what makes it usable as an oracle rather than a second opinion.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatImage {
    width: u32,
    height: u32,
    data: Vec<f64>,
}

impl FloatImage {
    /// Allocate a zeroed level.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        FloatImage {
            width,
            height,
            data: vec![0.0; (width as usize) * (height as usize)],
        }
    }

    /// Wrap an existing tightly-packed buffer.
    ///
    /// # Panics
    /// If `data.len() != width * height`.
    #[must_use]
    pub fn from_vec(width: u32, height: u32, data: Vec<f64>) -> Self {
        assert_eq!(
            data.len(),
            (width as usize) * (height as usize),
            "FloatImage buffer size mismatch"
        );
        FloatImage {
            width,
            height,
            data,
        }
    }

    /// Level 0 of the pyramid: the luma bytes widened, exactly what
    /// `grayscale.wgsl`'s `unpack_luma` writes.
    #[must_use]
    pub fn from_gray(image: &GrayImage) -> Self {
        FloatImage {
            width: image.width(),
            height: image.height(),
            data: image.data().iter().map(|&v| v as f64).collect(),
        }
    }

    /// Width in pixels.
    #[inline]
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    #[inline]
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Row-major pixel data.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Mutable row-major pixel data.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Border-clamped nearest fetch, matching `fetch`/`fetch_prev` in the
    /// shaders and [`GrayImage::at`].
    #[inline]
    #[must_use]
    pub fn at(&self, x: i32, y: i32) -> f64 {
        let x = x.clamp(0, self.width as i32 - 1) as usize;
        let y = y.clamp(0, self.height as i32 - 1) as usize;
        self.data[y * self.width as usize + x]
    }

    /// Border-clamped bilinear sample, matching `sample_prev`/`sample_next`.
    #[must_use]
    pub fn sample_bilinear(&self, x: f64, y: f64) -> f64 {
        let bx = x.floor();
        let by = y.floor();
        let fx = x - bx;
        let fy = y - by;
        let x0 = bx as i32;
        let y0 = by as i32;
        let p00 = self.at(x0, y0);
        let p10 = self.at(x0 + 1, y0);
        let p01 = self.at(x0, y0 + 1);
        let p11 = self.at(x0 + 1, y0 + 1);
        let top = p00 + (p10 - p00) * fx;
        let bot = p01 + (p11 - p01) * fx;
        top + (bot - top) * fy
    }

    /// Central-difference gradient of the bilinear interpolant, step one pixel.
    #[must_use]
    pub fn gradient(&self, x: f64, y: f64) -> (f64, f64) {
        (
            0.5 * (self.sample_bilinear(x + 1.0, y) - self.sample_bilinear(x - 1.0, y)),
            0.5 * (self.sample_bilinear(x, y + 1.0) - self.sample_bilinear(x, y - 1.0)),
        )
    }
}

// ---------------------------------------------------------------------------
// grayscale.wgsl
// ---------------------------------------------------------------------------

/// Reference for `rgba_to_luma`.
///
/// The integer weights are not a micro-optimisation: they are the reason the
/// native and browser builds agree on every intensity, and they match
/// [`GrayImage::from_rgba`] byte for byte.
///
/// # Panics
/// If `rgba.len() != width * height * 4`.
#[must_use]
pub fn luma_from_rgba(width: u32, height: u32, rgba: &[u8]) -> FloatImage {
    let n = (width as usize) * (height as usize);
    assert_eq!(rgba.len(), n * 4, "RGBA buffer size mismatch");
    let data = rgba
        .chunks_exact(4)
        .map(|px| {
            let y = (19_595 * px[0] as u32 + 38_470 * px[1] as u32 + 7_471 * px[2] as u32) >> 16;
            y as f64
        })
        .collect();
    FloatImage::from_vec(width, height, data)
}

// ---------------------------------------------------------------------------
// pyramid.wgsl
// ---------------------------------------------------------------------------

/// Dimensions of every pyramid level: level `k` is `max(w >> k, 1)` wide.
#[must_use]
pub fn pyramid_dims(width: u32, height: u32, levels: u32) -> Vec<(u32, u32)> {
    (0..levels)
        .map(|k| ((width >> k).max(1), (height >> k).max(1)))
        .collect()
}

/// Reference for `downsample_2x2`.
#[must_use]
pub fn downsample_2x2(src: &FloatImage) -> FloatImage {
    let dw = (src.width() / 2).max(1);
    let dh = (src.height() / 2).max(1);
    let mut dst = FloatImage::new(dw, dh);
    for y in 0..dh {
        for x in 0..dw {
            let sx = (x * 2) as i32;
            let sy = (y * 2) as i32;
            // Parenthesised to pin the order the shader uses.
            let s = ((src.at(sx, sy) + src.at(sx + 1, sy)) + src.at(sx, sy + 1))
                + src.at(sx + 1, sy + 1);
            dst.data_mut()[(y * dw + x) as usize] = s * 0.25;
        }
    }
    dst
}

/// Map a level-0 pixel coordinate onto pyramid level `level`.
///
/// `downsample_2x2` averages source pixels `2x` and `2x+1` into destination
/// pixel `x`, which places the destination pixel centre at source coordinate
/// `2x + 0.5`. Inverting that gives `(p + 0.5) / 2^level - 0.5` — the same
/// half-pixel convention [`wslam_core::CameraIntrinsics::scaled`] uses for the
/// principal point. It is linear in `p`, so a *displacement* scales by exactly
/// `2^-level` and the KLT guess can be doubled between levels with no
/// correction term.
///
/// Bouguet's plain `p / 2^level` differs by up to half a pixel at the coarsest
/// level, which the finer levels usually absorb. "Usually" is the problem: it
/// eats into the convergence basin exactly when the motion is large enough to
/// have needed the pyramid in the first place.
#[inline]
#[must_use]
pub fn level_coordinate(p: f64, level: u32) -> f64 {
    let s = 1.0 / (1u32 << level) as f64;
    (p + 0.5) * s - 0.5
}

/// Full pyramid from an 8-bit frame — the reference for `unpack_luma` followed
/// by `levels - 1` `downsample_2x2` dispatches.
///
/// # Panics
/// If `levels == 0`.
#[must_use]
pub fn build_pyramid(image: &GrayImage, levels: u32) -> Vec<FloatImage> {
    assert!(levels >= 1, "a pyramid needs at least one level");
    let mut out = Vec::with_capacity(levels as usize);
    out.push(FloatImage::from_gray(image));
    for k in 1..levels as usize {
        out.push(downsample_2x2(&out[k - 1]));
    }
    out
}

// ---------------------------------------------------------------------------
// corners.wgsl
// ---------------------------------------------------------------------------

/// Reference for `shi_tomasi`: smaller eigenvalue of the box-summed structure
/// tensor, zero inside [`CORNER_MARGIN`] of the border.
///
/// Returns one value per pixel, row-major. This buffer is the thing that never
/// crosses the bus in the GPU pipeline.
#[must_use]
pub fn shi_tomasi_response(image: &FloatImage) -> Vec<f64> {
    let w = image.width() as i32;
    let h = image.height() as i32;
    let mut out = vec![0.0; (w * h) as usize];
    for y in CORNER_MARGIN..(h - CORNER_MARGIN) {
        for x in CORNER_MARGIN..(w - CORNER_MARGIN) {
            let (mut a, mut b, mut c) = (0.0, 0.0, 0.0);
            for dy in -CORNER_RADIUS..=CORNER_RADIUS {
                for dx in -CORNER_RADIUS..=CORNER_RADIUS {
                    let ix = 0.5 * (image.at(x + dx + 1, y + dy) - image.at(x + dx - 1, y + dy));
                    let iy = 0.5 * (image.at(x + dx, y + dy + 1) - image.at(x + dx, y + dy - 1));
                    a += ix * ix;
                    b += ix * iy;
                    c += iy * iy;
                }
            }
            let half_trace = 0.5 * (a + c);
            let half_diff = 0.5 * (a - c);
            let r = half_trace - (half_diff * half_diff + b * b).sqrt();
            out[(y * w + x) as usize] = r.max(0.0);
        }
    }
    out
}

/// The single candidate a grid cell contributes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellWinner {
    /// Pixel column.
    pub x: u32,
    /// Pixel row.
    pub y: u32,
    /// Shi-Tomasi response there.
    pub response: f64,
}

/// Reference for `cell_reduce` + `cell_pick`.
///
/// One entry per grid cell in row-major cell order; `None` where every response
/// in the cell was zero. Ties inside a cell resolve to the lowest raster index,
/// which is what makes the result reproducible independently of dispatch order.
#[must_use]
pub fn cell_winners(
    response: &[f64],
    width: u32,
    height: u32,
    cell_px: u32,
) -> Vec<Option<CellWinner>> {
    let grid_w = width.div_ceil(cell_px);
    let grid_h = height.div_ceil(cell_px);
    let mut best: Vec<Option<CellWinner>> = vec![None; (grid_w * grid_h) as usize];
    for y in 0..height {
        for x in 0..width {
            let r = response[(y * width + x) as usize];
            if r <= 0.0 {
                continue;
            }
            let cell = ((y / cell_px) * grid_w + (x / cell_px)) as usize;
            let replace = match best[cell] {
                None => true,
                // Strictly greater only: an equal response leaves the earlier
                // (lower raster index) winner in place.
                Some(w) => r > w.response,
            };
            if replace {
                best[cell] = Some(CellWinner { x, y, response: r });
            }
        }
    }
    best
}

/// Rank, threshold and space out the per-cell candidates.
///
/// Runs on the host in both pipelines — it is bookkeeping over a few hundred
/// points, not kernel work, and duplicating it would only mean two places to
/// get the ordering wrong.
///
/// - `quality` is relative to the strongest candidate in the frame, the
///   `qualityLevel` convention.
/// - `min_distance` is enforced greedily in descending response order.
///
/// Returns `(x, y, response)` sorted strongest first.
#[must_use]
pub fn select_corners(
    winners: &[Option<CellWinner>],
    max_corners: usize,
    quality: f32,
    min_distance: f32,
) -> Vec<(f32, f32, f32)> {
    let mut candidates: Vec<CellWinner> = winners.iter().flatten().copied().collect();
    if candidates.is_empty() || max_corners == 0 {
        return Vec::new();
    }
    let peak = candidates.iter().map(|c| c.response).fold(0.0f64, f64::max);
    let threshold = (quality as f64) * peak;

    // Descending response, ties broken by raster order so the output is a
    // deterministic function of the response image alone.
    candidates.sort_by(|a, b| {
        b.response
            .partial_cmp(&a.response)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| (a.y, a.x).cmp(&(b.y, b.x)))
    });

    let min_d2 = (min_distance as f64) * (min_distance as f64);
    let mut out: Vec<(f32, f32, f32)> = Vec::with_capacity(max_corners.min(candidates.len()));
    for c in candidates {
        if out.len() >= max_corners {
            break;
        }
        if c.response < threshold || c.response <= 0.0 {
            continue;
        }
        if min_distance > 0.0 {
            let too_close = out.iter().any(|&(ox, oy, _)| {
                let dx = ox as f64 - c.x as f64;
                let dy = oy as f64 - c.y as f64;
                dx * dx + dy * dy < min_d2
            });
            if too_close {
                continue;
            }
        }
        out.push((c.x as f32, c.y as f32, c.response as f32));
    }
    out
}

/// End-to-end CPU corner detection: the reference for
/// [`crate::ImagePipeline::detect_corners`].
#[must_use]
pub fn detect_corners(
    image: &FloatImage,
    max_corners: usize,
    quality: f32,
    min_distance: f32,
) -> Vec<(f32, f32, f32)> {
    let response = shi_tomasi_response(image);
    let winners = cell_winners(&response, image.width(), image.height(), CORNER_CELL_PX);
    select_corners(&winners, max_corners, quality, min_distance)
}

// ---------------------------------------------------------------------------
// klt.wgsl
// ---------------------------------------------------------------------------

/// The Lucas-Kanade normal equations at one window position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LkSystem {
    /// Structure tensor `[Ixx, Ixy, Iyy]` of the template over the window.
    pub tensor: [f64; 3],
    /// Right-hand side `sum(residual * grad_template)`.
    pub rhs: [f64; 2],
    /// Mean absolute residual over the window, in grey levels.
    pub error: f64,
}

/// Accumulate the normal equations exactly as `klt.wgsl` does: a strided
/// partition across [`KLT_WORKGROUP`] lanes, then a binary tree reduction.
///
/// `p` is the template centre *at this level*; `flow` is the current
/// displacement estimate at this level.
#[must_use]
pub fn lk_normal_equations(
    prev: &FloatImage,
    next: &FloatImage,
    p: (f64, f64),
    flow: (f64, f64),
    window: u32,
) -> LkSystem {
    let win = window as usize;
    let half = (window / 2) as i32;
    let wcount = win * win;

    let mut lane = [[0.0f64; 6]; KLT_WORKGROUP];
    for (t, slot) in lane.iter_mut().enumerate() {
        let mut k = t;
        while k < wcount {
            let dx = ((k % win) as i32 - half) as f64;
            let dy = ((k / win) as i32 - half) as f64;
            let tx = p.0 + dx;
            let ty = p.1 + dy;
            let tmpl = prev.sample_bilinear(tx, ty);
            let (ix, iy) = prev.gradient(tx, ty);
            let warped = next.sample_bilinear(tx + flow.0, ty + flow.1);
            let d = tmpl - warped;
            slot[0] += ix * ix;
            slot[1] += ix * iy;
            slot[2] += iy * iy;
            slot[3] += d * ix;
            slot[4] += d * iy;
            slot[5] += d.abs();
            k += KLT_WORKGROUP;
        }
    }

    let mut stride = KLT_WORKGROUP / 2;
    while stride > 0 {
        let (head, tail) = lane.split_at_mut(stride);
        for (dst, src) in head.iter_mut().zip(&tail[..stride]) {
            for (d, s) in dst.iter_mut().zip(src) {
                *d += *s;
            }
        }
        stride >>= 1;
    }

    LkSystem {
        tensor: [lane[0][0], lane[0][1], lane[0][2]],
        rhs: [lane[0][3], lane[0][4]],
        error: lane[0][5] / wcount as f64,
    }
}

/// Photometric cost `0.5 * sum(next(p + w + flow) - prev(p + w))^2` over the
/// window. Exists so the Jacobians below can be checked against central finite
/// differences of something computed independently of them.
#[must_use]
pub fn lk_cost(
    prev: &FloatImage,
    next: &FloatImage,
    p: (f64, f64),
    flow: (f64, f64),
    window: u32,
) -> f64 {
    let win = window as usize;
    let half = (window / 2) as i32;
    let mut sum = 0.0;
    for k in 0..win * win {
        let dx = ((k % win) as i32 - half) as f64;
        let dy = ((k / win) as i32 - half) as f64;
        let d = next.sample_bilinear(p.0 + dx + flow.0, p.1 + dy + flow.1)
            - prev.sample_bilinear(p.0 + dx, p.1 + dy);
        sum += d * d;
    }
    0.5 * sum
}

/// Analytic gradient of [`lk_cost`] with respect to `flow`, in the
/// forward-additive form (gradient of the *warped* image).
///
/// The tracker uses the template gradient instead — the two coincide at the
/// optimum and the template form lets `G` be hoisted — so this exists to
/// validate the residual Jacobian against finite differences rather than to be
/// called on any hot path.
#[must_use]
pub fn lk_cost_gradient(
    prev: &FloatImage,
    next: &FloatImage,
    p: (f64, f64),
    flow: (f64, f64),
    window: u32,
) -> [f64; 2] {
    let win = window as usize;
    let half = (window / 2) as i32;
    let mut gx = 0.0;
    let mut gy = 0.0;
    for k in 0..win * win {
        let dx = ((k % win) as i32 - half) as f64;
        let dy = ((k / win) as i32 - half) as f64;
        let wx = p.0 + dx + flow.0;
        let wy = p.1 + dy + flow.1;
        let d = next.sample_bilinear(wx, wy) - prev.sample_bilinear(p.0 + dx, p.1 + dy);
        let (jx, jy) = next.gradient(wx, wy);
        gx += d * jx;
        gy += d * jy;
    }
    [gx, gy]
}

/// Reference for `klt.wgsl`'s `track` entry point.
///
/// `prev` and `next` are pyramids of equal depth, coarsest level last (index
/// `levels - 1`). Points are in level-0 pixel coordinates.
///
/// # Panics
/// If the two pyramids have different depths, or either is empty.
#[must_use]
pub fn track_flow(
    prev: &[FloatImage],
    next: &[FloatImage],
    points: &[(f32, f32)],
    config: &FlowConfig,
) -> Vec<FlowResult> {
    assert!(!prev.is_empty(), "pyramid must have at least one level");
    assert_eq!(prev.len(), next.len(), "pyramid depth mismatch");
    points
        .iter()
        .map(|&(x, y)| track_point(prev, next, (x as f64, y as f64), config))
        .collect()
}

fn track_point(
    prev: &[FloatImage],
    next: &[FloatImage],
    p0: (f64, f64),
    config: &FlowConfig,
) -> FlowResult {
    let wcount = (config.window as f64) * (config.window as f64);
    let half = (config.window / 2) as f64;
    // One pass beyond the solver iterations so `error` describes the flow that
    // is actually returned — see `klt.wgsl`.
    let passes = config.iterations + 1;

    let mut flow = (0.0f64, 0.0f64);
    let mut failed = false;
    let mut error = 0.0f64;

    for li in (0..prev.len()).rev() {
        let level = li as u32;
        let p = (level_coordinate(p0.0, level), level_coordinate(p0.1, level));
        let mut converged = false;

        for it in 0..passes {
            let sys = lk_normal_equations(&prev[li], &next[li], p, flow, config.window);
            error = sys.error;
            let [a, b, c] = sys.tensor;
            let det = a * c - b * b;
            let half_trace = 0.5 * (a + c);
            let half_diff = 0.5 * (a - c);
            let lmin = half_trace - (half_diff * half_diff + b * b).sqrt();
            if it < config.iterations && !failed && !converged {
                if det <= 0.0 || lmin < MIN_STRUCTURE_EIGENVALUE * wcount {
                    failed = true;
                } else {
                    let nx = (c * sys.rhs[0] - b * sys.rhs[1]) / det;
                    let ny = (a * sys.rhs[1] - b * sys.rhs[0]) / det;
                    flow.0 += nx;
                    flow.1 += ny;
                    if (nx * nx + ny * ny).sqrt() < config.epsilon as f64 {
                        converged = true;
                    }
                }
            }
        }

        if li > 0 {
            flow.0 *= 2.0;
            flow.1 *= 2.0;
        }
    }

    let tracked = (p0.0 + flow.0, p0.1 + flow.1);
    let base = &prev[0];
    let inside = tracked.0 >= half
        && tracked.1 >= half
        && tracked.0 <= (base.width() - 1) as f64 - half
        && tracked.1 <= (base.height() - 1) as f64 - half;
    let ok = !failed && inside && error <= config.max_error as f64;

    FlowResult {
        x: tracked.0 as f32,
        y: tracked.1 as f32,
        ok,
        error: error as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testdata::{analytic_texture, bilinear_image, checkerboard};
    use approx::assert_relative_eq;

    fn textured(w: u32, h: u32) -> GrayImage {
        analytic_texture(w, h, 0.0, 0.0)
    }

    // -- grayscale -----------------------------------------------------------

    #[test]
    fn rgba_luma_matches_core_conversion() {
        // The GPU kernel and GrayImage::from_rgba must agree bit for bit, or
        // the browser path and the replay path see different images.
        let rgba: Vec<u8> = (0..16 * 4).map(|i| (i * 7 % 251) as u8).collect();
        let gpu_ref = luma_from_rgba(4, 4, &rgba);
        let core = GrayImage::from_rgba(4, 4, &rgba);
        for i in 0..16 {
            assert_eq!(gpu_ref.data()[i], core.data()[i] as f64);
        }
    }

    // -- pyramid -------------------------------------------------------------

    #[test]
    fn pyramid_level_k_has_dimensions_shifted_by_k() {
        let img = GrayImage::new(640, 480);
        let pyr = build_pyramid(&img, 5);
        for (k, level) in pyr.iter().enumerate() {
            assert_eq!(level.width(), 640 >> k, "level {k} width");
            assert_eq!(level.height(), 480 >> k, "level {k} height");
        }
    }

    #[test]
    fn pyramid_never_collapses_to_zero_dimension() {
        // 5 levels of a 6x3 image would reach 0 px without the clamp; the
        // dispatch grid would then be empty and the level unwritten.
        let img = GrayImage::new(6, 3);
        let pyr = build_pyramid(&img, 5);
        for level in &pyr {
            assert!(level.width() >= 1 && level.height() >= 1);
        }
        assert_eq!(pyramid_dims(6, 3, 5).last().copied(), Some((1, 1)));
    }

    #[test]
    fn downsample_averages_a_known_2x2_block() {
        let src = FloatImage::from_vec(4, 2, vec![0.0, 100.0, 10.0, 20.0, 200.0, 40.0, 30.0, 50.0]);
        let dst = downsample_2x2(&src);
        assert_eq!((dst.width(), dst.height()), (2, 1));
        assert_relative_eq!(dst.at(0, 0), (0.0 + 100.0 + 200.0 + 40.0) / 4.0);
        assert_relative_eq!(dst.at(1, 0), (10.0 + 20.0 + 30.0 + 50.0) / 4.0);
    }

    #[test]
    fn downsample_of_a_constant_image_is_that_constant() {
        let src = FloatImage::from_vec(8, 8, vec![37.0; 64]);
        let dst = downsample_2x2(&src);
        for &v in dst.data() {
            assert_relative_eq!(v, 37.0);
        }
    }

    #[test]
    fn level_coordinate_inverts_the_downsample_geometry() {
        // On a ramp I(x) = x the box filter is exact: level k has value
        // 2^k * x + (2^k - 1)/2, so the source coordinate X shows up at
        // destination coordinate level_coordinate(X, k). Sampling both and
        // demanding the same intensity pins the convention to a closed form
        // rather than to whatever the tracker tolerates.
        let mut src = FloatImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                src.data_mut()[y * 64 + x] = x as f64;
            }
        }
        let pyr = {
            let mut v = vec![src];
            for k in 1..4 {
                v.push(downsample_2x2(&v[k - 1]));
            }
            v
        };
        for level in 1..4u32 {
            for &x in &[16.0f64, 20.25, 31.5, 40.75] {
                let lx = level_coordinate(x, level);
                assert_relative_eq!(
                    pyr[level as usize].sample_bilinear(lx, level_coordinate(32.0, level)),
                    pyr[0].sample_bilinear(x, 32.0),
                    epsilon = 1e-9
                );
            }
        }
        // Displacements must scale by exactly 2^-level, or the between-level
        // guess doubling needs a correction term it does not have.
        for level in 0..5u32 {
            let scale = 1.0 / (1u32 << level) as f64;
            assert_relative_eq!(
                level_coordinate(17.0, level) - level_coordinate(9.0, level),
                8.0 * scale,
                epsilon = 1e-12
            );
        }
        assert_relative_eq!(level_coordinate(7.25, 0), 7.25);
    }

    #[test]
    fn odd_width_downsample_clamps_the_last_column() {
        // 5 -> 2: the last destination column reads source columns 2 and 3, and
        // column 4 is dropped. Getting this wrong reads out of bounds.
        let src = FloatImage::from_vec(5, 1, vec![0.0, 0.0, 8.0, 4.0, 255.0]);
        let dst = downsample_2x2(&src);
        assert_eq!(dst.width(), 2);
        // Row clamping folds y+1 back onto y for a single-row image.
        assert_relative_eq!(dst.at(1, 0), (8.0 + 4.0 + 8.0 + 4.0) / 4.0);
    }

    // -- Shi-Tomasi ----------------------------------------------------------

    #[test]
    fn linear_ramp_has_zero_shi_tomasi_response() {
        // A ramp is the canonical aperture-problem case: one eigenvalue large,
        // the other exactly zero. Any nonzero response here means the two
        // eigenvalues have been swapped or the tensor is not being formed.
        let mut img = FloatImage::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                img.data_mut()[y * 32 + x] = 3.0 * x as f64;
            }
        }
        let r = shi_tomasi_response(&img);
        for (i, &v) in r.iter().enumerate() {
            assert!(v.abs() < 1e-9, "pixel {i} scored {v} on a pure ramp");
        }
    }

    #[test]
    fn diagonal_ramp_also_has_zero_response() {
        // Same aperture problem rotated 45 degrees, which a tensor built from
        // axis-aligned gradients alone would get wrong.
        let mut img = FloatImage::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                img.data_mut()[y * 32 + x] = 2.0 * (x as f64 + y as f64);
            }
        }
        let r = shi_tomasi_response(&img);
        assert!(r.iter().all(|v| v.abs() < 1e-9));
    }

    #[test]
    fn constant_image_has_zero_response_everywhere() {
        let img = FloatImage::from_vec(16, 16, vec![120.0; 256]);
        assert!(shi_tomasi_response(&img).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn response_is_zero_inside_the_border_margin() {
        let img = FloatImage::from_gray(&textured(24, 24));
        let r = shi_tomasi_response(&img);
        for y in 0..24i32 {
            for x in 0..24i32 {
                let border = x < CORNER_MARGIN
                    || y < CORNER_MARGIN
                    || x >= 24 - CORNER_MARGIN
                    || y >= 24 - CORNER_MARGIN;
                if border {
                    assert_eq!(r[(y * 24 + x) as usize], 0.0, "({x},{y}) is border");
                }
            }
        }
    }

    #[test]
    fn checkerboard_corners_land_on_the_intersections() {
        // Square wave with period 2*S: the only places with gradient energy in
        // both directions are the block intersections, and we know where we put
        // them.
        const S: u32 = 20;
        const N: u32 = 160;
        let img = checkerboard(N, N, S);
        let level0 = FloatImage::from_gray(&img);
        let corners = detect_corners(&level0, 256, 0.05, 8.0);

        // Intersections far enough from the border for the response to exist.
        let intersections: Vec<(f64, f64)> = (1..N / S)
            .flat_map(|i| (1..N / S).map(move |j| ((i * S) as f64, (j * S) as f64)))
            .filter(|&(x, y)| {
                x > CORNER_MARGIN as f64 + 1.0
                    && y > CORNER_MARGIN as f64 + 1.0
                    && x < (N - 2) as f64
                    && y < (N - 2) as f64
            })
            .collect();
        assert_eq!(intersections.len(), 49);
        assert_eq!(
            corners.len(),
            intersections.len(),
            "expected one corner per intersection, got {corners:?}"
        );

        // The response plateaus over the 2x2 pixels straddling an intersection,
        // so "exactly where we put them" means within that plateau.
        for &(cx, cy, _) in &corners {
            let d = intersections
                .iter()
                .map(|&(ix, iy)| ((ix - cx as f64).powi(2) + (iy - cy as f64).powi(2)).sqrt())
                .fold(f64::INFINITY, f64::min);
            assert!(
                d <= 1.5,
                "corner ({cx},{cy}) is {d:.2} px from any intersection"
            );
        }
    }

    #[test]
    fn quality_threshold_is_relative_to_the_strongest_corner() {
        let winners = vec![
            Some(CellWinner {
                x: 0,
                y: 0,
                response: 100.0,
            }),
            Some(CellWinner {
                x: 50,
                y: 0,
                response: 40.0,
            }),
            Some(CellWinner {
                x: 100,
                y: 0,
                response: 5.0,
            }),
        ];
        assert_eq!(select_corners(&winners, 10, 0.3, 0.0).len(), 2);
        assert_eq!(select_corners(&winners, 10, 0.01, 0.0).len(), 3);
        assert_eq!(select_corners(&winners, 10, 0.5, 0.0).len(), 1);
    }

    #[test]
    fn min_distance_suppresses_the_weaker_of_a_close_pair() {
        let winners = vec![
            Some(CellWinner {
                x: 10,
                y: 10,
                response: 100.0,
            }),
            Some(CellWinner {
                x: 13,
                y: 10,
                response: 90.0,
            }),
            Some(CellWinner {
                x: 40,
                y: 10,
                response: 80.0,
            }),
        ];
        let kept = select_corners(&winners, 10, 0.0, 5.0);
        assert_eq!(kept.len(), 2);
        assert_eq!((kept[0].0, kept[0].1), (10.0, 10.0));
        assert_eq!((kept[1].0, kept[1].1), (40.0, 10.0));
    }

    #[test]
    fn cell_winner_ties_resolve_to_the_lowest_raster_index() {
        // Two pixels in the same cell with identical response. Without a
        // deterministic tie-break the GPU would pick by dispatch order.
        let w = 8u32;
        let mut response = vec![0.0; 64];
        response[(2 * w + 5) as usize] = 7.0;
        response[(3 * w + 1) as usize] = 7.0;
        let winners = cell_winners(&response, w, 8, 16);
        assert_eq!(winners.len(), 1);
        assert_eq!(
            winners[0].unwrap(),
            CellWinner {
                x: 5,
                y: 2,
                response: 7.0
            }
        );
    }

    #[test]
    fn empty_response_yields_no_corners() {
        let img = FloatImage::from_vec(32, 32, vec![90.0; 1024]);
        assert!(detect_corners(&img, 100, 0.01, 3.0).is_empty());
    }

    #[test]
    fn max_corners_zero_returns_nothing() {
        let img = FloatImage::from_gray(&textured(64, 64));
        assert!(detect_corners(&img, 0, 0.01, 0.0).is_empty());
    }

    // -- Lucas-Kanade --------------------------------------------------------

    #[test]
    fn lk_cost_gradient_matches_central_differences() {
        // The residual Jacobian, checked against a cost computed without it.
        let img = bilinear_image(40, 40, 30.0, 2.5, -1.75, 0.31);
        let p = (20.5, 18.5);
        let flow = (0.3, -0.2);
        let analytic = lk_cost_gradient(&img, &img, p, flow, 9);
        let h = 1e-4;
        let num_x = (lk_cost(&img, &img, p, (flow.0 + h, flow.1), 9)
            - lk_cost(&img, &img, p, (flow.0 - h, flow.1), 9))
            / (2.0 * h);
        let num_y = (lk_cost(&img, &img, p, (flow.0, flow.1 + h), 9)
            - lk_cost(&img, &img, p, (flow.0, flow.1 - h), 9))
            / (2.0 * h);
        assert_relative_eq!(analytic[0], num_x, max_relative = 1e-6);
        assert_relative_eq!(analytic[1], num_y, max_relative = 1e-6);
    }

    #[test]
    fn structure_tensor_is_the_hessian_of_the_photometric_cost() {
        // At zero flow with prev == next the residual vanishes, so the
        // Gauss-Newton Hessian is the exact Hessian. On a bilinear image the
        // one-pixel central difference is the exact derivative, so this is an
        // equality, not an approximation.
        let img = bilinear_image(40, 40, 30.0, 2.5, -1.75, 0.31);
        let p = (20.5, 18.5);
        let sys = lk_normal_equations(&img, &img, p, (0.0, 0.0), 9);
        let h = 1e-2;
        let e0 = lk_cost(&img, &img, p, (0.0, 0.0), 9);
        let hxx = (lk_cost(&img, &img, p, (h, 0.0), 9) + lk_cost(&img, &img, p, (-h, 0.0), 9)
            - 2.0 * e0)
            / (h * h);
        let hyy = (lk_cost(&img, &img, p, (0.0, h), 9) + lk_cost(&img, &img, p, (0.0, -h), 9)
            - 2.0 * e0)
            / (h * h);
        let hxy = (lk_cost(&img, &img, p, (h, h), 9) + lk_cost(&img, &img, p, (-h, -h), 9)
            - lk_cost(&img, &img, p, (h, -h), 9)
            - lk_cost(&img, &img, p, (-h, h), 9))
            / (4.0 * h * h);
        assert_relative_eq!(sys.tensor[0], hxx, max_relative = 1e-6);
        assert_relative_eq!(sys.tensor[2], hyy, max_relative = 1e-6);
        assert_relative_eq!(sys.tensor[1], hxy, max_relative = 1e-6);
    }

    #[test]
    fn residual_is_zero_and_rhs_vanishes_at_the_true_alignment() {
        let img = bilinear_image(40, 40, 30.0, 2.5, -1.75, 0.31);
        let sys = lk_normal_equations(&img, &img, (20.5, 18.5), (0.0, 0.0), 9);
        assert_relative_eq!(sys.rhs[0], 0.0, epsilon = 1e-9);
        assert_relative_eq!(sys.rhs[1], 0.0, epsilon = 1e-9);
        assert_relative_eq!(sys.error, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn strided_reduction_sums_every_window_pixel_exactly_once() {
        // The lane partition is what makes the GPU and CPU agree; if it drops
        // or double-counts a pixel the whole equivalence claim is void.
        // A window of ones over a unit-gradient image must sum to window^2.
        let mut img = FloatImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                img.data_mut()[y * 64 + x] = x as f64;
            }
        }
        for &window in &[3u32, 7, 15, 21] {
            let sys = lk_normal_equations(&img, &img, (32.0, 32.0), (0.0, 0.0), window);
            // Ix == 1 everywhere, so sum(Ix*Ix) == window^2.
            assert_relative_eq!(
                sys.tensor[0],
                (window * window) as f64,
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn recovers_a_known_subpixel_translation() {
        // Shift a smooth analytic texture by a known non-integer amount and ask
        // the tracker to find it. 0.05 px is the bar in spec.md's L3 tier-1
        // brief.
        const DX: f64 = 0.37;
        const DY: f64 = -0.62;
        let prev_img = analytic_texture(96, 96, 0.0, 0.0);
        let next_img = analytic_texture(96, 96, DX, DY);
        let prev = build_pyramid(&prev_img, 3);
        let next = build_pyramid(&next_img, 3);

        let cfg = FlowConfig::default();
        let points = [(40.0f32, 44.0f32), (52.0, 60.0), (33.0, 51.0)];
        let out = track_flow(&prev, &next, &points, &cfg);
        for (r, p) in out.iter().zip(points.iter()) {
            assert!(r.ok, "point {p:?} lost: {r:?}");
            assert!(
                (r.x as f64 - (p.0 as f64 + DX)).abs() < 0.05,
                "x error {:.4}",
                r.x as f64 - (p.0 as f64 + DX)
            );
            assert!(
                (r.y as f64 - (p.1 as f64 + DY)).abs() < 0.05,
                "y error {:.4}",
                r.y as f64 - (p.1 as f64 + DY)
            );
        }
    }

    #[test]
    fn recovers_a_multi_pixel_translation_only_because_of_the_pyramid() {
        // 9 px is far outside the basin of a 15 px window at full resolution;
        // it is the coarse levels that make it tractable. A single-level run
        // must fail, or the pyramid is not doing anything.
        const D: f64 = 9.0;
        let prev_img = analytic_texture(128, 128, 0.0, 0.0);
        let next_img = analytic_texture(128, 128, D, 0.0);
        let cfg = FlowConfig::default();
        let points = [(60.0f32, 64.0f32)];

        let deep = track_flow(
            &build_pyramid(&prev_img, 4),
            &build_pyramid(&next_img, 4),
            &points,
            &cfg,
        );
        assert!(deep[0].ok);
        assert!((deep[0].x as f64 - (60.0 + D)).abs() < 0.1, "{:?}", deep[0]);

        let flat = track_flow(
            &build_pyramid(&prev_img, 1),
            &build_pyramid(&next_img, 1),
            &points,
            &cfg,
        );
        assert!(
            (flat[0].x as f64 - (60.0 + D)).abs() > 1.0,
            "single level should not have found a 9 px shift: {:?}",
            flat[0]
        );
    }

    #[test]
    fn flat_region_is_rejected_rather_than_answered() {
        // The degenerate case: no gradient, singular system. Returning ok with
        // a fabricated displacement is the failure mode that poisons PnP.
        let flat = GrayImage::from_vec(64, 64, vec![100; 64 * 64]);
        let pyr = build_pyramid(&flat, 3);
        let out = track_flow(&pyr, &pyr, &[(32.0, 32.0)], &FlowConfig::default());
        assert!(!out[0].ok);
    }

    #[test]
    fn aperture_problem_is_rejected() {
        // A pure vertical edge fixes only the x displacement; lambda_min is
        // zero and the system must be declared unsolvable.
        let mut img = GrayImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                img.data_mut()[y * 64 + x] = if x < 32 { 20 } else { 220 };
            }
        }
        let pyr = build_pyramid(&img, 3);
        let out = track_flow(&pyr, &pyr, &[(32.0, 32.0)], &FlowConfig::default());
        assert!(!out[0].ok, "vertical edge is rank deficient: {:?}", out[0]);
    }

    #[test]
    fn point_leaving_the_frame_is_marked_lost() {
        let img = analytic_texture(64, 64, 0.0, 0.0);
        let pyr = build_pyramid(&img, 3);
        let cfg = FlowConfig::default();
        // Sits inside the border margin, so its window is not fully supported.
        let out = track_flow(&pyr, &pyr, &[(2.0, 32.0)], &cfg);
        assert!(!out[0].ok);
    }

    #[test]
    fn photometric_mismatch_trips_max_error() {
        // Same geometry, different brightness: the flow is right but the
        // residual is not, and max_error is the only thing that notices.
        let a = analytic_texture(96, 96, 0.0, 0.0);
        let mut b_img = a.clone();
        for v in b_img.data_mut() {
            *v = v.saturating_add(60);
        }
        let pa = build_pyramid(&a, 3);
        let pb = build_pyramid(&b_img, 3);
        let strict = FlowConfig {
            max_error: 5.0,
            ..FlowConfig::default()
        };
        let lax = FlowConfig {
            max_error: 250.0,
            ..FlowConfig::default()
        };
        let p = [(48.0f32, 48.0f32)];
        assert!(!track_flow(&pa, &pb, &p, &strict)[0].ok);
        assert!(track_flow(&pa, &pb, &p, &lax)[0].ok);
    }

    #[test]
    fn zero_displacement_round_trips_exactly() {
        let img = analytic_texture(96, 96, 0.0, 0.0);
        let pyr = build_pyramid(&img, 3);
        let points = [(40.0f32, 40.0f32), (55.0, 61.0)];
        let out = track_flow(&pyr, &pyr, &points, &FlowConfig::default());
        for (r, p) in out.iter().zip(points.iter()) {
            assert!(r.ok);
            assert_relative_eq!(r.x, p.0, epsilon = 1e-4);
            assert_relative_eq!(r.y, p.1, epsilon = 1e-4);
            assert_relative_eq!(r.error, 0.0, epsilon = 1e-4);
        }
    }
}
