//! Shi-Tomasi corner detection with grid-bucketed selection.
//!
//! The response is the smaller eigenvalue of the windowed structure tensor
//! (Shi & Tomasi 1994), which is the direct answer to "how well conditioned is
//! the KLT normal-equation matrix at this pixel" — the same `H` [`crate::klt`]
//! inverts. Harris' determinant-minus-trace approximation exists to avoid a
//! square root that costs nothing here.
//!
//! **Selection is grid-bucketed, and that is not a refinement.** Ranking the
//! whole image by response puts every feature in the one heavily textured
//! corner of the frame, and a bundle of features spanning a small solid angle
//! constrains rotation well and translation barely at all — the pose solve goes
//! ill-conditioned along the depth direction while every individual track looks
//! healthy. Round-robin over grid cells buys spatial spread at the cost of some
//! average response, which is the right trade for pose observability.

use wslam_core::{GrayImage, Scalar, Vec2};

/// One detected corner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Corner {
    /// Sub-pixel location in the image the detector ran on.
    pub px: Vec2,
    /// Shi-Tomasi response — the minimum eigenvalue of the structure tensor,
    /// averaged over the window so it is comparable between window sizes.
    pub response: Scalar,
}

/// Detector tuning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CornerConfig {
    /// Upper bound on returned corners.
    pub max_corners: usize,
    /// Response floor as a fraction of the strongest response in the image.
    /// OpenCV's `goodFeaturesToTrack` semantics.
    pub quality_level: Scalar,
    /// Minimum separation between accepted corners, in pixels.
    pub min_distance: Scalar,
    /// Half-width of the structure-tensor window; the window is
    /// `(2r+1) x (2r+1)`.
    pub block_radius: u32,
    /// Bucket grid width. `1` disables horizontal bucketing.
    pub grid_cols: u32,
    /// Bucket grid height. `1` disables vertical bucketing.
    pub grid_rows: u32,
    /// Pixels excluded at the image border. Must exceed `block_radius` or the
    /// clamped-border gradients leak into the response.
    pub border: u32,
    /// Whether to run [`refine_subpixel`] on accepted corners.
    pub subpixel: bool,
    /// Half-width of the sub-pixel refinement window.
    pub subpixel_radius: u32,
    /// Sub-pixel iteration cap.
    pub subpixel_iterations: u32,
}

impl Default for CornerConfig {
    fn default() -> Self {
        CornerConfig {
            max_corners: 300,
            quality_level: 0.01,
            min_distance: 12.0,
            block_radius: 3,
            grid_cols: 8,
            grid_rows: 6,
            border: 8,
            subpixel: true,
            subpixel_radius: 4,
            subpixel_iterations: 12,
        }
    }
}

/// Per-pixel Shi-Tomasi response.
#[derive(Debug, Clone)]
pub struct ResponseMap {
    width: u32,
    height: u32,
    data: Vec<f32>,
}

impl ResponseMap {
    /// Map width.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }
    /// Map height.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }
    /// Row-major responses.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    /// Response at an integer pixel, clamped to the map bounds.
    #[inline]
    #[must_use]
    pub fn at(&self, x: i32, y: i32) -> Scalar {
        let x = x.clamp(0, self.width as i32 - 1) as usize;
        let y = y.clamp(0, self.height as i32 - 1) as usize;
        self.data[y * self.width as usize + x] as Scalar
    }
    /// Largest response anywhere in the map. Zero for a featureless image.
    #[must_use]
    pub fn max(&self) -> Scalar {
        self.data.iter().fold(0.0f32, |a, &b| a.max(b)) as Scalar
    }
}

/// Shi-Tomasi response over a whole image.
///
/// Gradients are central differences on `u8` data, so every product is a
/// multiple of `0.25` and the box sums below `2^24 * 0.25` are *exact* in `f32`
/// — the map is bit-reproducible without paying for `f64`.
#[must_use]
pub fn shi_tomasi_response_map(image: &GrayImage, block_radius: u32) -> ResponseMap {
    let w = image.width();
    let h = image.height();
    let n = (w as usize) * (h as usize);
    let (mut ixx, mut ixy, mut iyy) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);

    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let gx = 0.5 * (image.at(x + 1, y) as f32 - image.at(x - 1, y) as f32);
            let gy = 0.5 * (image.at(x, y + 1) as f32 - image.at(x, y - 1) as f32);
            let i = y as usize * w as usize + x as usize;
            ixx[i] = gx * gx;
            ixy[i] = gx * gy;
            iyy[i] = gy * gy;
        }
    }

    box_blur_sum(&mut ixx, w, h, block_radius);
    box_blur_sum(&mut ixy, w, h, block_radius);
    box_blur_sum(&mut iyy, w, h, block_radius);

    let side = 2 * block_radius + 1;
    let inv_area = 1.0f32 / (side * side) as f32;
    let mut data = vec![0f32; n];
    for i in 0..n {
        let (a, b, c) = (ixx[i] * inv_area, ixy[i] * inv_area, iyy[i] * inv_area);
        // Smaller eigenvalue of [[a, b], [b, c]]. Clamped at zero: the
        // discriminant is non-negative in exact arithmetic, and rounding must
        // not produce a negative response the threshold logic would mishandle.
        let t = a + c;
        let d = ((a - c) * (a - c) + 4.0 * b * b).max(0.0).sqrt();
        data[i] = (0.5 * (t - d)).max(0.0);
    }

    ResponseMap {
        width: w,
        height: h,
        data,
    }
}

/// Separable box *sum* (not average) with radius `r`, clamping at the border.
fn box_blur_sum(buf: &mut [f32], w: u32, h: u32, r: u32) {
    let (w, h, r) = (w as usize, h as usize, r as usize);
    let mut scratch = vec![0f32; w.max(h)];
    // Clamped border: replicate the edge sample for the taps that fall outside,
    // so every sum covers 2r+1 taps and the response is not artificially small
    // at the margin.
    let span = |src: &[f32], i: usize, n: usize| -> f32 {
        let lo = i.saturating_sub(r);
        let hi = (i + r + 1).min(n);
        let inner: f32 = src[lo..hi].iter().sum();
        inner
            + r.saturating_sub(i) as f32 * src[0]
            + (i + r + 1).saturating_sub(n) as f32 * src[n - 1]
    };

    for y in 0..h {
        let row = &mut buf[y * w..(y + 1) * w];
        scratch[..w].copy_from_slice(row);
        for (x, out) in row.iter_mut().enumerate() {
            *out = span(&scratch[..w], x, w);
        }
    }
    for x in 0..w {
        for (y, s) in scratch.iter_mut().enumerate().take(h) {
            *s = buf[y * w + x];
        }
        for y in 0..h {
            buf[y * w + x] = span(&scratch[..h], y, h);
        }
    }
}

/// Detect corners, avoiding anything within `min_distance` of `occupied`.
///
/// `occupied` is how the tracker refills: pass the features it is still
/// tracking and the detector fills the gaps around them instead of restacking
/// the same corners.
///
/// Returned corners are in **round-robin cell order**, not response order. A
/// caller that truncates the list therefore keeps the spatial spread; sorting
/// by response would silently undo the bucketing.
#[must_use]
pub fn detect(image: &GrayImage, config: &CornerConfig, occupied: &[Vec2]) -> Vec<Corner> {
    let map = shi_tomasi_response_map(image, config.block_radius);
    detect_in_map(image, &map, config, occupied)
}

/// [`detect`] against a response map the caller already computed.
#[must_use]
pub fn detect_in_map(
    image: &GrayImage,
    map: &ResponseMap,
    config: &CornerConfig,
    occupied: &[Vec2],
) -> Vec<Corner> {
    let peak = map.max();
    if peak <= 0.0 || config.max_corners == 0 {
        return Vec::new(); // a constant image has no corners, not weak ones
    }
    let threshold = peak * config.quality_level;

    let border = config.border.max(config.block_radius + 1) as i32;
    let w = map.width() as i32;
    let h = map.height() as i32;
    if w - 2 * border <= 0 || h - 2 * border <= 0 {
        return Vec::new();
    }

    let cols = config.grid_cols.max(1) as usize;
    let rows = config.grid_rows.max(1) as usize;
    let mut cells: Vec<Vec<Corner>> = vec![Vec::new(); cols * rows];

    for y in border..h - border {
        for x in border..w - border {
            let r = map.at(x, y);
            if r < threshold {
                continue;
            }
            // 3x3 local maximum, non-strict. Step edges produce two-pixel-wide
            // response plateaus (a central-difference gradient straddles the
            // step), and a strict test rejects every pixel of a plateau; the
            // duplicates are collapsed by the min-distance check below.
            let mut is_max = true;
            'nb: for dy in -1..=1 {
                for dx in -1..=1 {
                    if (dx != 0 || dy != 0) && map.at(x + dx, y + dy) > r {
                        is_max = false;
                        break 'nb;
                    }
                }
            }
            if !is_max {
                continue;
            }
            let cx = (x as usize * cols) / w as usize;
            let cy = (y as usize * rows) / h as usize;
            cells[cy * cols + cx].push(Corner {
                px: Vec2::new(x as Scalar, y as Scalar),
                response: r,
            });
        }
    }

    // Strongest first; ties broken by position so the output does not depend on
    // scan order or sort stability (spec.md §6, replay must be bit-exact).
    for cell in &mut cells {
        cell.sort_by(|a, b| {
            b.response
                .total_cmp(&a.response)
                .then(a.px.y.total_cmp(&b.px.y))
                .then(a.px.x.total_cmp(&b.px.x))
        });
    }

    let min_d2 = config.min_distance * config.min_distance;
    let mut accepted: Vec<Vec2> = occupied.to_vec();
    let mut out: Vec<Corner> = Vec::with_capacity(config.max_corners);
    let mut cursor = vec![0usize; cells.len()];

    // Round-robin: one corner per cell per pass. Guarantees the spread rather
    // than hoping a per-cell quota produces it.
    loop {
        let mut progressed = false;
        for (ci, cell) in cells.iter().enumerate() {
            if out.len() >= config.max_corners {
                break;
            }
            while cursor[ci] < cell.len() {
                let cand = cell[cursor[ci]];
                cursor[ci] += 1;
                if accepted
                    .iter()
                    .any(|p| (p - cand.px).norm_squared() < min_d2)
                {
                    continue;
                }
                accepted.push(cand.px);
                out.push(cand);
                progressed = true;
                break;
            }
        }
        if !progressed || out.len() >= config.max_corners {
            break;
        }
    }

    if config.subpixel {
        for c in &mut out {
            c.px = refine_subpixel(
                image,
                c.px,
                config.subpixel_radius,
                config.subpixel_iterations,
                1e-3,
            );
        }
    }
    out
}

/// Sub-pixel corner refinement by gradient orthogonality.
///
/// At an ideal corner every image gradient in the neighbourhood is orthogonal
/// to the vector from the corner to the sample that produced it, so
/// `sum_i g_i g_i^T (q - p_i) = 0` and `q` is the weighted intersection of the
/// edges. Exact for two straight step edges, unlike a parabola fit on the
/// response map, which is only exact if the response happens to be quadratic.
///
/// The iteration is clamped: a refinement that wants to walk further than the
/// window is not refining, and the unrefined location is the safer answer.
#[must_use]
pub fn refine_subpixel(
    image: &GrayImage,
    px: Vec2,
    radius: u32,
    iterations: u32,
    epsilon: Scalar,
) -> Vec2 {
    let r = radius.max(1) as i32;
    let denom = (r as Scalar + 1.0) * (r as Scalar + 1.0);
    let mut q = px;
    for _ in 0..iterations {
        let mut a = nalgebra::Matrix2::<Scalar>::zeros();
        let mut b = Vec2::zeros();
        for dy in -r..=r {
            for dx in -r..=r {
                let p = Vec2::new(q.x + dx as Scalar, q.y + dy as Scalar);
                let g = image.gradient_bilinear(p.x, p.y);
                // Bartlett-style separable taper. Polynomial rather than a
                // Gaussian on purpose: `exp` is not guaranteed bit-identical
                // between the native and wasm libm, and spec.md §6 L3 wants any
                // native/wasm divergence to be attributable to a port bug.
                let wx = 1.0 - (dx * dx) as Scalar / denom;
                let wy = 1.0 - (dy * dy) as Scalar / denom;
                let gg = g * g.transpose() * (wx * wy);
                a += gg;
                b += gg * p;
            }
        }
        let Some(inv) = a.try_inverse() else {
            break; // rank-deficient neighbourhood: an edge, not a corner
        };
        let next = inv * b;
        if !next.x.is_finite() || !next.y.is_finite() {
            break;
        }
        if (next - px).norm() > r as Scalar {
            break;
        }
        let step = (next - q).norm();
        q = next;
        if step < epsilon {
            break;
        }
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyramid::{Pyramid, PyramidConfig};
    use wslam_core::CameraIntrinsics;

    fn filled(w: u32, h: u32, v: u8) -> GrayImage {
        GrayImage::from_vec(w, h, vec![v; (w * h) as usize])
    }

    /// White square covering pixels `[lo, hi)` in both axes on a black field.
    /// The continuous corner therefore sits at `lo - 0.5`, `hi - 0.5`.
    fn square(size: u32, lo: u32, hi: u32) -> GrayImage {
        let mut img = filled(size, size, 0);
        let data = img.data_mut();
        for y in lo..hi {
            for x in lo..hi {
                data[(y * size + x) as usize] = 255;
            }
        }
        img
    }

    #[test]
    fn constant_image_has_zero_response_at_every_pyramid_level() {
        for v in [0u8, 64, 200, 255] {
            let img = filled(320, 240, v);
            let p = Pyramid::build(
                &img,
                &CameraIntrinsics::from_focal(300.0, 320, 240),
                &PyramidConfig::default(),
            );
            assert!(p.len() >= 3);
            for (l, level) in p.levels().iter().enumerate() {
                let map = shi_tomasi_response_map(&level.image, 3);
                assert_eq!(map.max(), 0.0, "level {l} of a constant {v} image");
                assert!(detect(&level.image, &CornerConfig::default(), &[]).is_empty());
            }
        }
    }

    #[test]
    fn finds_the_four_corners_of_a_square_and_not_its_edges() {
        let img = square(96, 24, 72);
        let cfg = CornerConfig {
            max_corners: 8,
            quality_level: 0.05,
            min_distance: 6.0,
            grid_cols: 2,
            grid_rows: 2,
            ..CornerConfig::default()
        };
        let found = detect(&img, &cfg, &[]);
        assert_eq!(found.len(), 4, "got {found:?}");

        let truth = [
            Vec2::new(23.5, 23.5),
            Vec2::new(71.5, 23.5),
            Vec2::new(23.5, 71.5),
            Vec2::new(71.5, 71.5),
        ];
        for t in truth {
            let best = found
                .iter()
                .map(|c| (c.px - t).norm())
                .fold(Scalar::INFINITY, Scalar::min);
            assert!(best < 1.0, "no corner within 1 px of {t:?}; got {found:?}");
        }
    }

    #[test]
    fn edge_midpoints_have_negligible_response() {
        // An ideal straight edge has exactly one non-zero eigenvalue, so the
        // Shi-Tomasi response there is zero — this is what separates it from a
        // gradient-magnitude detector.
        let img = square(96, 24, 72);
        let map = shi_tomasi_response_map(&img, 3);
        let corner = map.at(24, 24).max(map.at(23, 23));
        for (x, y) in [(48, 24), (48, 71), (24, 48), (71, 48)] {
            assert!(
                map.at(x, y) < 1e-3 * corner,
                "edge midpoint ({x},{y}) response {} vs corner {corner}",
                map.at(x, y)
            );
        }
    }

    #[test]
    fn subpixel_recovers_a_half_pixel_corner() {
        // The square's corner is at (23.5, 23.5); the response plateau makes the
        // integer argmax land a full half pixel away, so this measures the
        // refinement and not the detector.
        let img = square(96, 24, 72);
        let raw = Vec2::new(23.0, 23.0);
        let refined = refine_subpixel(&img, raw, 4, 20, 1e-4);
        let truth = Vec2::new(23.5, 23.5);
        // A hard step corner is the worst case for gradient orthogonality: the
        // two edges are one-sided, so the window sees corner-adjacent diagonal
        // gradients that no straight edge accounts for. ~0.07 px of residual
        // bias is inherent, against 0.71 px for the integer argmax.
        assert!(
            (refined - truth).norm() < 0.1,
            "refined {refined:?} vs truth {truth:?}"
        );
        assert!((refined - truth).norm() < 0.2 * (raw - truth).norm());
    }

    #[test]
    fn subpixel_recovers_an_off_grid_edge_intersection() {
        // Two anti-aliased step edges crossing at a known non-half-integer
        // point. Area coverage gives an exact continuous model to compare to.
        let (cx, cy) = (40.37, 30.62);
        let coverage = |a: Scalar| a.clamp(0.0, 1.0);
        let mut img = GrayImage::new(80, 64);
        let data = img.data_mut();
        for y in 0..64 {
            for x in 0..80 {
                let fx = coverage(x as Scalar + 0.5 - cx);
                let fy = coverage(y as Scalar + 0.5 - cy);
                // Checkerboard corner: bright where exactly one side is past.
                let v = fx + fy - 2.0 * fx * fy;
                data[y * 80 + x] = (v * 255.0).round() as u8;
            }
        }
        let refined = refine_subpixel(&img, Vec2::new(40.0, 31.0), 5, 25, 1e-5);
        assert!(
            (refined - Vec2::new(cx, cy)).norm() < 0.1,
            "refined {refined:?} vs truth ({cx}, {cy})"
        );
    }

    #[test]
    fn subpixel_leaves_a_textureless_patch_where_it_found_it() {
        let img = filled(64, 64, 90);
        let p = Vec2::new(32.0, 32.0);
        assert_eq!(refine_subpixel(&img, p, 4, 10, 1e-4), p);
    }

    /// Dense high-contrast checkerboard in the top-left quadrant, sparse
    /// low-contrast checkerboard everywhere else. Ranking by response alone
    /// puts everything in the top-left twice over: more corners *and* stronger.
    fn lopsided_texture(size: u32) -> GrayImage {
        let mut img = filled(size, size, 128);
        let half = size / 2;
        let data = img.data_mut();
        for y in 0..size {
            for x in 0..size {
                let hot = x < half && y < half;
                let (period, amp) = if hot { (6u32, 127i32) } else { (24u32, 40i32) };
                let checker = ((x / period) % 2) ^ ((y / period) % 2);
                let v = 128 + if checker == 0 { amp } else { -amp };
                data[(y * size + x) as usize] = v.clamp(0, 255) as u8;
            }
        }
        img
    }

    fn quadrant_counts(corners: &[Corner], size: Scalar) -> [usize; 4] {
        let mut q = [0usize; 4];
        for c in corners {
            let i = usize::from(c.px.x >= size * 0.5) + 2 * usize::from(c.px.y >= size * 0.5);
            q[i] += 1;
        }
        q
    }

    #[test]
    fn grid_bucketing_spreads_features_across_quadrants() {
        let size = 256;
        let img = lopsided_texture(size);
        let cfg = CornerConfig {
            max_corners: 80,
            min_distance: 8.0,
            grid_cols: 8,
            grid_rows: 8,
            ..CornerConfig::default()
        };
        let bucketed = detect(&img, &cfg, &[]);
        assert!(bucketed.len() >= 60, "only {} corners", bucketed.len());
        let q = quadrant_counts(&bucketed, size as Scalar);
        for (i, &n) in q.iter().enumerate() {
            assert!(n >= 8, "quadrant {i} got only {n} of {q:?}");
        }
    }

    #[test]
    fn without_bucketing_features_clump_in_the_textured_quadrant() {
        // The control for the test above. If this ever stops clumping the
        // fixture has gone soft and the bucketing test proves nothing.
        let size = 256;
        let img = lopsided_texture(size);
        let cfg = CornerConfig {
            max_corners: 80,
            min_distance: 8.0,
            grid_cols: 1,
            grid_rows: 1,
            ..CornerConfig::default()
        };
        let flat = detect(&img, &cfg, &[]);
        let q = quadrant_counts(&flat, size as Scalar);
        assert!(
            q[0] as f64 > 0.85 * flat.len() as f64,
            "expected clumping, got {q:?}"
        );
    }

    #[test]
    fn min_distance_is_respected() {
        let img = lopsided_texture(256);
        let cfg = CornerConfig {
            max_corners: 200,
            min_distance: 14.0,
            subpixel: false, // refinement can move a corner a fraction of a pixel
            ..CornerConfig::default()
        };
        let found = detect(&img, &cfg, &[]);
        assert!(found.len() > 20);
        for (i, a) in found.iter().enumerate() {
            for b in &found[i + 1..] {
                assert!(
                    (a.px - b.px).norm() >= 14.0,
                    "{:?} and {:?} are {} apart",
                    a.px,
                    b.px,
                    (a.px - b.px).norm()
                );
            }
        }
    }

    #[test]
    fn occupied_positions_are_avoided() {
        let img = lopsided_texture(256);
        let cfg = CornerConfig {
            max_corners: 60,
            min_distance: 10.0,
            subpixel: false,
            ..CornerConfig::default()
        };
        let first = detect(&img, &cfg, &[]);
        let occupied: Vec<Vec2> = first.iter().map(|c| c.px).collect();
        let refill = detect(&img, &cfg, &occupied);
        for c in &refill {
            for o in &occupied {
                assert!((c.px - o).norm() >= 10.0);
            }
        }
        assert!(!refill.is_empty(), "refill should still find gaps");
    }

    #[test]
    fn detection_is_bit_identical_across_runs() {
        let img = lopsided_texture(192);
        let cfg = CornerConfig::default();
        let a = detect(&img, &cfg, &[]);
        let b = detect(&img, &cfg, &[]);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.px.x.to_bits(), y.px.x.to_bits());
            assert_eq!(x.px.y.to_bits(), y.px.y.to_bits());
            assert_eq!(x.response.to_bits(), y.response.to_bits());
        }
    }

    #[test]
    fn border_larger_than_the_image_yields_nothing_instead_of_panicking() {
        let img = lopsided_texture(32);
        let cfg = CornerConfig {
            border: 40,
            ..CornerConfig::default()
        };
        assert!(detect(&img, &cfg, &[]).is_empty());
    }

    #[test]
    fn max_corners_zero_is_honoured() {
        let img = lopsided_texture(128);
        let cfg = CornerConfig {
            max_corners: 0,
            ..CornerConfig::default()
        };
        assert!(detect(&img, &cfg, &[]).is_empty());
    }
}
