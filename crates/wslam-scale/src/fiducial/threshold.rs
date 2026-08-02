//! Adaptive thresholding — stage one of the detector.
//!
//! A global threshold fails on the first frame where one side of the tag is in
//! shadow, which on a handheld phone is most of them. The AprilTag 3 approach
//! is a tiled local extremum: split the image into small tiles, take the
//! minimum and maximum intensity over each tile's 3x3 tile neighbourhood, and
//! binarise at their midpoint. Tiles whose neighbourhood has no contrast are
//! marked [`SKIP`] rather than forced to a value — assigning them black or
//! white would fabricate component boundaries in flat regions and hand the quad
//! fitter garbage.

use wslam_core::GrayImage;

/// Pixel binarised as dark.
pub const BLACK: u8 = 0;
/// Pixel binarised as light.
pub const WHITE: u8 = 255;
/// Pixel in a region with too little local contrast to binarise.
pub const SKIP: u8 = 127;

/// A binarised image: every pixel is [`BLACK`], [`WHITE`] or [`SKIP`].
#[derive(Debug, Clone)]
pub struct Binary {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

impl Binary {
    /// Image width.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }
    /// Image height.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }
    /// Row-major labels.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }
    /// Label at `(x, y)`; [`SKIP`] outside the image.
    #[inline]
    #[must_use]
    pub fn at(&self, x: u32, y: u32) -> u8 {
        if x >= self.width || y >= self.height {
            SKIP
        } else {
            self.data[(y * self.width + x) as usize]
        }
    }
}

/// Tiled adaptive threshold.
///
/// `tile` is the tile edge in pixels; `min_contrast` is the smallest
/// (max - min) over a tile neighbourhood that will be binarised at all.
///
/// Returns an all-[`SKIP`] image for degenerate sizes rather than panicking —
/// a 1x1 frame is not a detector failure worth unwinding for.
#[must_use]
pub fn adaptive_threshold(image: &GrayImage, tile: u32, min_contrast: u8) -> Binary {
    let (w, h) = (image.width(), image.height());
    let tile = tile.max(1);
    let mut out = Binary {
        width: w,
        height: h,
        data: vec![SKIP; (w as usize) * (h as usize)],
    };
    if w == 0 || h == 0 {
        return out;
    }

    let tw = w.div_ceil(tile) as usize;
    let th = h.div_ceil(tile) as usize;
    let mut tile_min = vec![255u8; tw * th];
    let mut tile_max = vec![0u8; tw * th];

    for y in 0..h {
        let ty = (y / tile) as usize;
        for x in 0..w {
            let tx = (x / tile) as usize;
            let v = image.at(x as i32, y as i32);
            let i = ty * tw + tx;
            tile_min[i] = tile_min[i].min(v);
            tile_max[i] = tile_max[i].max(v);
        }
    }

    // Dilate the extrema over a 3x3 tile neighbourhood. Without this, a tile
    // wholly inside a black cell has zero contrast and would be skipped even
    // though its neighbours resolve it unambiguously.
    //
    // At the image border the window is *shifted inward* rather than
    // truncated. Truncating leaves an edge tile with a 2x3 window and a corner
    // tile with 2x2 — four ninths of the support — exactly where there is
    // least evidence to work with. A tag whose border cell is larger than two
    // tiles then makes the corner window land wholly inside one cell, which
    // reads as zero contrast and SKIPs a region that is in fact unambiguous.
    // Shifting keeps the support constant everywhere; it costs an off-centre
    // window at the border, which is far cheaper than a hole in the mask.
    let window_start = |t: usize, n: usize| -> usize {
        // Whole axis when the grid is narrower than the window.
        n.saturating_sub(3).min(t.saturating_sub(1))
    };
    let mut blur_min = vec![255u8; tw * th];
    let mut blur_max = vec![0u8; tw * th];
    for ty in 0..th {
        let y0 = window_start(ty, th);
        let y1 = (y0 + 3).min(th);
        for tx in 0..tw {
            let x0 = window_start(tx, tw);
            let x1 = (x0 + 3).min(tw);
            let (mut lo, mut hi) = (255u8, 0u8);
            for ny in y0..y1 {
                for nx in x0..x1 {
                    let i = ny * tw + nx;
                    lo = lo.min(tile_min[i]);
                    hi = hi.max(tile_max[i]);
                }
            }
            blur_min[ty * tw + tx] = lo;
            blur_max[ty * tw + tx] = hi;
        }
    }

    for y in 0..h {
        let ty = (y / tile) as usize;
        for x in 0..w {
            let tx = (x / tile) as usize;
            let i = ty * tw + tx;
            let (lo, hi) = (blur_min[i], blur_max[i]);
            if hi.saturating_sub(lo) < min_contrast {
                continue; // already SKIP
            }
            let mid = ((lo as u16 + hi as u16) / 2) as u8;
            let v = image.at(x as i32, y as i32);
            out.data[(y * w + x) as usize] = if v > mid { WHITE } else { BLACK };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checker(w: u32, h: u32, cell: u32, lo: u8, hi: u8) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let on = ((x / cell) + (y / cell)) % 2 == 0;
                img.data_mut()[(y * w + x) as usize] = if on { hi } else { lo };
            }
        }
        img
    }

    #[test]
    fn a_uniform_image_is_entirely_skipped() {
        let mut img = GrayImage::new(32, 32);
        img.data_mut().fill(180);
        let b = adaptive_threshold(&img, 4, 25);
        assert!(
            b.data().iter().all(|&v| v == SKIP),
            "flat region must not be binarised"
        );
    }

    #[test]
    fn a_checkerboard_binarises_to_the_checkerboard() {
        let img = checker(32, 32, 8, 30, 220);
        let b = adaptive_threshold(&img, 4, 25);
        for y in 0..32u32 {
            for x in 0..32u32 {
                let expect = if ((x / 8) + (y / 8)) % 2 == 0 {
                    WHITE
                } else {
                    BLACK
                };
                assert_eq!(b.at(x, y), expect, "({x},{y})");
            }
        }
    }

    /// Local, not global: a lit half and a shadowed half must both binarise
    /// correctly even though the shadowed white is darker than the lit black.
    ///
    /// This test previously sampled four points and demanded the stripe colour
    /// at each; one of them, `x = 34`, is **not achievable by any centred local
    /// threshold**, and the assertion was wrong rather than merely tight. The
    /// derivation, since "we relaxed it" is not an acceptable answer:
    ///
    /// - `x = 34` is shadowed white, value 90. Calling it WHITE requires a
    ///   window with `(min + max) / 2 < 90`, i.e. `min + max < 180`.
    /// - The only other value in the shadowed half is 20, in the stripe
    ///   `[40, 47]`. So a window that has any contrast at all must reach
    ///   `x >= 40`.
    /// - Centred on 34, reaching `x = 40` means also reaching `x = 28` — back
    ///   across the illumination step, where black is 170. That forces
    ///   `max >= 170`, hence `mid >= 95 > 90`, hence BLACK.
    /// - Shrinking the window to `[32, 39]` instead puts it wholly inside one
    ///   stripe: zero contrast, SKIP.
    ///
    /// Two pixels from a step in illumination *larger than the local
    /// black/white contrast*, no local statistic escapes the transition band.
    /// The property the detector actually depends on is that the band is
    /// **bounded by the window width** and everything outside it is correct on
    /// both sides of the step — so that is what is asserted, over every pixel
    /// rather than four, which is a strictly stronger test than the original.
    #[test]
    fn survives_an_illumination_gradient_a_global_threshold_would_fail() {
        const LIT_WHITE: u8 = 250;
        const LIT_BLACK: u8 = 170;
        const SHADOW_WHITE: u8 = 90;
        const SHADOW_BLACK: u8 = 20;

        let (w, h) = (64u32, 32u32);
        let tile = 4u32;
        let stripe = 8u32;
        let step_x = w / 2; // illumination discontinuity, on a stripe boundary

        let value = |x: u32| -> u8 {
            match (x >= step_x, (x / stripe) % 2 == 0) {
                (false, true) => LIT_WHITE,
                (false, false) => LIT_BLACK,
                (true, true) => SHADOW_WHITE,
                (true, false) => SHADOW_BLACK,
            }
        };
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.data_mut()[(y * w + x) as usize] = value(x);
            }
        }

        // The premise, asserted rather than asserted-in-a-comment: one global
        // midpoint cannot work here, so passing below is evidence of locality.
        let global_mid = (u16::from(SHADOW_BLACK) + u16::from(LIT_WHITE)) / 2;
        assert!(
            u16::from(LIT_BLACK) > global_mid && u16::from(SHADOW_WHITE) < global_mid,
            "the image must be one a global threshold gets backwards"
        );

        let b = adaptive_threshold(&img, tile, 25);

        // A pixel in tile `t` is thresholded from tiles `[t-1, t+1]`, so its
        // window lies wholly on one side of the step exactly when the pixel is
        // at least `tile` pixels clear of it. That is the whole transition
        // band: 8 of 64 columns here, and the assertion covers the other 56.
        let mut asserted = 0u32;
        for y in 0..h {
            for x in 0..w {
                if x >= step_x - tile && x < step_x + tile {
                    continue;
                }
                let expect = if (x / stripe) % 2 == 0 { WHITE } else { BLACK };
                assert_eq!(b.at(x, y), expect, "({x},{y}) value {}", value(x));
                asserted += 1;
            }
        }
        assert_eq!(asserted, (w - 2 * tile) * h, "the band must not widen");
    }

    #[test]
    fn low_contrast_edges_are_skipped_not_amplified() {
        // 4 levels of contrast is noise, not an edge.
        let img = checker(32, 32, 8, 126, 130);
        let b = adaptive_threshold(&img, 4, 25);
        assert!(b.data().iter().all(|&v| v == SKIP));
    }

    #[test]
    fn out_of_bounds_reads_are_skip_not_a_panic() {
        let b = adaptive_threshold(&checker(16, 16, 4, 0, 255), 4, 25);
        assert_eq!(b.at(999, 0), SKIP);
        assert_eq!(b.at(0, 999), SKIP);
    }

    #[test]
    fn ragged_image_sizes_are_covered_to_the_last_pixel() {
        // 4 does not divide 30: the trailing strip must still be thresholded.
        let img = checker(30, 22, 6, 20, 235);
        let b = adaptive_threshold(&img, 4, 25);
        assert_ne!(b.at(29, 21), SKIP);
    }
}
