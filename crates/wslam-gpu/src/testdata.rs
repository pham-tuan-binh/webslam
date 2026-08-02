//! Synthetic images with known answers, shared by the reference tests and the
//! GPU equivalence tests.
//!
//! Every generator here is analytic: the answer is known before the pipeline
//! runs. A fixture built by running part of the implementation would make the
//! tests agree with whatever the implementation happens to do.

use wslam_core::GrayImage;

use crate::reference::FloatImage;

/// Band-limited texture sampled analytically at a sub-pixel offset.
///
/// The shifted image is a true resampling of the continuous function, not an
/// interpolation of the unshifted one — otherwise a flow test would be
/// measuring its own interpolator against itself.
///
/// Periods are roughly 13 and 17 px: enough structure for a well-conditioned
/// structure tensor, smooth enough that bilinear resampling bias stays well
/// under the 0.05 px accuracy bar.
#[must_use]
pub fn analytic_texture(w: u32, h: u32, dx: f64, dy: f64) -> GrayImage {
    let mut img = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let fx = x as f64 - dx;
            let fy = y as f64 - dy;
            let v = 128.0
                + 55.0 * (fx * 0.48 + fy * 0.19).sin()
                + 45.0 * (fx * 0.21 - fy * 0.37).sin()
                + 25.0 * (fx * 0.09 + fy * 0.13).cos();
            img.data_mut()[(y * w + x) as usize] = v.clamp(0.0, 255.0) as u8;
        }
    }
    img
}

/// Square-wave checkerboard with blocks of `block` pixels. Corners are exactly
/// at the multiples of `block`.
#[must_use]
pub fn checkerboard(w: u32, h: u32, block: u32) -> GrayImage {
    let mut img = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let on = ((x / block) + (y / block)) % 2 == 0;
            img.data_mut()[(y * w + x) as usize] = if on { 230 } else { 25 };
        }
    }
    img
}

/// `I = a + b x + c y + d x y`.
///
/// Bilinear interpolation reproduces this class exactly, and its true partial
/// derivatives equal the one-pixel central difference exactly. That makes
/// analytic and finite-difference quantities comparable without a truncation
/// term muddying the tolerance.
#[must_use]
pub fn bilinear_image(w: u32, h: u32, a: f64, b: f64, c: f64, d: f64) -> FloatImage {
    let mut img = FloatImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img.data_mut()[(y * w + x) as usize] =
                a + b * x as f64 + c * y as f64 + d * x as f64 * y as f64;
        }
    }
    img
}
