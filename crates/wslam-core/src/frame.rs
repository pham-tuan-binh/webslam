//! Frames and the frame source interface.
//!
//! spec.md §6: *"The frame source is an interface, with live and replay
//! implementations. The same binary then runs live and replays a canned
//! trajectory bit-for-bit reproducibly."*

use crate::camera::CameraIntrinsics;
use crate::math::{Scalar, Vec2};
use crate::time::Timestamp;

/// Monotonic frame counter. Not a timestamp; frames may be dropped and the
/// index still increments, which is exactly what the clock model needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct FrameId(pub u64);

impl std::fmt::Display for FrameId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// An 8-bit single-channel image.
///
/// Luma only: everything above L2 works on intensity, and keeping colour out of
/// the core type stops it leaking into the tracking path where it would cost
/// bandwidth for nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrayImage {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

impl GrayImage {
    /// Allocate a black image.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        GrayImage {
            width,
            height,
            data: vec![0u8; (width as usize) * (height as usize)],
        }
    }

    /// Wrap an existing tightly-packed buffer.
    ///
    /// # Panics
    /// If `data.len() != width * height`.
    #[must_use]
    pub fn from_vec(width: u32, height: u32, data: Vec<u8>) -> Self {
        assert_eq!(
            data.len(),
            (width as usize) * (height as usize),
            "GrayImage buffer size mismatch"
        );
        GrayImage {
            width,
            height,
            data,
        }
    }

    /// Convert tightly-packed RGBA (the shape `getImageData` and WebGPU readback
    /// produce) to luma using the Rec. 601 weights.
    ///
    /// # Panics
    /// If `rgba.len() != width * height * 4`.
    #[must_use]
    pub fn from_rgba(width: u32, height: u32, rgba: &[u8]) -> Self {
        let n = (width as usize) * (height as usize);
        assert_eq!(rgba.len(), n * 4, "RGBA buffer size mismatch");
        let mut data = Vec::with_capacity(n);
        for px in rgba.chunks_exact(4) {
            // Integer weights: 0.299/0.587/0.114 scaled by 2^16, so the result
            // is bit-identical between native and wasm. Float weights are not.
            let y = (19_595 * px[0] as u32 + 38_470 * px[1] as u32 + 7_471 * px[2] as u32) >> 16;
            data.push(y as u8);
        }
        GrayImage {
            width,
            height,
            data,
        }
    }

    /// Image width in pixels.
    #[inline]
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }
    /// Image height in pixels.
    #[inline]
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }
    /// Raw luma bytes, row-major, tightly packed.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }
    /// Mutable raw luma bytes.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Nearest-neighbour fetch, clamped to the image bounds.
    #[inline]
    #[must_use]
    pub fn at(&self, x: i32, y: i32) -> u8 {
        let x = x.clamp(0, self.width as i32 - 1) as usize;
        let y = y.clamp(0, self.height as i32 - 1) as usize;
        self.data[y * self.width as usize + x]
    }

    /// Bilinear sample at sub-pixel coordinates, clamped at the border.
    ///
    /// The KLT inner loop lives here, so it is written to avoid bounds checks on
    /// the common in-interior path.
    #[must_use]
    pub fn sample_bilinear(&self, x: Scalar, y: Scalar) -> Scalar {
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = x - x0;
        let fy = y - y0;
        let x0 = x0 as i32;
        let y0 = y0 as i32;

        let w = self.width as i32;
        let h = self.height as i32;
        if x0 >= 0 && y0 >= 0 && x0 + 1 < w && y0 + 1 < h {
            let base = (y0 as usize) * (self.width as usize) + (x0 as usize);
            let p00 = self.data[base] as Scalar;
            let p10 = self.data[base + 1] as Scalar;
            let p01 = self.data[base + self.width as usize] as Scalar;
            let p11 = self.data[base + self.width as usize + 1] as Scalar;
            let top = p00 + (p10 - p00) * fx;
            let bot = p01 + (p11 - p01) * fx;
            top + (bot - top) * fy
        } else {
            let p00 = self.at(x0, y0) as Scalar;
            let p10 = self.at(x0 + 1, y0) as Scalar;
            let p01 = self.at(x0, y0 + 1) as Scalar;
            let p11 = self.at(x0 + 1, y0 + 1) as Scalar;
            let top = p00 + (p10 - p00) * fx;
            let bot = p01 + (p11 - p01) * fx;
            top + (bot - top) * fy
        }
    }

    /// Central-difference spatial gradient at sub-pixel coordinates.
    #[must_use]
    pub fn gradient_bilinear(&self, x: Scalar, y: Scalar) -> Vec2 {
        Vec2::new(
            0.5 * (self.sample_bilinear(x + 1.0, y) - self.sample_bilinear(x - 1.0, y)),
            0.5 * (self.sample_bilinear(x, y + 1.0) - self.sample_bilinear(x, y - 1.0)),
        )
    }

    /// Mean intensity. Used by the low-light tracking-state heuristic.
    #[must_use]
    pub fn mean_intensity(&self) -> Scalar {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().map(|&v| v as u64).sum::<u64>() as Scalar / self.data.len() as Scalar
    }

    /// Half-resolution copy by 2x2 box average — one pyramid level.
    ///
    /// Integer arithmetic with round-half-up, so native and wasm agree bit for
    /// bit (spec.md §6 L3: any divergence must be a port bug, and that claim is
    /// only checkable if the reference path is deterministic).
    #[must_use]
    pub fn downsample_half(&self) -> GrayImage {
        let nw = (self.width / 2).max(1);
        let nh = (self.height / 2).max(1);
        let mut out = GrayImage::new(nw, nh);
        for y in 0..nh {
            for x in 0..nw {
                let sx = (x * 2) as i32;
                let sy = (y * 2) as i32;
                let s = self.at(sx, sy) as u32
                    + self.at(sx + 1, sy) as u32
                    + self.at(sx, sy + 1) as u32
                    + self.at(sx + 1, sy + 1) as u32;
                out.data[(y * nw + x) as usize] = ((s + 2) / 4) as u8;
            }
        }
        out
    }
}

/// One camera frame in the unified timebase.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Delivery index.
    pub id: FrameId,
    /// Capture time in the unified timebase, as mapped by [`crate::TimeBase`].
    pub timestamp: Timestamp,
    /// Luma image.
    pub image: GrayImage,
}

impl Frame {
    /// Construct a frame.
    #[must_use]
    pub fn new(id: FrameId, timestamp: Timestamp, image: GrayImage) -> Self {
        Frame {
            id,
            timestamp,
            image,
        }
    }
}

/// Source of camera frames.
///
/// The live implementation is fed by the TypeScript shim through the wasm
/// boundary; the replay implementation reads a dataset. Nothing above this trait
/// can tell the difference, which is the whole point.
pub trait FrameSource {
    /// Next frame, or `None` at end of stream. Non-blocking: a live source
    /// returns `None` when no frame has arrived since the last call.
    fn next_frame(&mut self) -> Option<Frame>;

    /// Intrinsics if the source knows them (replay datasets do; a browser
    /// camera does not, which is why L2 exists).
    fn intrinsics_hint(&self) -> Option<CameraIntrinsics> {
        None
    }

    /// Whether frames arrive in real time. Replay sources return `false`, which
    /// lets the orchestrator skip real-time budget enforcement.
    fn is_live(&self) -> bool;

    /// Total frame count if known — replay only.
    fn len_hint(&self) -> Option<usize> {
        None
    }
}

/// In-memory frame source. Backs both unit tests and the "recorded frame
/// source fed into the browser build" separation trick from spec.md §6 Tier 4.
#[derive(Debug, Clone)]
pub struct ReplayFrameSource {
    frames: std::collections::VecDeque<Frame>,
    intrinsics: Option<CameraIntrinsics>,
    total: usize,
}

impl ReplayFrameSource {
    /// Build from a frame list.
    #[must_use]
    pub fn new(frames: Vec<Frame>, intrinsics: Option<CameraIntrinsics>) -> Self {
        let total = frames.len();
        ReplayFrameSource {
            frames: frames.into(),
            intrinsics,
            total,
        }
    }

    /// Append a frame — used when streaming a dataset in rather than loading it
    /// all at once.
    pub fn push(&mut self, frame: Frame) {
        self.frames.push_back(frame);
        self.total += 1;
    }

    /// Frames not yet consumed.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.frames.len()
    }
}

impl FrameSource for ReplayFrameSource {
    fn next_frame(&mut self) -> Option<Frame> {
        self.frames.pop_front()
    }
    fn intrinsics_hint(&self) -> Option<CameraIntrinsics> {
        self.intrinsics
    }
    fn is_live(&self) -> bool {
        false
    }
    fn len_hint(&self) -> Option<usize> {
        Some(self.total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bilinear_is_exact_at_pixel_centres() {
        let mut img = GrayImage::new(4, 4);
        for i in 0..16 {
            img.data_mut()[i] = (i * 16) as u8;
        }
        for y in 0..4 {
            for x in 0..4 {
                assert_relative_eq!(
                    img.sample_bilinear(x as f64, y as f64),
                    img.at(x, y) as f64,
                    epsilon = 1e-12
                );
            }
        }
    }

    #[test]
    fn bilinear_interpolates_linearly() {
        let img = GrayImage::from_vec(2, 2, vec![0, 100, 0, 100]);
        assert_relative_eq!(img.sample_bilinear(0.5, 0.0), 50.0, epsilon = 1e-12);
        assert_relative_eq!(img.sample_bilinear(0.25, 0.0), 25.0, epsilon = 1e-12);
    }

    #[test]
    fn bilinear_clamps_at_border_instead_of_panicking() {
        let img = GrayImage::from_vec(2, 2, vec![10, 20, 30, 40]);
        assert_relative_eq!(img.sample_bilinear(-5.0, -5.0), 10.0, epsilon = 1e-12);
        assert_relative_eq!(img.sample_bilinear(50.0, 50.0), 40.0, epsilon = 1e-12);
    }

    #[test]
    fn gradient_of_linear_ramp_is_constant() {
        let mut img = GrayImage::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                img.data_mut()[y * 16 + x] = (x * 8) as u8;
            }
        }
        let g = img.gradient_bilinear(8.0, 8.0);
        assert_relative_eq!(g.x, 8.0, epsilon = 1e-9);
        assert_relative_eq!(g.y, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn rgba_to_luma_uses_integer_weights() {
        // Pure white must round-trip to 255, not 254 — an off-by-one here
        // silently biases every intensity in the pipeline.
        let img = GrayImage::from_rgba(1, 1, &[255, 255, 255, 255]);
        assert_eq!(img.at(0, 0), 255);
        let black = GrayImage::from_rgba(1, 1, &[0, 0, 0, 255]);
        assert_eq!(black.at(0, 0), 0);
        let green = GrayImage::from_rgba(1, 1, &[0, 255, 0, 255]);
        assert_eq!(green.at(0, 0), 149); // (38470 * 255) >> 16 == 149
    }

    #[test]
    fn downsample_halves_and_averages() {
        let img = GrayImage::from_vec(
            4,
            4,
            vec![0, 100, 0, 100, 100, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        );
        let half = img.downsample_half();
        assert_eq!((half.width(), half.height()), (2, 2));
        assert_eq!(half.at(0, 0), 50); // (0+100+100+0+2)/4
        assert_eq!(half.at(0, 1), 0);
    }

    #[test]
    fn downsample_never_produces_zero_dimension() {
        let img = GrayImage::new(1, 1);
        let half = img.downsample_half();
        assert_eq!((half.width(), half.height()), (1, 1));
    }

    #[test]
    fn replay_source_drains_in_order_and_reports_length() {
        let frames: Vec<Frame> = (0..3)
            .map(|i| {
                Frame::new(
                    FrameId(i),
                    Timestamp::from_seconds(i as f64 / 30.0),
                    GrayImage::new(2, 2),
                )
            })
            .collect();
        let mut src = ReplayFrameSource::new(frames, None);
        assert_eq!(src.len_hint(), Some(3));
        assert!(!src.is_live());
        for i in 0..3 {
            assert_eq!(src.next_frame().unwrap().id, FrameId(i));
        }
        assert!(src.next_frame().is_none());
    }
}
