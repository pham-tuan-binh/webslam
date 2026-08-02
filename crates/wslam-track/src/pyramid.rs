//! Image pyramid with per-level intrinsics.
//!
//! Two things make a pyramid correct rather than merely smaller:
//!
//! 1. **The half-pixel convention is shared with the intrinsics.** A 2x area
//!    downsample maps input pixel centre `u` to output `u/2 - 0.25`, which is
//!    exactly what [`CameraIntrinsics::scaled`] encodes. Every point mapping in
//!    this module goes through [`scale_point`], which is the same expression, so
//!    a feature tracked at level 3 unprojects through the level-3 intrinsics
//!    without a quarter-pixel bias creeping in per level.
//! 2. **The decimation filter is anti-aliasing and integer-only.** Decimating a
//!    sharp image without a low-pass moves corners between levels, which is a
//!    coarse-to-fine tracker's worst failure mode: the coarse level converges to
//!    a location the fine level then has to walk away from. Integer arithmetic
//!    keeps the levels bit-identical between native and wasm, which is what makes
//!    spec.md §6 L3 ("any divergence is a port bug") a checkable claim.

use wslam_core::{CameraIntrinsics, GrayImage, Scalar, Vec2};

/// Levels stop being built once halving would take a side below this. A KLT
/// window plus its border has to fit inside the coarsest level or the level
/// contributes nothing but clamped-border artefacts.
pub const MIN_LEVEL_SIZE: u32 = 16;

/// Decimation filter used between pyramid levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PyramidFilter {
    /// 2x2 box average — [`GrayImage::downsample_half`]. Cheapest, and the
    /// filter the GPU pipeline mirrors.
    Box,
    /// Separable binomial `[1 3 3 1]/8`, tapped at `{-1, 0, +1, +2}` around each
    /// even source column.
    ///
    /// The odd offsets matter: the taps are centred on `2x + 0.5`, the same
    /// point the box filter and [`CameraIntrinsics::scaled`] are centred on. The
    /// textbook `[1 4 6 4 1]/16` kernel is centred on `2x` instead and shifts
    /// every level by a quarter pixel relative to the intrinsics.
    #[default]
    Binomial,
}

/// How a pyramid is built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyramidConfig {
    /// Maximum number of levels including level 0. The actual count may be
    /// smaller; see [`MIN_LEVEL_SIZE`].
    pub levels: u32,
    /// Decimation filter.
    pub filter: PyramidFilter,
}

impl Default for PyramidConfig {
    fn default() -> Self {
        PyramidConfig {
            levels: 4,
            filter: PyramidFilter::Binomial,
        }
    }
}

/// One resolution of the pyramid.
#[derive(Debug, Clone)]
pub struct PyramidLevel {
    /// Decimated luma image.
    pub image: GrayImage,
    /// Intrinsics for *this* level, so PnP and triangulation can consume a
    /// coarse-level observation directly.
    pub intrinsics: CameraIntrinsics,
    /// Linear factor from level 0 to this level: `2^-level`.
    pub scale: Scalar,
}

/// A Gaussian/box image pyramid over a [`GrayImage`].
#[derive(Debug, Clone)]
pub struct Pyramid {
    levels: Vec<PyramidLevel>,
}

impl Pyramid {
    /// Build a pyramid from a full-resolution image and its intrinsics.
    ///
    /// Always produces at least one level (level 0 is the input, unfiltered).
    #[must_use]
    pub fn build(image: &GrayImage, intrinsics: &CameraIntrinsics, config: &PyramidConfig) -> Self {
        let mut levels = Vec::with_capacity(config.levels.max(1) as usize);
        levels.push(PyramidLevel {
            image: image.clone(),
            intrinsics: *intrinsics,
            scale: 1.0,
        });

        for l in 1..config.levels.max(1) {
            let prev = levels.last().expect("level 0 pushed above");
            if prev.image.width() / 2 < MIN_LEVEL_SIZE || prev.image.height() / 2 < MIN_LEVEL_SIZE {
                break;
            }
            let image = match config.filter {
                PyramidFilter::Box => prev.image.downsample_half(),
                PyramidFilter::Binomial => downsample_binomial(&prev.image),
            };
            // `scaled` composes exactly in fx/cx, but its width/height round
            // where the decimation truncates: a 641-wide level halves to 320
            // while `scaled(0.5)` would round to 321. The image is the authority.
            let mut intrinsics = prev.intrinsics.scaled(0.5);
            intrinsics.width = image.width();
            intrinsics.height = image.height();
            levels.push(PyramidLevel {
                image,
                intrinsics,
                scale: 0.5_f64.powi(l as i32),
            });
        }

        Pyramid { levels }
    }

    /// Build with the default filter and a level count.
    #[must_use]
    pub fn with_levels(image: &GrayImage, intrinsics: &CameraIntrinsics, levels: u32) -> Self {
        Self::build(
            image,
            intrinsics,
            &PyramidConfig {
                levels,
                ..PyramidConfig::default()
            },
        )
    }

    /// All levels, finest first.
    #[must_use]
    pub fn levels(&self) -> &[PyramidLevel] {
        &self.levels
    }

    /// Number of levels actually built.
    #[must_use]
    pub fn len(&self) -> usize {
        self.levels.len()
    }

    /// Always false — [`Pyramid::build`] guarantees level 0 exists. Present
    /// because clippy asks for it next to `len`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    /// Index of the coarsest level.
    #[must_use]
    pub fn top_level(&self) -> u32 {
        self.levels.len().saturating_sub(1) as u32
    }

    /// Level 0 — the full-resolution image.
    #[must_use]
    pub fn base(&self) -> &PyramidLevel {
        &self.levels[0]
    }

    /// A level by index, or `None` if it was not built.
    #[must_use]
    pub fn level(&self, index: u32) -> Option<&PyramidLevel> {
        self.levels.get(index as usize)
    }
}

/// Rescale a pixel coordinate by `factor`, preserving pixel *centres*.
///
/// The one expression the whole module is built on, and the same one
/// [`CameraIntrinsics::scaled`] applies to the principal point. Using
/// `px * factor` instead is the classic quarter-pixel-per-level drift.
#[inline]
#[must_use]
pub fn scale_point(px: Vec2, factor: Scalar) -> Vec2 {
    Vec2::new((px.x + 0.5) * factor - 0.5, (px.y + 0.5) * factor - 0.5)
}

/// Full-resolution pixel -> level-`level` pixel.
#[inline]
#[must_use]
pub fn to_level(px: Vec2, level: u32) -> Vec2 {
    scale_point(px, 0.5_f64.powi(level as i32))
}

/// Level-`level` pixel -> full-resolution pixel.
#[inline]
#[must_use]
pub fn from_level(px: Vec2, level: u32) -> Vec2 {
    scale_point(px, 2.0_f64.powi(level as i32))
}

/// Half-resolution copy by separable binomial `[1 3 3 1]/8`.
///
/// Integer arithmetic with round-half-up, matching
/// [`GrayImage::downsample_half`], so a pyramid is bit-reproducible across
/// targets. Borders clamp.
#[must_use]
pub fn downsample_binomial(src: &GrayImage) -> GrayImage {
    let nw = (src.width() / 2).max(1);
    let nh = (src.height() / 2).max(1);
    let sh = src.height() as i32;

    // Horizontal pass into u16: max value 255 * 8 = 2040.
    let mut rows = vec![0u16; nw as usize * src.height() as usize];
    for y in 0..sh {
        let base = y as usize * nw as usize;
        for x in 0..nw {
            let sx = (x * 2) as i32;
            let s = src.at(sx - 1, y) as u32
                + 3 * src.at(sx, y) as u32
                + 3 * src.at(sx + 1, y) as u32
                + src.at(sx + 2, y) as u32;
            rows[base + x as usize] = s as u16;
        }
    }
    let row = |y: i32, x: u32| -> u32 {
        let y = y.clamp(0, sh - 1) as usize;
        rows[y * nw as usize + x as usize] as u32
    };

    // Vertical pass: max 2040 * 8 = 16320, so (s + 32) / 64 lands in [0, 255].
    let mut out = GrayImage::new(nw, nh);
    let data = out.data_mut();
    for y in 0..nh {
        let sy = (y * 2) as i32;
        for x in 0..nw {
            let s = row(sy - 1, x) + 3 * row(sy, x) + 3 * row(sy + 1, x) + row(sy + 2, x);
            data[(y * nw + x) as usize] = ((s + 32) / 64) as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Deterministic band-limited texture, evaluated in continuous coordinates
    /// so a translated copy is exact rather than resampled.
    pub(crate) fn texture(x: Scalar, y: Scalar) -> Scalar {
        128.0
            + 55.0 * (0.031 * x + 0.019 * y).sin()
            + 38.0 * (0.113 * x - 0.071 * y).cos()
            + 22.0 * (0.211 * x).sin() * (0.173 * y).cos()
    }

    pub(crate) fn render(
        width: u32,
        height: u32,
        f: impl Fn(Scalar, Scalar) -> Scalar,
    ) -> GrayImage {
        let mut img = GrayImage::new(width, height);
        let data = img.data_mut();
        for y in 0..height {
            for x in 0..width {
                data[(y * width + x) as usize] =
                    f(x as Scalar, y as Scalar).round().clamp(0.0, 255.0) as u8;
            }
        }
        img
    }

    fn k() -> CameraIntrinsics {
        CameraIntrinsics::from_focal(600.0, 640, 480)
    }

    #[test]
    fn level_dimensions_halve() {
        let img = render(640, 480, texture);
        let p = Pyramid::with_levels(&img, &k(), 4);
        assert_eq!(p.len(), 4);
        let dims: Vec<(u32, u32)> = p
            .levels()
            .iter()
            .map(|l| (l.image.width(), l.image.height()))
            .collect();
        assert_eq!(dims, vec![(640, 480), (320, 240), (160, 120), (80, 60)]);
    }

    #[test]
    fn intrinsics_scale_with_the_level() {
        let img = render(640, 480, texture);
        let p = Pyramid::with_levels(&img, &k(), 4);
        for (l, level) in p.levels().iter().enumerate() {
            let f = 0.5_f64.powi(l as i32);
            assert_relative_eq!(level.scale, f, epsilon = 1e-15);
            assert_relative_eq!(level.intrinsics.fx, 600.0 * f, epsilon = 1e-12);
            // Half-pixel-correct principal point, chained from level 0.
            assert_relative_eq!(
                level.intrinsics.cx,
                (320.0 + 0.5) * f - 0.5,
                epsilon = 1e-12
            );
            assert_eq!(level.intrinsics.width, level.image.width());
            assert_eq!(level.intrinsics.height, level.image.height());
        }
    }

    #[test]
    fn level_intrinsics_agree_with_point_mapping() {
        // The whole point of the shared convention: projecting a 3D point with
        // the level-l intrinsics must equal projecting at level 0 and then
        // mapping the pixel down. A `px * 0.5` mapping fails this.
        let img = render(640, 480, texture);
        let p = Pyramid::with_levels(&img, &k(), 4);
        let point = wslam_core::Vec3::new(0.37, -0.21, 2.4);
        let full = k().project(&point).unwrap();
        for (l, level) in p.levels().iter().enumerate() {
            let direct = level.intrinsics.project(&point).unwrap();
            assert_relative_eq!(direct, to_level(full, l as u32), epsilon = 1e-9);
        }
    }

    #[test]
    fn point_mapping_roundtrips_across_levels() {
        for level in 0..5u32 {
            let px = Vec2::new(317.25, 91.75);
            assert_relative_eq!(from_level(to_level(px, level), level), px, epsilon = 1e-9);
        }
    }

    #[test]
    fn levels_stop_before_the_image_gets_useless() {
        let img = render(64, 48, texture);
        let p = Pyramid::with_levels(&img, &CameraIntrinsics::from_focal(60.0, 64, 48), 8);
        // 64x48 -> 32x24 -> would be 16x12, and 12 < MIN_LEVEL_SIZE.
        assert_eq!(p.len(), 2);
        assert!(!p.is_empty());
        assert_eq!(p.top_level(), 1);
        assert!(p.level(5).is_none());
    }

    #[test]
    fn binomial_filter_preserves_a_constant() {
        // Rounding that drifts on a flat field biases every intensity in the
        // pyramid, and a low-light threshold read off level 2 would be wrong.
        for v in [0u8, 1, 127, 128, 254, 255] {
            let img = GrayImage::from_vec(32, 32, vec![v; 1024]);
            let half = downsample_binomial(&img);
            assert!(
                half.data().iter().all(|&p| p == v),
                "constant {v} did not survive decimation"
            );
        }
    }

    #[test]
    fn binomial_filter_attenuates_nyquist_where_box_does_not() {
        // A 1-px checkerboard is pure Nyquist. The box filter averages it to a
        // flat field (fine), but a 2-px vertical stripe pattern aliases through
        // a box filter and must not through the binomial one.
        let stripes = render(
            64,
            64,
            |x, _| if (x as i32 / 2) % 2 == 0 { 0.0 } else { 255.0 },
        );
        let boxed = stripes.downsample_half();
        let binom = downsample_binomial(&stripes);
        let spread = |i: &GrayImage| {
            let row: Vec<i32> = (16..48).map(|x| i.at(x, 16) as i32).collect();
            row.iter().max().unwrap() - row.iter().min().unwrap()
        };
        // Box turns 2-px stripes into 1-px stripes at full contrast (aliasing).
        assert!(spread(&boxed) > 200, "box spread {}", spread(&boxed));
        assert!(spread(&binom) < 160, "binomial spread {}", spread(&binom));
    }

    #[test]
    fn binomial_taps_are_centred_like_the_box_filter() {
        // A linear horizontal ramp is the sharpest test of filter centring: any
        // shift in the kernel centre of mass shows up as a constant offset.
        let ramp = render(64, 64, |x, _| x * 3.0);
        let boxed = ramp.downsample_half();
        let binom = downsample_binomial(&ramp);
        for x in 4..28 {
            assert!(
                (boxed.at(x, 16) as i32 - binom.at(x, 16) as i32).abs() <= 1,
                "filters disagree at x={x}: box {} binomial {}",
                boxed.at(x, 16),
                binom.at(x, 16)
            );
        }
    }

    #[test]
    fn box_filter_selected_by_config() {
        let img = render(64, 64, texture);
        let k = CameraIntrinsics::from_focal(60.0, 64, 64);
        let boxed = Pyramid::build(
            &img,
            &k,
            &PyramidConfig {
                levels: 2,
                filter: PyramidFilter::Box,
            },
        );
        assert_eq!(boxed.level(1).unwrap().image, img.downsample_half());
    }

    #[test]
    fn single_level_pyramid_is_the_input() {
        let img = render(32, 32, texture);
        let p = Pyramid::with_levels(&img, &CameraIntrinsics::from_focal(30.0, 32, 32), 1);
        assert_eq!(p.len(), 1);
        assert_eq!(p.base().image, img);
        assert_relative_eq!(p.base().scale, 1.0);
    }
}
