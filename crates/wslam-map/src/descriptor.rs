//! Binary descriptors: oriented FAST keypoints and steered BRIEF.
//!
//! spec.md §4 L4a asks for *"sparse keyframes with binary descriptors plus a
//! bag-of-words database"*, and §7 is explicit that the C++ prior art is
//! reimplemented rather than forked. This is ORB in the sense of Rublee et al.
//! (ICCV 2011): FAST-9 corners (Rosten & Drummond), an intensity-centroid
//! orientation, and a BRIEF pattern (Calonder et al., ECCV 2010) steered by
//! that orientation.
//!
//! ## The sampling pattern is data, and it must never move
//!
//! A descriptor is only comparable to another built from the *same* 256
//! sampling pairs. A map serialised in one session is matched against features
//! extracted in a later one, possibly on a different device, so the pattern has
//! to be bit-identical across runs, builds and architectures — otherwise two
//! sessions occupy two different descriptor spaces and relocalization silently
//! never fires.
//!
//! The pattern is therefore generated once from [`PATTERN_SEED`] through
//! [`DeterministicRng`] using **integer arithmetic only**. No transcendental
//! function touches it, because a one-ULP `libm` difference between platforms
//! is enough to move a sample point by a pixel.
//!
//! Steering is the one place a `cos`/`sin` is unavoidable. It is confined to a
//! 64-entry Q12 fixed-point table computed once; every per-keypoint rotation is
//! then integer arithmetic on that table, and the table itself is pinned by
//! `rotation_table_is_pinned` so a platform that disagrees fails CI loudly
//! instead of shipping an incompatible descriptor space.

use std::sync::OnceLock;
use wslam_core::{DeterministicRng, GrayImage, Vec2};

/// Bytes in a descriptor.
pub const DESCRIPTOR_BYTES: usize = 32;

/// Bits in a descriptor — one per BRIEF intensity comparison.
pub const DESCRIPTOR_BITS: u32 = 256;

/// Radius of the descriptor patch, in pixels. 15 gives ORB's 31x31 patch.
pub const PATCH_RADIUS: i32 = 15;

/// Seed for the BRIEF sampling pattern. **Changing this invalidates every map
/// ever serialised**, so it is a format-breaking change and belongs with a bump
/// of [`wslam_core::MAP_FORMAT_VERSION`].
pub const PATTERN_SEED: u64 = 0x5742_5249_4546_0001;

/// Half-width of the box filter applied before sampling. Calonder et al. show
/// BRIEF is unusable on raw pixels; the smoothing is part of the descriptor,
/// not a preprocessing nicety.
pub const SMOOTHING_RADIUS: i32 = 2;

/// Quantisation of the keypoint orientation used when steering the pattern.
///
/// ORB uses 30 bins (12 degrees). 64 halves the quantisation error to 2.8
/// degrees — 0.75 px at the patch rim — for a table that is still trivially
/// small, and a power of two keeps the binning a shift rather than a modulo.
pub const ANGLE_BINS: usize = 64;

/// Fractional bits in the fixed-point rotation table.
const FIXED_SHIFT: i32 = 12;
const FIXED_ONE: i32 = 1 << FIXED_SHIFT;

/// A 256-bit binary descriptor, ORB-style.
///
/// Bit `i` lives in `0.i / 8` at position `i % 8`, least-significant first. The
/// layout is part of the serialised map format, so it is fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BinaryDescriptor(pub [u8; DESCRIPTOR_BYTES]);

impl Default for BinaryDescriptor {
    fn default() -> Self {
        BinaryDescriptor::ZERO
    }
}

impl BinaryDescriptor {
    /// The all-zero descriptor. Used as a tree-node placeholder, never as a
    /// real observation.
    pub const ZERO: Self = BinaryDescriptor([0u8; DESCRIPTOR_BYTES]);

    /// Hamming distance, in bits. Ranges over `0..=256`.
    ///
    /// Four 64-bit popcounts rather than 32 byte lookups: `count_ones` is a
    /// single instruction on every target we ship to, including wasm
    /// (`i64.popcnt`).
    #[must_use]
    pub fn hamming(&self, other: &Self) -> u32 {
        let mut d = 0u32;
        for (a, b) in self.0.chunks_exact(8).zip(other.0.chunks_exact(8)) {
            let a = u64::from_le_bytes(a.try_into().expect("chunks_exact(8)"));
            let b = u64::from_le_bytes(b.try_into().expect("chunks_exact(8)"));
            d += (a ^ b).count_ones();
        }
        d
    }

    /// Number of set bits.
    #[must_use]
    pub fn popcount(&self) -> u32 {
        self.0.iter().map(|b| b.count_ones()).sum()
    }

    /// Read bit `i`. Out-of-range indices read `false`.
    #[inline]
    #[must_use]
    pub fn bit(&self, i: usize) -> bool {
        i < DESCRIPTOR_BITS as usize && (self.0[i >> 3] >> (i & 7)) & 1 == 1
    }

    /// Set bit `i` to `value`. Out-of-range indices are ignored.
    #[inline]
    pub fn set_bit(&mut self, i: usize, value: bool) {
        if i >= DESCRIPTOR_BITS as usize {
            return;
        }
        let mask = 1u8 << (i & 7);
        if value {
            self.0[i >> 3] |= mask;
        } else {
            self.0[i >> 3] &= !mask;
        }
    }

    /// Bitwise-majority "median" of a set of descriptors.
    ///
    /// This is the centroid operator in Hamming space, and the reason
    /// [`crate::Vocabulary`] does k-**medians** rather than k-means: the
    /// arithmetic mean of binary vectors is not a binary vector, while the
    /// per-bit majority minimises the summed Hamming distance exactly.
    ///
    /// Ties (exactly half the members set) resolve to zero, deterministically.
    /// Returns [`BinaryDescriptor::ZERO`] for an empty set.
    #[must_use]
    pub fn majority(members: &[BinaryDescriptor]) -> BinaryDescriptor {
        let n = members.len();
        if n == 0 {
            return BinaryDescriptor::ZERO;
        }
        let mut out = BinaryDescriptor::ZERO;
        for bit in 0..DESCRIPTOR_BITS as usize {
            let ones = members.iter().filter(|d| d.bit(bit)).count();
            out.set_bit(bit, ones * 2 > n);
        }
        out
    }
}

/// One BRIEF intensity-comparison pair, as integer offsets inside the patch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BriefPair {
    /// First sample, x offset.
    pub ax: i32,
    /// First sample, y offset.
    pub ay: i32,
    /// Second sample, x offset.
    pub bx: i32,
    /// Second sample, y offset.
    pub by: i32,
}

static PATTERN: OnceLock<[BriefPair; DESCRIPTOR_BITS as usize]> = OnceLock::new();

/// The 256 BRIEF sampling pairs, built once from [`PATTERN_SEED`].
///
/// See the module docs: this table is part of the on-disk contract.
#[must_use]
pub fn sampling_pattern() -> &'static [BriefPair; DESCRIPTOR_BITS as usize] {
    PATTERN.get_or_init(build_pattern)
}

fn build_pattern() -> [BriefPair; DESCRIPTOR_BITS as usize] {
    let mut rng = DeterministicRng::new("brief-pattern", PATTERN_SEED);
    let mut out = [BriefPair {
        ax: 0,
        ay: 0,
        bx: 0,
        by: 0,
    }; DESCRIPTOR_BITS as usize];
    let mut i = 0;
    while i < out.len() {
        let (ax, ay) = sample_offset(&mut rng);
        let (bx, by) = sample_offset(&mut rng);
        // A pair that samples the same pixel twice always yields the same bit
        // and carries no information.
        if (ax, ay) == (bx, by) {
            continue;
        }
        out[i] = BriefPair { ax, ay, bx, by };
        i += 1;
    }
    out
}

/// A point drawn from an integer approximation of Calonder's G-II isotropic
/// Gaussian, rejected back into the patch disc.
///
/// The mean of three uniforms on `[-R, R]` has standard deviation `R/3` — 5 px
/// at `R = 15`, close to ORB's `patch/5`. Integer throughout, so the pattern is
/// reproducible bit for bit on every target.
fn sample_offset(rng: &mut DeterministicRng) -> (i32, i32) {
    loop {
        let x = irwin_hall(rng);
        let y = irwin_hall(rng);
        if x * x + y * y <= PATCH_RADIUS * PATCH_RADIUS {
            return (x, y);
        }
    }
}

fn irwin_hall(rng: &mut DeterministicRng) -> i32 {
    let span = (2 * PATCH_RADIUS + 1) as usize;
    let mut s = 0i32;
    for _ in 0..3 {
        s += rng.below(span) as i32 - PATCH_RADIUS;
    }
    s / 3
}

static ROT_TABLE: OnceLock<[(i32, i32); ANGLE_BINS]> = OnceLock::new();

/// `(cos, sin)` per orientation bin, in Q12 fixed point.
///
/// Pinned by a test: this is the only place a transcendental function reaches
/// the descriptor, and a platform whose `cos` rounds differently must fail
/// loudly rather than produce descriptors nothing else can match.
#[must_use]
pub fn rotation_table() -> &'static [(i32, i32); ANGLE_BINS] {
    ROT_TABLE.get_or_init(|| {
        let mut t = [(0i32, 0i32); ANGLE_BINS];
        for (bin, entry) in t.iter_mut().enumerate() {
            let theta = std::f64::consts::TAU * bin as f64 / ANGLE_BINS as f64;
            *entry = (
                (theta.cos() * FIXED_ONE as f64).round() as i32,
                (theta.sin() * FIXED_ONE as f64).round() as i32,
            );
        }
        t
    })
}

/// Quantise an orientation in radians to a table bin.
#[must_use]
pub fn angle_bin(theta: f64) -> usize {
    if !theta.is_finite() {
        return 0;
    }
    let turns = theta / std::f64::consts::TAU;
    let b = (turns * ANGLE_BINS as f64).round() as i64;
    b.rem_euclid(ANGLE_BINS as i64) as usize
}

/// Integer rotation of a patch offset by a quantised orientation, Q12 with
/// round-to-nearest.
#[inline]
fn steer(cos_q: i32, sin_q: i32, x: i32, y: i32) -> (i32, i32) {
    let half = FIXED_ONE / 2;
    (
        (cos_q * x - sin_q * y + half) >> FIXED_SHIFT,
        (sin_q * x + cos_q * y + half) >> FIXED_SHIFT,
    )
}

/// Summed-area table over an image, for the pre-smoothing box filter.
///
/// `u32` accumulator: the largest possible sum is `255 * w * h`, so this is
/// exact up to about 16.8 megapixels — an order of magnitude above anything a
/// browser hands us.
struct Integral {
    w: i32,
    h: i32,
    sum: Vec<u32>,
}

impl Integral {
    fn new(image: &GrayImage) -> Self {
        let w = image.width() as i32;
        let h = image.height() as i32;
        let stride = (w + 1) as usize;
        let mut sum = vec![0u32; stride * (h + 1) as usize];
        let data = image.data();
        for y in 0..h as usize {
            let mut row = 0u32;
            for x in 0..w as usize {
                row += data[y * w as usize + x] as u32;
                sum[(y + 1) * stride + x + 1] = sum[y * stride + x + 1] + row;
            }
        }
        Integral { w, h, sum }
    }

    /// `(sum, area)` of the box of half-width `r` around `(cx, cy)`, clamped to
    /// the image. Returning the area rather than a mean keeps the comparison in
    /// [`describe`] exact at the border, where the two boxes differ in size.
    #[inline]
    fn box_sum(&self, cx: i32, cy: i32, r: i32) -> (i64, i64) {
        if self.w == 0 || self.h == 0 {
            return (0, 1);
        }
        let x0 = (cx - r).clamp(0, self.w - 1);
        let x1 = (cx + r).clamp(0, self.w - 1);
        let y0 = (cy - r).clamp(0, self.h - 1);
        let y1 = (cy + r).clamp(0, self.h - 1);
        let stride = (self.w + 1) as usize;
        let idx = |x: i32, y: i32| (y as usize) * stride + (x as usize);
        let s = self.sum[idx(x1 + 1, y1 + 1)] as i64
            - self.sum[idx(x0, y1 + 1)] as i64
            - self.sum[idx(x1 + 1, y0)] as i64
            + self.sum[idx(x0, y0)] as i64;
        (s, ((x1 - x0 + 1) as i64) * ((y1 - y0 + 1) as i64))
    }
}

/// Compute steered BRIEF descriptors at the given keypoints.
///
/// `orientations[i]` is the orientation of `keypoints[i]` in radians, as
/// produced by [`fast_keypoints`] or [`intensity_centroid_orientation`]. A
/// short `orientations` slice is treated as zero orientation for the missing
/// entries, which is the upright-BRIEF behaviour.
///
/// Keypoints near the border are **not** dropped: the patch clamps at the image
/// edge instead. Dropping them would break the index correspondence between
/// `keypoints`, `descriptors` and `landmarks` that [`crate::Keyframe`] relies
/// on, and a clamped patch is merely a weak descriptor rather than a wrong one.
#[must_use]
pub fn describe(
    image: &GrayImage,
    keypoints: &[Vec2],
    orientations: &[f64],
) -> Vec<BinaryDescriptor> {
    if keypoints.is_empty() {
        return Vec::new();
    }
    let integral = Integral::new(image);
    let pattern = sampling_pattern();
    let table = rotation_table();

    keypoints
        .iter()
        .enumerate()
        .map(|(i, kp)| {
            let theta = orientations.get(i).copied().unwrap_or(0.0);
            let (cos_q, sin_q) = table[angle_bin(theta)];
            let x0 = kp.x.round() as i32;
            let y0 = kp.y.round() as i32;
            let mut d = BinaryDescriptor::ZERO;
            for (bit, p) in pattern.iter().enumerate() {
                let (ax, ay) = steer(cos_q, sin_q, p.ax, p.ay);
                let (bx, by) = steer(cos_q, sin_q, p.bx, p.by);
                let (sa, area_a) = integral.box_sum(x0 + ax, y0 + ay, SMOOTHING_RADIUS);
                let (sb, area_b) = integral.box_sum(x0 + bx, y0 + by, SMOOTHING_RADIUS);
                // Cross-multiply instead of dividing: exact, and it stays
                // correct where the two clamped boxes have different areas.
                d.set_bit(bit, sa * area_b < sb * area_a);
            }
            d
        })
        .collect()
}

/// Bresenham circle of radius 3 — the 16 FAST test pixels, in order.
const FAST_CIRCLE: [(i32, i32); 16] = [
    (0, -3),
    (1, -3),
    (2, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 3),
    (-1, 3),
    (-2, 2),
    (-3, 1),
    (-3, 0),
    (-3, -1),
    (-2, -2),
    (-1, -3),
];

/// Minimum contiguous arc length for FAST-9.
const FAST_ARC: usize = 9;

/// Longest circular run of `true` in a 16-element predicate.
fn longest_circular_run(flags: [bool; 16]) -> usize {
    if flags.iter().all(|&f| f) {
        return 16;
    }
    let mut best = 0;
    let mut run = 0;
    // Two laps: a run that wraps the seam is found on the second.
    for i in 0..32 {
        if flags[i & 15] {
            run += 1;
            best = best.max(run);
        } else {
            run = 0;
        }
    }
    best.min(16)
}

/// FAST-9 corner response at an integer pixel, or 0 if it is not a corner.
///
/// The score is the standard "sum of absolute deviations beyond the threshold"
/// variant: cheap, monotone in corner strength, and adequate for ranking.
fn fast_response(image: &GrayImage, x: i32, y: i32, threshold: i32) -> i32 {
    let p = image.at(x, y) as i32;
    let mut vals = [0i32; 16];
    for (v, off) in vals.iter_mut().zip(FAST_CIRCLE.iter()) {
        *v = image.at(x + off.0, y + off.1) as i32;
    }

    // High-speed reject on the four compass points. A contiguous window of 9
    // out of 16 contains 3 compass indices when it starts on one and 2
    // otherwise, so the *guaranteed* coverage is 2 — see
    // `high_speed_reject_is_the_fast9_bound_not_the_fast12_one`. Requiring 3
    // here is the FAST-**12** rule (a window of 12 always covers 3) and it
    // silently discards three quarters of all real FAST-9 corners, including
    // every axis-aligned step corner.
    const MIN_COMPASS: i32 = 2;
    let mut bright = 0;
    let mut dark = 0;
    for &i in &[0usize, 4, 8, 12] {
        if vals[i] > p + threshold {
            bright += 1;
        } else if vals[i] < p - threshold {
            dark += 1;
        }
    }
    if bright < MIN_COMPASS && dark < MIN_COMPASS {
        return 0;
    }

    let mut bright_flags = [false; 16];
    let mut dark_flags = [false; 16];
    for (i, &v) in vals.iter().enumerate() {
        bright_flags[i] = v > p + threshold;
        dark_flags[i] = v < p - threshold;
    }
    let is_bright = longest_circular_run(bright_flags) >= FAST_ARC;
    let is_dark = longest_circular_run(dark_flags) >= FAST_ARC;
    if !is_bright && !is_dark {
        return 0;
    }

    let sb: i32 = vals.iter().map(|&v| (v - p - threshold).max(0)).sum();
    let sd: i32 = vals.iter().map(|&v| (p - v - threshold).max(0)).sum();
    sb.max(sd)
}

/// Intensity-centroid orientation of the patch at `(x, y)`, in radians.
///
/// Rosin's moment orientation, as used by ORB: `theta = atan2(m01, m10)` over a
/// disc of radius [`PATCH_RADIUS`]. Returns 0 for a rotationally symmetric
/// patch, where the orientation genuinely is undefined.
#[must_use]
pub fn intensity_centroid_orientation(image: &GrayImage, x: i32, y: i32) -> f64 {
    let r2 = PATCH_RADIUS * PATCH_RADIUS;
    let mut m10 = 0i64;
    let mut m01 = 0i64;
    for dy in -PATCH_RADIUS..=PATCH_RADIUS {
        for dx in -PATCH_RADIUS..=PATCH_RADIUS {
            if dx * dx + dy * dy > r2 {
                continue;
            }
            let i = image.at(x + dx, y + dy) as i64;
            m10 += dx as i64 * i;
            m01 += dy as i64 * i;
        }
    }
    if m10 == 0 && m01 == 0 {
        return 0.0;
    }
    (m01 as f64).atan2(m10 as f64)
}

/// Detect oriented FAST keypoints.
///
/// Returns up to `max` keypoints as `(pixel, orientation_radians)`, strongest
/// first. `threshold` is the FAST intensity threshold; 20 is a reasonable
/// starting point for a phone camera.
///
/// Non-maximum suppression is 3x3 on the response map, and ties break on
/// `(y, x)` so the output is a deterministic function of the image alone.
#[must_use]
pub fn fast_keypoints(image: &GrayImage, threshold: u8, max: usize) -> Vec<(Vec2, f64)> {
    let w = image.width() as i32;
    let h = image.height() as i32;
    if w < 7 || h < 7 || max == 0 {
        return Vec::new();
    }
    let t = threshold as i32;
    let mut response = vec![0i32; (w * h) as usize];
    for y in 3..h - 3 {
        for x in 3..w - 3 {
            response[(y * w + x) as usize] = fast_response(image, x, y, t);
        }
    }

    let mut peaks: Vec<(i32, i32, i32)> = Vec::new(); // (score, y, x)
    for y in 3..h - 3 {
        for x in 3..w - 3 {
            let s = response[(y * w + x) as usize];
            if s == 0 {
                continue;
            }
            let mut is_peak = true;
            'nms: for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let n = response[((y + dy) * w + x + dx) as usize];
                    // Strictly-greater on one side and greater-or-equal on the
                    // other would keep both members of a plateau; break the tie
                    // by scan order so exactly one survives.
                    let earlier = (dy, dx) < (0, 0);
                    if n > s || (n == s && earlier) {
                        is_peak = false;
                        break 'nms;
                    }
                }
            }
            if is_peak {
                peaks.push((s, y, x));
            }
        }
    }

    peaks.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    peaks.truncate(max);
    peaks
        .into_iter()
        .map(|(_, y, x)| {
            (
                Vec2::new(x as f64, y as f64),
                intensity_centroid_orientation(image, x, y),
            )
        })
        .collect()
}

/// Brute-force nearest neighbour in Hamming space.
///
/// Returns `(index, distance)` of the best match and the second-best distance,
/// which is what a Lowe ratio test needs. `None` when `haystack` is empty.
#[must_use]
pub fn nearest_two(
    needle: &BinaryDescriptor,
    haystack: &[BinaryDescriptor],
) -> Option<(usize, u32, u32)> {
    let mut best = (usize::MAX, u32::MAX);
    let mut second = u32::MAX;
    for (i, d) in haystack.iter().enumerate() {
        let dist = needle.hamming(d);
        if dist < best.1 {
            second = best.1;
            best = (i, dist);
        } else if dist < second {
            second = dist;
        }
    }
    (best.0 != usize::MAX).then_some((best.0, best.1, second))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// A smooth, aperiodic texture. Smooth so that rotating it by bilinear
    /// resampling is faithful; aperiodic so that distant patches are genuinely
    /// unrelated.
    fn texture(w: u32, h: u32, seed: f64) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let fx = x as f64 * 0.11 + seed;
                let fy = y as f64 * 0.13 + seed * 2.0;
                let v = 128.0
                    + 50.0 * (fx).sin() * (fy * 0.8).cos()
                    + 40.0 * (fx * 0.37 + fy * 0.21).sin()
                    + 25.0 * (fy * 0.53 - fx * 0.19).cos();
                img.data_mut()[(y * w + x) as usize] = v.clamp(0.0, 255.0) as u8;
            }
        }
        img
    }

    /// Dark discs of varying depth on a light field, at pseudo-random
    /// positions: an image that actually contains FAST corners, with genuinely
    /// different response strengths so "strongest first" means something.
    ///
    /// `texture` above deliberately cannot serve here — see
    /// `a_band_limited_texture_has_no_fast_corners_by_construction`.
    fn dots(w: u32, h: u32, seed: u64) -> GrayImage {
        let mut rng = DeterministicRng::new("dots", seed);
        let mut img = GrayImage::from_vec(w, h, vec![210u8; (w * h) as usize]);
        for _ in 0..(w * h / 220).max(1) {
            let cx = 6 + rng.below(w as usize - 12) as i32;
            let cy = 6 + rng.below(h as usize - 12) as i32;
            let level = rng.below(120) as u8;
            for y in cy - 2..=cy + 2 {
                for x in cx - 2..=cx + 2 {
                    if (x - cx).pow(2) + (y - cy).pow(2) <= 4 {
                        img.data_mut()[(y * w as i32 + x) as usize] = level;
                    }
                }
            }
        }
        img
    }

    /// Rotate `src` about its centre by `theta`, bilinearly.
    fn rotate(src: &GrayImage, theta: f64) -> GrayImage {
        let w = src.width();
        let h = src.height();
        let cx = (w as f64 - 1.0) * 0.5;
        let cy = (h as f64 - 1.0) * 0.5;
        let (c, s) = (theta.cos(), theta.sin());
        let mut out = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                // Inverse map: sample the source at the un-rotated location.
                let sx = cx + c * dx + s * dy;
                let sy = cy - s * dx + c * dy;
                out.data_mut()[(y * w + x) as usize] =
                    src.sample_bilinear(sx, sy).clamp(0.0, 255.0) as u8;
            }
        }
        out
    }

    #[test]
    fn hamming_matches_hand_computed_values() {
        let mut a = BinaryDescriptor::ZERO;
        let mut b = BinaryDescriptor::ZERO;
        assert_eq!(a.hamming(&b), 0);

        // 0b1011_0001 vs 0b0000_0000 -> 4 bits.
        a.0[0] = 0b1011_0001;
        assert_eq!(a.hamming(&b), 4);
        assert_eq!(a.popcount(), 4);

        // Differ in byte 0 (0b1011_0001 ^ 0b0000_1111 = 0b1011_1110 -> 6) and
        // in byte 31 (0xFF ^ 0x00 -> 8). Total 14.
        b.0[0] = 0b0000_1111;
        b.0[31] = 0xFF;
        assert_eq!(a.hamming(&b), 6 + 8);
        assert_eq!(b.hamming(&a), 14, "hamming must be symmetric");

        // Complement is maximally distant.
        let all_ones = BinaryDescriptor([0xFF; DESCRIPTOR_BYTES]);
        assert_eq!(BinaryDescriptor::ZERO.hamming(&all_ones), DESCRIPTOR_BITS);
    }

    #[test]
    fn bit_accessors_agree_with_byte_layout() {
        let mut d = BinaryDescriptor::ZERO;
        d.set_bit(0, true);
        assert_eq!(d.0[0], 1);
        d.set_bit(9, true);
        assert_eq!(d.0[1], 0b10);
        assert!(d.bit(0) && d.bit(9) && !d.bit(1));
        d.set_bit(0, false);
        assert!(!d.bit(0));
        // Out of range is a no-op, not a panic.
        d.set_bit(999, true);
        assert!(!d.bit(999));
    }

    #[test]
    fn majority_is_the_hamming_centroid() {
        let a = BinaryDescriptor([0b0000_0111; DESCRIPTOR_BYTES]);
        let b = BinaryDescriptor([0b0000_0011; DESCRIPTOR_BYTES]);
        let c = BinaryDescriptor([0b0000_0001; DESCRIPTOR_BYTES]);
        // bit0: 3/3, bit1: 2/3, bit2: 1/3 -> 0b011
        assert_eq!(BinaryDescriptor::majority(&[a, b, c]).0[0], 0b0000_0011);
        // Exact ties resolve to zero.
        assert_eq!(BinaryDescriptor::majority(&[a, c]).0[0], 0b0000_0001);
        assert_eq!(BinaryDescriptor::majority(&[]), BinaryDescriptor::ZERO);
    }

    #[test]
    fn sampling_pattern_is_inside_the_patch_and_never_degenerate() {
        let p = sampling_pattern();
        assert_eq!(p.len(), DESCRIPTOR_BITS as usize);
        let r2 = PATCH_RADIUS * PATCH_RADIUS;
        for pair in p.iter() {
            assert!(pair.ax * pair.ax + pair.ay * pair.ay <= r2);
            assert!(pair.bx * pair.bx + pair.by * pair.by <= r2);
            assert_ne!((pair.ax, pair.ay), (pair.bx, pair.by));
        }
    }

    #[test]
    fn sampling_pattern_is_stable_across_calls() {
        // Two calls hit the same OnceLock, so this pins the *contents* by
        // rebuilding independently.
        let a = build_pattern();
        let b = build_pattern();
        assert_eq!(a.as_slice(), b.as_slice());
        assert_eq!(a.as_slice(), sampling_pattern().as_slice());
    }

    #[test]
    fn rotation_table_is_pinned() {
        // These four are exact by construction; if a platform's cos/sin round
        // differently anywhere the whole descriptor space shifts, so pin the
        // quadrants and a checksum of the rest.
        let t = rotation_table();
        assert_eq!(t[0], (FIXED_ONE, 0));
        assert_eq!(t[ANGLE_BINS / 4], (0, FIXED_ONE));
        assert_eq!(t[ANGLE_BINS / 2], (-FIXED_ONE, 0));
        assert_eq!(t[3 * ANGLE_BINS / 4], (0, -FIXED_ONE));
        // 45 degrees: cos = sin = sqrt(2)/2 -> round(0.70710678 * 4096) = 2896.
        assert_eq!(t[ANGLE_BINS / 8], (2896, 2896));
        for k in 0..ANGLE_BINS {
            // Antipodal bins must be exact negatives, and every entry must lie
            // on the unit circle to within one Q12 ULP.
            let (c, s) = t[k];
            let (c2, s2) = t[(k + ANGLE_BINS / 2) % ANGLE_BINS];
            assert_eq!((c2, s2), (-c, -s), "bin {k}");
            let norm = (c as f64).hypot(s as f64);
            assert!(
                (norm - FIXED_ONE as f64).abs() < 1.0,
                "bin {k}: |v| = {norm}"
            );
        }
    }

    #[test]
    fn steering_is_integer_exact_at_the_quadrants() {
        let t = rotation_table();
        // 90 degrees maps (x, y) -> (-y, x).
        let (c, s) = t[ANGLE_BINS / 4];
        assert_eq!(steer(c, s, 7, -3), (3, 7));
        // Identity bin is a no-op.
        let (c0, s0) = t[0];
        assert_eq!(steer(c0, s0, 7, -3), (7, -3));
    }

    #[test]
    fn angle_bin_wraps_and_survives_non_finite_input() {
        assert_eq!(angle_bin(0.0), 0);
        assert_eq!(angle_bin(std::f64::consts::TAU), 0);
        assert_eq!(angle_bin(-std::f64::consts::TAU), 0);
        assert_eq!(angle_bin(PI), ANGLE_BINS / 2);
        assert_eq!(angle_bin(-PI / 2.0), 3 * ANGLE_BINS / 4);
        assert_eq!(angle_bin(f64::NAN), 0);
        assert_eq!(angle_bin(f64::INFINITY), 0);
    }

    #[test]
    fn fast_finds_a_synthetic_corner_at_the_right_pixel() {
        // A dark quadrant on a bright field: the corner is at (10, 10).
        let mut img = GrayImage::new(21, 21);
        for y in 0..21 {
            for x in 0..21 {
                let v = if x >= 10 && y >= 10 { 20u8 } else { 220u8 };
                img.data_mut()[y * 21 + x] = v;
            }
        }
        let kps = fast_keypoints(&img, 30, 16);
        assert!(!kps.is_empty(), "a step corner must be detected");
        let best = kps[0].0;
        assert!(
            (best.x - 10.0).abs() <= 1.0 && (best.y - 10.0).abs() <= 1.0,
            "strongest response at {best:?}, expected near (10, 10)"
        );
    }

    #[test]
    fn fast_returns_nothing_on_a_uniform_image() {
        // Degenerate case: no texture at all. spec.md §3 names
        // `insufficient-features` as a first-class tracking state, so the
        // detector must report emptiness rather than invent corners.
        let img = GrayImage::new(64, 64);
        assert!(fast_keypoints(&img, 20, 500).is_empty());
        let flat = GrayImage::from_vec(32, 32, vec![137u8; 32 * 32]);
        assert!(fast_keypoints(&flat, 1, 500).is_empty());
    }

    #[test]
    fn fast_handles_images_smaller_than_the_test_circle() {
        for n in 1u32..7 {
            assert!(fast_keypoints(&GrayImage::new(n, n), 20, 10).is_empty());
        }
        assert!(fast_keypoints(&texture(64, 64, 0.0), 20, 0).is_empty());
    }

    #[test]
    fn fast_respects_the_max_and_returns_strongest_first() {
        // `dots`, not `texture`: see the test below for why the smooth texture
        // has no corners to rank at any threshold a real detector would use.
        let img = dots(96, 96, 5);
        let all = fast_keypoints(&img, 12, 10_000);
        let few = fast_keypoints(&img, 12, 5);
        assert!(all.len() > 5, "texture should yield plenty of corners");
        assert_eq!(few.len(), 5);
        // The truncated list is the prefix of the full one.
        for (a, b) in few.iter().zip(all.iter()) {
            assert_eq!(a.0, b.0);
        }
        // ... and the full list really is ordered by strength, so the prefix
        // above is the *strongest* five and not merely the first five found.
        let scores: Vec<i32> = all
            .iter()
            .map(|(p, _)| fast_response(&img, p.x as i32, p.y as i32, 12))
            .collect();
        assert!(
            scores.windows(2).all(|w| w[0] >= w[1]),
            "not sorted: {scores:?}"
        );
        assert!(
            scores[0] > *scores.last().unwrap(),
            "every corner scored the same, so the ordering was untested"
        );
    }

    #[test]
    fn a_band_limited_texture_has_no_fast_corners_by_construction() {
        // Why the test above cannot use `texture`, recorded so nobody puts it
        // back. FAST-9 needs nine contiguous pixels of the radius-3 circle to
        // differ from the centre by more than the threshold. On a locally
        // linear patch the brighter side is a half-circle — eight pixels — so
        // the arc can only reach nine where the surface *curves*, and the
        // curvature of this band-limited texture is tiny: its strongest term is
        // 50*sin(0.11x)*cos(...), whose second derivative gives about
        // 0.5 * 50 * 0.11^2 * 3^2 = 2.7 grey levels across the circle radius.
        //
        // Measured, that is exactly where it dies: corners exist at threshold 1
        // and 2 and vanish at 3. The original test asked it for corners at
        // threshold 12.
        let img = texture(96, 96, 0.3);
        assert!(fast_keypoints(&img, 2, 10_000).len() > 50);
        for t in [3u8, 4, 8, 12, 20, 30] {
            assert!(
                fast_keypoints(&img, t, 10_000).is_empty(),
                "threshold {t} found corners in a smooth texture"
            );
        }
        // The detector is not simply broken on this image: give it real
        // corners at the same threshold and it finds them.
        assert!(!fast_keypoints(&dots(96, 96, 5), 12, 10_000).is_empty());
    }

    #[test]
    fn high_speed_reject_uses_the_fast9_bound_not_the_fast12_one() {
        // `fast_response` rejects early unless at least MIN_COMPASS of the four
        // compass pixels {0, 4, 8, 12} are all-brighter or all-darker. That is
        // sound only if *every* contiguous arc of FAST_ARC covers at least that
        // many of them. Enumerate all 16 placements rather than trusting the
        // arithmetic: a window of 9 covers 3 when it starts on a compass point
        // and 2 otherwise, so the guaranteed coverage is 2.
        let covered =
            |len: usize, start: usize| (0..len).filter(|k| ((start + k) % 16) % 4 == 0).count();
        let min_cover = |len: usize| (0..16).map(|s| covered(len, s)).min().unwrap();
        assert_eq!(min_cover(FAST_ARC), 2);
        // 3 is the bound for FAST-**12**, which is where the wrong constant
        // came from. It discarded every FAST-9 corner whose arc is not compass
        // aligned — three placements in four, including every axis-aligned step
        // corner, which is why `fast_finds_a_synthetic_corner_at_the_right_pixel`
        // and the two tests around it all failed together.
        assert_eq!(min_cover(12), 3);

        // The behavioural half: a step corner at each of the four orientations
        // must be found. Under the FAST-12 bound none of them was.
        for (dx, dy) in [(1i32, 1i32), (-1, 1), (1, -1), (-1, -1)] {
            let mut img = GrayImage::new(21, 21);
            for y in 0..21i32 {
                for x in 0..21i32 {
                    let inside = (x - 10) * dx >= 0 && (y - 10) * dy >= 0;
                    img.data_mut()[(y * 21 + x) as usize] = if inside { 20 } else { 220 };
                }
            }
            let kps = fast_keypoints(&img, 30, 16);
            assert!(!kps.is_empty(), "quadrant ({dx}, {dy}) produced no corner");
            let best = kps[0].0;
            assert!(
                (best.x - 10.0).abs() <= 1.0 && (best.y - 10.0).abs() <= 1.0,
                "quadrant ({dx}, {dy}): strongest response at {best:?}"
            );
        }
    }

    #[test]
    fn intensity_centroid_points_at_the_bright_lobe() {
        for &deg in &[0.0f64, 45.0, 90.0, 180.0, -135.0] {
            let theta = deg.to_radians();
            let mut img = GrayImage::new(64, 64);
            let (cx, cy) = (32i32, 32i32);
            // A bright blob 8 px from the centre along `theta`.
            let bx = cx + (8.0 * theta.cos()).round() as i32;
            let by = cy + (8.0 * theta.sin()).round() as i32;
            for y in 0..64i32 {
                for x in 0..64i32 {
                    let d2 = (x - bx).pow(2) + (y - by).pow(2);
                    img.data_mut()[(y * 64 + x) as usize] = if d2 <= 16 { 255 } else { 10 };
                }
            }
            let got = intensity_centroid_orientation(&img, cx, cy);
            let err = ((got - theta + PI).rem_euclid(std::f64::consts::TAU) - PI).abs();
            assert!(err < 0.15, "{deg} deg: got {got}, error {err} rad");
        }
    }

    #[test]
    fn intensity_centroid_is_zero_on_a_symmetric_patch() {
        let flat = GrayImage::from_vec(64, 64, vec![100u8; 64 * 64]);
        assert_eq!(intensity_centroid_orientation(&flat, 32, 32), 0.0);
    }

    #[test]
    fn describe_is_a_pure_function_of_image_and_keypoint() {
        let img = texture(80, 80, 1.7);
        let kps = vec![Vec2::new(40.0, 40.0), Vec2::new(25.0, 55.0)];
        let ors = vec![0.3, -1.1];
        assert_eq!(describe(&img, &kps, &ors), describe(&img, &kps, &ors));
        // Missing orientations default to upright rather than panicking.
        assert_eq!(describe(&img, &kps, &[]).len(), 2);
        assert_eq!(describe(&img, &[], &[]).len(), 0);
    }

    #[test]
    fn describe_clamps_at_the_border_instead_of_panicking() {
        let img = texture(40, 40, 0.5);
        let kps = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(39.0, 39.0),
            Vec2::new(-5.0, 20.0),
            Vec2::new(1e6, 1e6),
        ];
        let d = describe(&img, &kps, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(d.len(), 4);
    }

    #[test]
    fn identical_patches_give_identical_descriptors() {
        let img = texture(128, 128, 2.2);
        let a = describe(&img, &[Vec2::new(64.0, 64.0)], &[0.0]);
        let b = describe(&img, &[Vec2::new(64.0, 64.0)], &[0.0]);
        assert_eq!(a[0], b[0]);
        assert_eq!(a[0].hamming(&b[0]), 0);
    }

    #[test]
    fn rotated_patch_matches_the_unrotated_one_better_than_a_random_patch() {
        // This is what the orientation compensation is FOR. Without steering,
        // the rotated descriptor is no closer to its original than a patch of
        // unrelated texture is.
        let img = texture(201, 201, 0.9);
        let centre = Vec2::new(100.0, 100.0);
        let theta0 = intensity_centroid_orientation(&img, 100, 100);
        let upright = describe(&img, &[centre], &[theta0])[0];

        // Controls: patches from independent broadband images. Several of them,
        // because one sample of a distribution whose spread is 8 bits is not a
        // control, and because the weakest of them is the bar that matters.
        let mut controls: Vec<u32> = Vec::new();
        for seed in 0..10u64 {
            let other = dots(201, 201, seed);
            for (x, y) in [(60i32, 60i32), (100, 100), (140, 90)] {
                let t = intensity_centroid_orientation(&other, x, y);
                let d = describe(&other, &[Vec2::new(x as f64, y as f64)], &[t])[0];
                controls.push(upright.hamming(&d));
            }
        }
        controls.sort_unstable();
        let weakest_control = controls[0];
        let typical_control = controls[controls.len() / 2];

        // The control this test used to use: a *different phase of the same
        // band-limited texture*, which it then asserted was more than 90 bits
        // away. That is not achievable and never was. Two patches of a smooth
        // sinusoid sum are both, locally, a ramp; steering rotates each so its
        // ramp points the same way, and the two descriptors then agree on most
        // bits by construction. Measured, this "unrelated" patch sits about 41
        // bits away and a distant patch of the *same* image about 24 — nowhere
        // near the 128 +/- 8 of independent codes. It is kept as a much tighter
        // bar for the steering to clear, which is the useful thing about it.
        let other_phase = texture(201, 201, 11.3);
        let theta_other = intensity_centroid_orientation(&other_phase, 100, 100);
        let same_family = upright.hamming(&describe(&other_phase, &[centre], &[theta_other])[0]);
        assert!(
            same_family < typical_control,
            "the same-family control ({same_family}) is supposed to be the \
             correlated one, but it beat the broadband control \
             ({typical_control}) — the finding above no longer holds"
        );

        for &deg in &[20.0f64, 45.0, 90.0, 137.0, 250.0] {
            let rot = rotate(&img, deg.to_radians());
            let theta = intensity_centroid_orientation(&rot, 100, 100);
            let steered = describe(&rot, &[centre], &[theta])[0];

            let matched = upright.hamming(&steered);
            assert!(
                matched < weakest_control,
                "{deg} deg: steered distance {matched} not better than the \
                 closest unrelated-patch control {weakest_control}"
            );
            // Tighter still: it must beat even the strongly correlated
            // same-family patch described above.
            assert!(
                matched < same_family,
                "{deg} deg: steered distance {matched} not better than the \
                 correlated same-texture control {same_family}"
            );
            // And it should be a *match*, not merely better than noise: ORB
            // accepts at ~64/256 bits.
            assert!(
                matched < 80,
                "{deg} deg: steered distance {matched} too large"
            );
        }
        // The sanity check the old `control > 90` was trying to be: the
        // broadband controls really are unrelated, i.e. they sit near the
        // 128 +/- 8 that independent 256-bit codes do.
        assert!(
            typical_control > 90,
            "control patches were not actually unrelated: {controls:?}"
        );
    }

    #[test]
    fn unsteered_descriptors_do_not_survive_rotation() {
        // The negative control for the test above: with the orientation forced
        // to zero, a 90-degree rotation destroys the match. If this ever starts
        // passing, `describe` has stopped steering and the previous test has
        // become a tautology.
        let img = texture(201, 201, 0.9);
        let centre = Vec2::new(100.0, 100.0);
        let upright = describe(&img, &[centre], &[0.0])[0];
        let rot = rotate(&img, std::f64::consts::FRAC_PI_2);
        let unsteered = describe(&rot, &[centre], &[0.0])[0];
        assert!(
            upright.hamming(&unsteered) > 70,
            "ignoring orientation should break the match"
        );
    }

    #[test]
    fn nearest_two_reports_best_and_runner_up() {
        let a = BinaryDescriptor::ZERO;
        let mut b = BinaryDescriptor::ZERO;
        b.0[0] = 0b11; // distance 2
        let mut c = BinaryDescriptor::ZERO;
        c.0[0] = 0b1111_1111; // distance 8
        let (idx, best, second) = nearest_two(&a, &[c, b]).unwrap();
        assert_eq!((idx, best, second), (1, 2, 8));
        assert!(nearest_two(&a, &[]).is_none());
        // A single candidate has no runner-up.
        let (_, _, second) = nearest_two(&a, &[b]).unwrap();
        assert_eq!(second, u32::MAX);
    }

    #[test]
    fn longest_circular_run_wraps_the_seam() {
        let mut f = [false; 16];
        for (i, slot) in f.iter_mut().enumerate() {
            *slot = !(7..14).contains(&i); // 9 in a row across the seam
        }
        assert_eq!(longest_circular_run(f), 9);
        assert_eq!(longest_circular_run([true; 16]), 16);
        assert_eq!(longest_circular_run([false; 16]), 0);
    }
}
