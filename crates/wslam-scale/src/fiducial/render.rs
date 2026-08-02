//! Synthetic tag rendering.
//!
//! Not a runtime path — this exists so the detector can be tested against
//! ground truth that is known exactly rather than measured. spec.md §6 Tier 1
//! is *"everything with a closed-form answer, on synthetic input"*, and for a
//! fiducial the closed-form answer is "we placed the tag at this pose, at this
//! size; recover it". It is also the frame source for the arm-rig harness when
//! a physical target is not to hand.
//!
//! Rendering ray-casts each pixel onto the tag plane, so perspective, lens
//! distortion and off-axis placement all come out right by construction rather
//! than by an approximate warp. Supersampling antialiases the cell edges, which
//! matters: a hard-edged render quantises the boundary points the quad fitter
//! sees and puts a floor on corner accuracy that has nothing to do with the
//! detector.

use super::code36h11::{self, Codebook, TOTAL_WIDTH, WIDTH_AT_BORDER};
use wslam_core::{
    CameraIntrinsics, DeterministicRng, Error, GrayImage, Result, Scalar, Se3, Vec2, Vec3,
};

/// How to draw a tag.
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// Camera the tag is seen through. Distortion is applied.
    pub intrinsics: CameraIntrinsics,
    /// Pose of the tag relative to the camera, `T_camera_tag`.
    pub t_camera_tag: Se3,
    /// Edge length of the **black-bordered square** in metres — the same
    /// quantity `FiducialScale::new` takes, and the one AprilTag calls the tag
    /// size. The white quiet margin is drawn outside it.
    pub size_meters: Scalar,
    /// Samples per pixel per axis. 1 disables antialiasing.
    pub supersample: u32,
    /// Intensity of a black cell.
    pub black: u8,
    /// Intensity of a white cell and of the quiet margin.
    pub white: u8,
    /// Intensity of everything that is not the tag.
    pub background: u8,
    /// Standard deviation of additive Gaussian sensor noise, in intensity
    /// units.
    pub noise_stddev: Scalar,
    /// Seed for that noise. spec.md §6: every RNG is seeded.
    pub seed: u64,
}

impl RenderConfig {
    /// A tag facing the camera at `distance` metres, filling a comfortable
    /// fraction of the frame.
    #[must_use]
    pub fn facing(intrinsics: CameraIntrinsics, size_meters: Scalar, distance: Scalar) -> Self {
        RenderConfig {
            intrinsics,
            t_camera_tag: Se3::from_translation(Vec3::new(0.0, 0.0, distance)),
            size_meters,
            supersample: 3,
            black: 25,
            white: 230,
            background: 230,
            noise_stddev: 0.0,
            seed: 20260801,
        }
    }
}

/// Draw tag `id` from the standard codebook.
///
/// # Errors
/// [`Error::Config`] if the id is not in the codebook or the size is not
/// positive.
pub fn render_tag(id: u32, config: &RenderConfig) -> Result<GrayImage> {
    render_tag_with(&Codebook::standard(), id, config)
}

/// Draw tag `id` from an explicit codebook.
///
/// # Errors
/// [`Error::Config`] if the id is not in the codebook or the size is not
/// positive.
pub fn render_tag_with(book: &Codebook, id: u32, config: &RenderConfig) -> Result<GrayImage> {
    let code = book
        .code(id)
        .ok_or_else(|| Error::Config(format!("tag id {id} is not in the codebook")))?;
    if !(config.size_meters.is_finite() && config.size_meters > 0.0) {
        return Err(Error::Config(format!(
            "tag size must be positive metres, got {}",
            config.size_meters
        )));
    }

    let cells = cell_grid(code);
    let k = &config.intrinsics;
    let (w, h) = (k.width, k.height);
    let mut img = GrayImage::new(w, h);
    img.data_mut().fill(config.background);

    let half = 0.5 * config.size_meters;
    let margin = half * TOTAL_WIDTH as Scalar / WIDTH_AT_BORDER as Scalar;
    let r = config.t_camera_tag.rotation();
    let t = config.t_camera_tag.translation();
    let normal = r.act(&Vec3::z());
    let plane_offset = normal.dot(&t);

    let (x0, y0, x1, y1) = bounding_box(config, margin);
    let ss = config.supersample.max(1);
    let inv_ss = 1.0 / ss as Scalar;
    let weight = inv_ss * inv_ss;

    for py in y0..y1 {
        for px in x0..x1 {
            let mut acc = 0.0;
            for sy in 0..ss {
                for sx in 0..ss {
                    let sample = Vec2::new(
                        px as Scalar + (sx as Scalar + 0.5) * inv_ss,
                        py as Scalar + (sy as Scalar + 0.5) * inv_ss,
                    );
                    acc += shade(
                        config,
                        &cells,
                        k,
                        &r,
                        &normal,
                        plane_offset,
                        half,
                        margin,
                        sample,
                        t,
                    );
                }
            }
            img.data_mut()[(py * w + px) as usize] = (acc * weight).round().clamp(0.0, 255.0) as u8;
        }
    }

    if config.noise_stddev > 0.0 {
        let mut rng = DeterministicRng::new("fiducial-render-noise", config.seed);
        for v in img.data_mut() {
            let n = *v as Scalar + rng.normal() * config.noise_stddev;
            *v = n.round().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(img)
}

/// Intensity of one sub-pixel sample.
#[allow(clippy::too_many_arguments)] // a config struct here would be a struct per pixel
fn shade(
    config: &RenderConfig,
    cells: &[[bool; WIDTH_AT_BORDER]; WIDTH_AT_BORDER],
    k: &CameraIntrinsics,
    r: &wslam_core::So3,
    normal: &Vec3,
    plane_offset: Scalar,
    half: Scalar,
    margin: Scalar,
    pixel: Vec2,
    t: Vec3,
) -> Scalar {
    let bearing = k.unproject_bearing(pixel);
    let denom = normal.dot(&bearing);
    if denom.abs() < 1e-12 {
        return config.background as Scalar;
    }
    let lambda = plane_offset / denom;
    if lambda <= 0.0 {
        return config.background as Scalar; // plane is behind the camera
    }
    let p_tag = r.inverse().act(&(bearing * lambda - t));
    let (u, v) = (p_tag.x, p_tag.y);

    if u.abs() <= half && v.abs() <= half {
        let cell = half * 2.0 / WIDTH_AT_BORDER as Scalar;
        let col = (((u + half) / cell) as usize).min(WIDTH_AT_BORDER - 1);
        let row = (((v + half) / cell) as usize).min(WIDTH_AT_BORDER - 1);
        if cells[row][col] {
            config.white as Scalar
        } else {
            config.black as Scalar
        }
    } else if u.abs() <= margin && v.abs() <= margin {
        config.white as Scalar
    } else {
        config.background as Scalar
    }
}

/// Pixel bounding box of the printed tag, clamped to the frame.
fn bounding_box(config: &RenderConfig, margin: Scalar) -> (u32, u32, u32, u32) {
    let k = &config.intrinsics;
    let full = (0, 0, k.width, k.height);
    let corners = [
        Vec3::new(-margin, -margin, 0.0),
        Vec3::new(margin, -margin, 0.0),
        Vec3::new(margin, margin, 0.0),
        Vec3::new(-margin, margin, 0.0),
    ];
    let (mut lo_x, mut lo_y) = (Scalar::INFINITY, Scalar::INFINITY);
    let (mut hi_x, mut hi_y) = (Scalar::NEG_INFINITY, Scalar::NEG_INFINITY);
    for c in corners {
        match k.project(&config.t_camera_tag.act(&c)) {
            // Any corner behind the camera makes the projected outline
            // meaningless; fall back to the whole frame rather than clip
            // something we cannot bound.
            None => return full,
            Some(p) => {
                lo_x = lo_x.min(p.x);
                lo_y = lo_y.min(p.y);
                hi_x = hi_x.max(p.x);
                hi_y = hi_y.max(p.y);
            }
        }
    }
    // Two pixels of slack for distortion curving the outline outwards.
    let pad = 2.0;
    (
        (lo_x - pad).floor().clamp(0.0, k.width as Scalar) as u32,
        (lo_y - pad).floor().clamp(0.0, k.height as Scalar) as u32,
        (hi_x + pad).ceil().clamp(0.0, k.width as Scalar) as u32,
        (hi_y + pad).ceil().clamp(0.0, k.height as Scalar) as u32,
    )
}

/// The 8x8 cell grid: a one-cell black border around the 6x6 payload.
/// `true` is white.
#[must_use]
pub fn cell_grid(code: u64) -> [[bool; WIDTH_AT_BORDER]; WIDTH_AT_BORDER] {
    let data = code36h11::to_grid(code);
    let mut cells = [[false; WIDTH_AT_BORDER]; WIDTH_AT_BORDER];
    for (r, row) in cells.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            let border = r == 0 || c == 0 || r == WIDTH_AT_BORDER - 1 || c == WIDTH_AT_BORDER - 1;
            *cell = !border && data[r - 1][c - 1];
        }
    }
    cells
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k() -> CameraIntrinsics {
        CameraIntrinsics::from_focal(600.0, 320, 240)
    }

    #[test]
    fn the_border_ring_is_always_black() {
        for id in [0u32, 5, 63] {
            let cells = cell_grid(Codebook::standard().code(id).unwrap());
            for (i, row) in cells.iter().enumerate() {
                assert!(!cells[0][i] && !cells[WIDTH_AT_BORDER - 1][i]);
                assert!(!row[0] && !row[WIDTH_AT_BORDER - 1]);
            }
        }
    }

    #[test]
    fn a_fronto_parallel_tag_lands_where_the_pinhole_says_it_should() {
        let cfg = RenderConfig::facing(k(), 0.1, 0.6);
        let img = render_tag(0, &cfg).unwrap();
        // Tag half-width 0.05 m at 0.6 m through f = 600 px spans
        // 2 * 600 * 0.05 / 0.6 = 100 px, centred.
        assert!(img.at(160, 120) != 0, "something was drawn");
        // Just inside the border: black. Just outside the margin: background.
        assert!(img.at(160, 120 - 48) < 60, "border ring should be dark");
        assert!(
            img.at(160, 120 - 70) > 200,
            "outside the tag should be light"
        );
    }

    #[test]
    fn tag_size_scales_inversely_with_distance() {
        fn dark_pixels(distance: Scalar) -> usize {
            let cfg = RenderConfig::facing(k(), 0.1, distance);
            let img = render_tag(1, &cfg).unwrap();
            img.data().iter().filter(|&&v| v < 128).count()
        }
        let near = dark_pixels(0.5);
        let far = dark_pixels(1.0);
        // Area goes as 1/d^2, so halving the distance quadruples it.
        let ratio = near as Scalar / far as Scalar;
        assert!((3.4..4.6).contains(&ratio), "area ratio {ratio}");
    }

    #[test]
    fn noise_is_seeded_and_reproducible() {
        let mut cfg = RenderConfig::facing(k(), 0.1, 0.6);
        cfg.noise_stddev = 6.0;
        let a = render_tag(2, &cfg).unwrap();
        let b = render_tag(2, &cfg).unwrap();
        assert_eq!(a.data(), b.data());

        cfg.seed += 1;
        let c = render_tag(2, &cfg).unwrap();
        assert_ne!(a.data(), c.data());
    }

    #[test]
    fn unknown_ids_and_bad_sizes_are_refused() {
        let cfg = RenderConfig::facing(k(), 0.1, 0.6);
        assert!(render_tag(99_999, &cfg).is_err());
        let mut bad = cfg.clone();
        bad.size_meters = 0.0;
        assert!(render_tag(0, &bad).is_err());
    }

    #[test]
    fn a_tag_behind_the_camera_draws_nothing() {
        let mut cfg = RenderConfig::facing(k(), 0.1, 0.6);
        cfg.t_camera_tag = Se3::from_translation(Vec3::new(0.0, 0.0, -0.6));
        let img = render_tag(0, &cfg).unwrap();
        assert!(img.data().iter().all(|&v| v == cfg.background));
    }
}
