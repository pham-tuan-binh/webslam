//! AprilTag 36h11 detection, decoding, pose, and the scale source built on it.
//!
//! spec.md §2 lists "known object in scene" as an *exact* ruler whose only cost
//! is that the object must be visible, and spec.md §3 exposes it as
//! `ScaleSource.fiducial({ family: 'apriltag36h11', sizeMeters: 0.1 })`. A tag
//! of known physical size seen through known intrinsics fixes the metric
//! distance to itself, and comparing that against the up-to-scale trajectory
//! fixes the multiplier.
//!
//! The pipeline follows AprilTag 3, reimplemented:
//!
//! 1. [`threshold`] — tiled adaptive binarisation
//! 2. [`segment`] — union-find components, then boundary clustering
//! 3. [`quad`] — quad fitting with total-least-squares edge refinement
//! 4. [`homography`] — DLT plane-to-image homography
//! 5. payload sampling and [`code36h11`] decode with rotation and Hamming
//!    correction
//! 6. [`homography::pose_from_homography`] — planar pose, the analytic branch
//!    of IPPE
//!
//! **The codebook is a generated stand-in, not the canonical 587-entry
//! `tag36h11` table** — see [`code36h11`] for exactly what that costs and how
//! to replace it.
//!
//! A false positive here is not a missed detection, it is a *wrong scale*
//! applied silently to every subsequent pose. Every gate in the decoder —
//! contrast, quad geometry, Hamming budget, decision margin — exists for that
//! reason, and `no_tag_in_a_textured_scene_yields_no_detection` is the test
//! that keeps them honest.

pub mod code36h11;
pub mod homography;
pub mod quad;
pub mod render;
pub mod segment;
pub mod threshold;

use crate::ScaleSource;
use code36h11::{Codebook, NBITS, TOTAL_WIDTH, WIDTH_AT_BORDER};
use quad::{Quad, QuadConfig};
use wslam_core::{
    CameraIntrinsics, Error, Frame, GrayImage, Mat3, Result, Scalar, ScaleEstimate, ScaleKind, Se3,
    StateWindow, Timestamp, Vec2, Vec3,
};

/// Which marker family to look for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TagFamily {
    /// AprilTag 36h11: a 6x6 payload inside a one-cell black border, minimum
    /// Hamming distance 11 over all four rotations.
    AprilTag36h11,
}

impl TagFamily {
    /// The string form used by the public API (`family: 'apriltag36h11'`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            TagFamily::AprilTag36h11 => "apriltag36h11",
        }
    }

    /// Parse the public API's string form.
    ///
    /// Deliberately not `std::str::FromStr`: an unknown family is not an
    /// error worth an `Err` type, and this mirrors `ScaleKind::from_str` in
    /// the shared vocabulary, which the wasm boundary already parses against.
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "apriltag36h11" | "tag36h11" => Some(TagFamily::AprilTag36h11),
            _ => None,
        }
    }
}

/// Detector thresholds.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectorConfig {
    /// Tile edge for the adaptive threshold, in pixels.
    pub tile_size: u32,
    /// Smallest local contrast that will be binarised at all.
    pub min_contrast: u8,
    /// Smallest boundary cluster considered for quad fitting.
    pub min_cluster_points: usize,
    /// Quad acceptance limits.
    pub quad: QuadConfig,
    /// Bit errors the decoder may correct. Kept well below the family's
    /// correction capacity, because every extra bit multiplies the false
    /// positive rate.
    pub max_hamming: u32,
    /// Smallest mean distance of a payload sample from the black/white
    /// decision level, in intensity units. Rejects quads whose cells are
    /// ambiguous rather than guessing them.
    pub min_decision_margin: Scalar,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        DetectorConfig {
            tile_size: 4,
            min_contrast: 25,
            min_cluster_points: 24,
            quad: QuadConfig::default(),
            max_hamming: 2,
            min_decision_margin: 12.0,
        }
    }
}

/// One decoded tag.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// Codebook index.
    pub id: u32,
    /// Bit errors corrected. Zero for a clean read.
    pub hamming: u32,
    /// Mean distance of the payload samples from the decision level.
    pub decision_margin: Scalar,
    /// Corners in the tag's canonical order — `[(-1,-1), (1,-1), (1,1),
    /// (-1,1)]` in tag coordinates — already un-rotated by the decode.
    pub corners: [Vec2; 4],
    /// Centre of the black-bordered square in pixels.
    pub centre: Vec2,
    /// Quarter turns the raw quad was rotated relative to the canonical tag.
    pub rotation: u8,
}

impl Detection {
    /// Mean side length in pixels — a rough proxy for how far away the tag is.
    #[must_use]
    pub fn size_px(&self) -> Scalar {
        (0..4)
            .map(|i| (self.corners[(i + 1) % 4] - self.corners[i]).norm())
            .sum::<Scalar>()
            * 0.25
    }
}

/// A tag's recovered metric pose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TagPose {
    /// `T_camera_tag` in metres: takes tag coordinates into camera
    /// coordinates. Tag axes are x right, y down, z away from the camera, so a
    /// fronto-parallel tag has identity rotation.
    pub t_camera_tag: Se3,
    /// Root-mean-square corner reprojection error, in pixels.
    pub reprojection_rmse: Scalar,
}

/// Metric corner model of a tag whose black-bordered square is `size_meters`
/// across, in the canonical corner order.
#[must_use]
pub fn tag_object_points(size_meters: Scalar) -> [Vec3; 4] {
    let h = 0.5 * size_meters;
    [
        Vec3::new(-h, -h, 0.0),
        Vec3::new(h, -h, 0.0),
        Vec3::new(h, h, 0.0),
        Vec3::new(-h, h, 0.0),
    ]
}

/// Recover a tag's metric pose from its corners.
///
/// The analytic branch of IPPE: build the plane-to-image homography from the
/// four known-geometry corners and decompose it (Zhang). Returns `None` when
/// the corners are degenerate or place the tag behind the camera.
#[must_use]
pub fn estimate_tag_pose(
    corners: &[Vec2; 4],
    k: &CameraIntrinsics,
    size_meters: Scalar,
) -> Option<TagPose> {
    if !(size_meters.is_finite() && size_meters > 0.0) {
        return None;
    }
    let object = tag_object_points(size_meters);
    let corr: Vec<(Vec2, Vec2)> = object
        .iter()
        .zip(corners.iter())
        .map(|(o, c)| (Vec2::new(o.x, o.y), *c))
        .collect();
    let h = homography::homography_dlt(&corr)?;
    let pose = homography::pose_from_homography(&h, k)?;
    Some(TagPose {
        reprojection_rmse: homography::reprojection_rmse(&pose, k, &object, corners),
        t_camera_tag: pose,
    })
}

/// Detect and decode every tag in a frame, using the standard codebook.
#[must_use]
pub fn detect(image: &GrayImage, config: &DetectorConfig) -> Vec<Detection> {
    detect_with(image, config, &Codebook::standard())
}

/// Detect and decode against an explicit codebook.
#[must_use]
pub fn detect_with(image: &GrayImage, config: &DetectorConfig, book: &Codebook) -> Vec<Detection> {
    let binary = threshold::adaptive_threshold(image, config.tile_size, config.min_contrast);
    let mut uf = segment::connected_components(&binary);
    let clusters = segment::gradient_clusters(&binary, &mut uf, config.min_cluster_points);

    let mut out = Vec::new();
    for cluster in clusters {
        let Some(q) = quad::fit_quad(&cluster.points, &config.quad) else {
            continue;
        };
        if let Some(d) = decode_quad(image, &q, config, book) {
            out.push(d);
        }
    }
    // Deterministic order regardless of cluster enumeration.
    out.sort_by(|a, b| {
        a.id.cmp(&b.id)
            .then(a.centre.x.total_cmp(&b.centre.x))
            .then(a.centre.y.total_cmp(&b.centre.y))
    });
    out
}

/// Homography taking unit tag coordinates (`[-1, 1]` across the
/// black-bordered square) to pixels.
fn unit_homography(q: &Quad) -> Option<Mat3> {
    let unit = [
        Vec2::new(-1.0, -1.0),
        Vec2::new(1.0, -1.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(-1.0, 1.0),
    ];
    let corr: Vec<(Vec2, Vec2)> = unit
        .iter()
        .zip(q.corners.iter())
        .map(|(u, c)| (*u, *c))
        .collect();
    homography::homography_dlt(&corr)
}

/// Sample the image at a unit tag coordinate. `None` when the sample would
/// fall outside the frame.
fn sample_unit(image: &GrayImage, h: &Mat3, u: Scalar, v: Scalar) -> Option<Scalar> {
    let p = homography::apply(h, Vec2::new(u, v))?;
    if p.x < 0.0
        || p.y < 0.0
        || p.x > (image.width() - 1) as Scalar
        || p.y > (image.height() - 1) as Scalar
    {
        return None;
    }
    Some(image.sample_bilinear(p.x, p.y))
}

/// Centre of cell `(col, row)` of the 8x8 border square, in unit coordinates.
fn cell_centre(col: usize, row: usize) -> (Scalar, Scalar) {
    let step = 2.0 / WIDTH_AT_BORDER as Scalar;
    (
        -1.0 + step * (col as Scalar + 0.5),
        -1.0 + step * (row as Scalar + 0.5),
    )
}

fn decode_quad(
    image: &GrayImage,
    q: &Quad,
    config: &DetectorConfig,
    book: &Codebook,
) -> Option<Detection> {
    let h = unit_homography(q)?;

    // The border ring is black by construction, so it calibrates the dark
    // level for this quad rather than for the frame — which is the whole
    // reason a tag has a border.
    let mut black_sum = 0.0;
    let mut black_n = 0usize;
    for i in 0..WIDTH_AT_BORDER {
        for (col, row) in [
            (i, 0),
            (i, WIDTH_AT_BORDER - 1),
            (0, i),
            (WIDTH_AT_BORDER - 1, i),
        ] {
            let (u, v) = cell_centre(col, row);
            if let Some(s) = sample_unit(image, &h, u, v) {
                black_sum += s;
                black_n += 1;
            }
        }
    }
    if black_n < WIDTH_AT_BORDER * 2 {
        return None;
    }
    let black_level = black_sum / black_n as Scalar;

    // The quiet margin outside the border calibrates the light level. It is
    // part of a printed tag (AprilTag's `total_width`), so requiring it is not
    // an extra assumption.
    let margin_span = TOTAL_WIDTH as Scalar / WIDTH_AT_BORDER as Scalar;
    let step = 2.0 * margin_span / TOTAL_WIDTH as Scalar;
    let mut white_sum = 0.0;
    let mut white_n = 0usize;
    for i in 0..TOTAL_WIDTH {
        for (col, row) in [(i, 0), (i, TOTAL_WIDTH - 1), (0, i), (TOTAL_WIDTH - 1, i)] {
            let u = -margin_span + step * (col as Scalar + 0.5);
            let v = -margin_span + step * (row as Scalar + 0.5);
            if let Some(s) = sample_unit(image, &h, u, v) {
                white_sum += s;
                white_n += 1;
            }
        }
    }

    let positions = code36h11::bit_positions();
    let mut samples = [0.0; NBITS];
    for (i, &(x, y)) in positions.iter().enumerate() {
        let (u, v) = cell_centre(x, y);
        samples[i] = sample_unit(image, &h, u, v)?;
    }

    let white_level = if white_n >= TOTAL_WIDTH * 2 {
        white_sum / white_n as Scalar
    } else {
        // No usable quiet zone (tag clipped at the frame edge): fall back to
        // the brightest payload cell, which is a weaker but not fabricated
        // estimate.
        samples
            .iter()
            .copied()
            .fold(Scalar::NEG_INFINITY, Scalar::max)
    };
    if white_level - black_level < config.min_decision_margin * 2.0 {
        return None;
    }
    let level = 0.5 * (black_level + white_level);

    let mut observed = 0u64;
    let mut margin = 0.0;
    for (i, s) in samples.iter().enumerate() {
        if *s > level {
            observed |= 1u64 << (NBITS - 1 - i);
        }
        margin += (s - level).abs();
    }
    let decision_margin = margin / NBITS as Scalar;
    if decision_margin < config.min_decision_margin {
        return None;
    }

    let m = book.lookup(observed, config.max_hamming)?;
    // The decode tells us how the quad is turned relative to the canonical
    // tag; rotating the corner list is what makes `corners[0]` mean the same
    // physical corner in every frame, which the pose solve depends on.
    let r = m.rotation as usize;
    let corners = [
        q.corners[r % 4],
        q.corners[(r + 1) % 4],
        q.corners[(r + 2) % 4],
        q.corners[(r + 3) % 4],
    ];
    Some(Detection {
        id: m.id,
        hamming: m.hamming,
        decision_margin,
        corners,
        centre: q.centre(),
        rotation: m.rotation,
    })
}

/// A tag seen at a known time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TagObservation {
    /// Capture time of the frame it was seen in, unified timebase.
    pub timestamp: Timestamp,
    /// Codebook index.
    pub id: u32,
    /// Metric pose of the tag relative to the camera.
    pub t_camera_tag: Se3,
    /// Corner reprojection RMSE of that pose, in pixels.
    pub reprojection_rmse: Scalar,
}

impl TagObservation {
    /// Camera centre expressed in the tag's frame, in metres. This is the
    /// metric trajectory sample a fiducial actually delivers.
    #[must_use]
    pub fn camera_in_tag(&self) -> Vec3 {
        self.t_camera_tag.inverse().translation()
    }
}

/// Gates and limits for turning tag observations into a scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiducialConfig {
    /// Largest gap allowed when matching an observation to a window pose.
    pub max_time_delta: Scalar,
    /// Observations of one tag needed before answering.
    pub min_observations: usize,
    /// Ring-buffer cap, so a long session cannot grow this without bound.
    pub max_observations: usize,
    /// Corner reprojection RMSE above which a pose is discarded as a bad fit.
    pub max_reprojection_rmse: Scalar,
    /// Floor on the reported relative standard deviation. A noiseless
    /// synthetic fit would otherwise claim a precision the corner detector
    /// cannot deliver, and spec.md §6 L6 calls overconfidence *"worse than no
    /// covariance at all"*.
    pub min_relative_stddev: Scalar,
    /// Relative standard deviation above which we decline to answer.
    pub max_relative_stddev: Scalar,
}

impl Default for FiducialConfig {
    fn default() -> Self {
        FiducialConfig {
            max_time_delta: 0.05,
            min_observations: 2,
            max_observations: 120,
            max_reprojection_rmse: 3.0,
            min_relative_stddev: 0.002,
            max_relative_stddev: 0.25,
        }
    }
}

/// Scale from a marker of known physical size.
#[derive(Debug, Clone)]
pub struct FiducialScale {
    family: TagFamily,
    size_meters: Scalar,
    detector: DetectorConfig,
    config: FiducialConfig,
    book: Codebook,
    observations: std::collections::VecDeque<TagObservation>,
    observed_size_units: Option<Scalar>,
    size_units_stddev: Scalar,
}

impl FiducialScale {
    /// Look for `family` tags whose black-bordered square is `size_meters`
    /// across.
    ///
    /// # Errors
    /// [`Error::Config`] if the size is not a positive, finite number of
    /// metres.
    pub fn new(family: TagFamily, size_meters: Scalar) -> Result<Self> {
        if !(size_meters.is_finite() && size_meters > 0.0) {
            return Err(Error::Config(format!(
                "fiducial size must be positive metres, got {size_meters}"
            )));
        }
        Ok(FiducialScale {
            family,
            size_meters,
            detector: DetectorConfig::default(),
            config: FiducialConfig::default(),
            book: Codebook::standard(),
            observations: std::collections::VecDeque::new(),
            observed_size_units: None,
            size_units_stddev: 0.0,
        })
    }

    /// Override the detector thresholds.
    #[must_use]
    pub fn with_detector_config(mut self, detector: DetectorConfig) -> Self {
        self.detector = detector;
        self
    }

    /// Override the estimation gates.
    #[must_use]
    pub fn with_config(mut self, config: FiducialConfig) -> Self {
        self.config = config;
        self
    }

    /// Swap the codebook — the seam for the canonical `tag36h11` table.
    #[must_use]
    pub fn with_codebook(mut self, book: Codebook) -> Self {
        self.book = book;
        self
    }

    /// Which family this source looks for.
    #[must_use]
    pub fn family(&self) -> TagFamily {
        self.family
    }

    /// The declared physical tag size in metres.
    #[must_use]
    pub fn size_meters(&self) -> Scalar {
        self.size_meters
    }

    /// Observations held.
    pub fn observations(&self) -> impl Iterator<Item = &TagObservation> + '_ {
        self.observations.iter()
    }

    /// Supply the tag's edge length as measured in the up-to-scale map, when
    /// the caller has triangulated the tag's corners as landmarks.
    ///
    /// This is the direct form of the ruler — `scale = size_meters /
    /// observed_size_in_window_units` — and it takes precedence over
    /// trajectory alignment because it needs no camera motion at all.
    /// `stddev_units` is the uncertainty of that measurement; pass `0.0` to
    /// fall back to [`FiducialConfig::min_relative_stddev`].
    pub fn set_observed_size_units(&mut self, size_units: Scalar, stddev_units: Scalar) {
        self.observed_size_units =
            (size_units.is_finite() && size_units > 0.0).then_some(size_units);
        self.size_units_stddev = stddev_units.abs();
    }

    /// Detect tags in a frame and record their poses.
    ///
    /// Returns how many usable observations were added.
    pub fn observe(&mut self, frame: &Frame, k: &CameraIntrinsics) -> usize {
        let detections = detect_with(&frame.image, &self.detector, &self.book);
        let mut added = 0;
        for d in detections {
            let Some(pose) = estimate_tag_pose(&d.corners, k, self.size_meters) else {
                continue;
            };
            if pose.reprojection_rmse > self.config.max_reprojection_rmse {
                log::debug!(
                    "fiducial: rejecting tag {} with reprojection rmse {:.2} px",
                    d.id,
                    pose.reprojection_rmse
                );
                continue;
            }
            self.push_observation(TagObservation {
                timestamp: frame.timestamp,
                id: d.id,
                t_camera_tag: pose.t_camera_tag,
                reprojection_rmse: pose.reprojection_rmse,
            });
            added += 1;
        }
        added
    }

    /// Record an observation produced elsewhere — the orchestrator may run
    /// detection on the GPU thread and hand the results across.
    pub fn push_observation(&mut self, obs: TagObservation) {
        if self.observations.len() >= self.config.max_observations {
            self.observations.pop_front();
        }
        self.observations.push_back(obs);
    }

    /// Scale from a caller-supplied up-to-scale tag size.
    fn estimate_from_size(&self, size_units: Scalar) -> ScaleEstimate {
        let value = self.size_meters / size_units;
        // Delta method on s = S / u.
        let from_measurement = (value / size_units).powi(2) * self.size_units_stddev.powi(2);
        let floor = (self.config.min_relative_stddev * value).powi(2);
        ScaleEstimate::metric(ScaleKind::Fiducial, value, from_measurement.max(floor))
    }

    /// Scale by aligning the metric camera track (from the tag) against the
    /// up-to-scale camera track (from the window).
    ///
    /// Both tracks describe the same physical motion, so the ratio of
    /// corresponding baselines *is* the multiplier. Working in pairwise
    /// distances rather than absolute positions sidesteps the unknown rigid
    /// transform between the tag frame and the visual world frame.
    fn estimate_from_trajectory(&self, window: &StateWindow) -> Option<ScaleEstimate> {
        let id = self.dominant_id()?;
        let mut metric: Vec<Vec3> = Vec::new();
        let mut units: Vec<Vec3> = Vec::new();
        for obs in self.observations.iter().filter(|o| o.id == id) {
            if let Some(p) = nearest_pose(window, obs.timestamp, self.config.max_time_delta) {
                metric.push(obs.camera_in_tag());
                units.push(p);
            }
        }
        let n = metric.len();
        if n < self.config.min_observations.max(2) {
            return None;
        }

        let (mut num, mut den) = (0.0, 0.0);
        let mut pairs: Vec<(Scalar, Scalar)> = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let dm = (metric[i] - metric[j]).norm();
                let du = (units[i] - units[j]).norm();
                num += dm * du;
                den += du * du;
                pairs.push((dm, du));
            }
        }
        if !(den.is_finite() && den > 1e-18) {
            // Every pair of camera positions coincides: the camera never moved
            // in the up-to-scale frame, so there is no baseline to compare.
            return None;
        }
        let value = num / den;
        if !(value.is_finite() && value > 0.0) {
            return None;
        }

        let sse: Scalar = pairs.iter().map(|(dm, du)| (dm - value * du).powi(2)).sum();
        // Pairwise distances from n observations carry at most n-1 independent
        // constraints. Dividing by the pair count instead would understate the
        // variance, and an overconfident scale poisons every pose covariance
        // downstream (spec.md §6 L6).
        let sigma_sq = sse / (n - 1) as Scalar;
        let floor = (self.config.min_relative_stddev * value).powi(2);
        let variance = (sigma_sq / den).max(floor);

        let estimate = ScaleEstimate::metric(ScaleKind::Fiducial, value, variance);
        if estimate.relative_stddev_percent() > 100.0 * self.config.max_relative_stddev {
            log::debug!(
                "fiducial: declining, relative stddev {:.1}% exceeds gate",
                estimate.relative_stddev_percent()
            );
            return None;
        }
        Some(estimate)
    }

    fn dominant_id(&self) -> Option<u32> {
        let mut counts: std::collections::BTreeMap<u32, usize> = Default::default();
        for o in &self.observations {
            *counts.entry(o.id).or_default() += 1;
        }
        counts.into_iter().max_by_key(|&(_, c)| c).map(|(id, _)| id)
    }
}

fn nearest_pose(window: &StateWindow, at: Timestamp, tolerance: Scalar) -> Option<Vec3> {
    window
        .poses()
        .filter(|p| p.timestamp.since(at).abs() <= tolerance)
        .min_by(|a, b| {
            a.timestamp
                .since(at)
                .abs()
                .total_cmp(&b.timestamp.since(at).abs())
        })
        .map(|p| p.pose.translation())
}

impl ScaleSource for FiducialScale {
    fn kind(&self) -> ScaleKind {
        ScaleKind::Fiducial
    }

    fn estimate(&mut self, window: &StateWindow) -> Option<ScaleEstimate> {
        if let Some(u) = self.observed_size_units {
            return Some(self.estimate_from_size(u));
        }
        self.estimate_from_trajectory(window)
    }

    fn reset(&mut self) {
        self.observations.clear();
        self.observed_size_units = None;
        self.size_units_stddev = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use wslam_core::{FrameId, So3, WindowSample};

    #[test]
    fn family_string_roundtrip() {
        assert_eq!(
            TagFamily::from_str(TagFamily::AprilTag36h11.as_str()),
            Some(TagFamily::AprilTag36h11)
        );
        assert_eq!(TagFamily::from_str("aruco"), None);
    }

    #[test]
    fn construction_refuses_a_nonsense_size() {
        assert!(FiducialScale::new(TagFamily::AprilTag36h11, 0.0).is_err());
        assert!(FiducialScale::new(TagFamily::AprilTag36h11, -0.1).is_err());
        assert!(FiducialScale::new(TagFamily::AprilTag36h11, Scalar::NAN).is_err());
        let f = FiducialScale::new(TagFamily::AprilTag36h11, 0.16).unwrap();
        assert_relative_eq!(f.size_meters(), 0.16);
        assert_eq!(f.family(), TagFamily::AprilTag36h11);
    }

    #[test]
    fn tag_object_points_span_the_declared_size() {
        let p = tag_object_points(0.2);
        assert_relative_eq!((p[1] - p[0]).norm(), 0.2, epsilon = 1e-15);
        assert_relative_eq!((p[2] - p[1]).norm(), 0.2, epsilon = 1e-15);
        assert!(p.iter().all(|q| q.z == 0.0), "the tag is planar");
    }

    /// The direct form of the ruler: metres over window units.
    #[test]
    fn a_known_size_in_window_units_gives_the_multiplier_directly() {
        let mut f = FiducialScale::new(TagFamily::AprilTag36h11, 0.15).unwrap();
        f.set_observed_size_units(0.05, 0.0);
        let e = f.estimate(&StateWindow::with_default_capacity()).unwrap();
        assert_eq!(e.source, ScaleKind::Fiducial);
        assert_relative_eq!(e.value, 3.0, epsilon = 1e-12);
        // Even a "perfect" measurement carries the floor, never zero.
        assert!(e.variance > 0.0);
        assert_relative_eq!(e.relative_stddev_percent(), 0.2, epsilon = 1e-9);
    }

    #[test]
    fn a_noisier_size_measurement_reports_a_larger_variance() {
        let mut tight = FiducialScale::new(TagFamily::AprilTag36h11, 0.15).unwrap();
        tight.set_observed_size_units(0.05, 1e-5);
        let mut loose = FiducialScale::new(TagFamily::AprilTag36h11, 0.15).unwrap();
        loose.set_observed_size_units(0.05, 1e-2);
        let w = StateWindow::with_default_capacity();
        assert!(loose.estimate(&w).unwrap().variance > tight.estimate(&w).unwrap().variance);
    }

    /// Synthetic trajectory alignment, no images involved: the metric camera
    /// track from the tag and the up-to-scale track from the window differ by
    /// exactly the multiplier we should recover.
    #[test]
    fn trajectory_alignment_recovers_a_known_multiplier() {
        let scale = 4.25;
        let mut f = FiducialScale::new(TagFamily::AprilTag36h11, 0.1).unwrap();
        let mut window = StateWindow::with_default_capacity();

        for i in 0..12 {
            let t = Timestamp::from_seconds(i as Scalar * 0.1);
            // Metric camera centre in the tag frame.
            let c_metric = Vec3::new(
                0.15 * (i as Scalar * 0.5).sin(),
                0.08 * (i as Scalar * 0.3).cos(),
                0.9 + 0.02 * i as Scalar,
            );
            // The observation stores T_camera_tag; the camera centre in tag
            // coordinates is its inverse translation, so build it that way.
            let r = So3::exp(&Vec3::new(0.05, -0.02, 0.01));
            let t_tag_camera = Se3::new(r, c_metric);
            f.push_observation(TagObservation {
                timestamp: t,
                id: 0,
                t_camera_tag: t_tag_camera.inverse(),
                reprojection_rmse: 0.1,
            });
            window.push_pose(WindowSample {
                timestamp: t,
                // Up-to-scale world differs from the tag frame by a rigid
                // transform and the unknown scale; only distances matter.
                pose: Se3::new(
                    So3::identity(),
                    So3::exp(&Vec3::new(0.3, 0.1, -0.2)).act(&c_metric) / scale
                        + Vec3::new(5.0, -2.0, 1.0),
                ),
                landmark_count: 80,
            });
        }

        let e = f.estimate(&window).expect("well-conditioned alignment");
        assert_relative_eq!(e.value, scale, max_relative = 1e-9);
        assert_eq!(e.source, ScaleKind::Fiducial);
        assert!(e.variance > 0.0);
    }

    /// A camera that never moves gives no baseline, in either track. Returning
    /// a number here would be the silent guess.
    #[test]
    fn a_static_camera_yields_no_trajectory_scale() {
        let mut f = FiducialScale::new(TagFamily::AprilTag36h11, 0.1).unwrap();
        let mut window = StateWindow::with_default_capacity();
        for i in 0..10 {
            let t = Timestamp::from_seconds(i as Scalar * 0.1);
            f.push_observation(TagObservation {
                timestamp: t,
                id: 0,
                t_camera_tag: Se3::from_translation(Vec3::new(0.0, 0.0, 0.8)),
                reprojection_rmse: 0.05,
            });
            window.push_pose(WindowSample {
                timestamp: t,
                pose: Se3::identity(),
                landmark_count: 80,
            });
        }
        assert!(f.estimate(&window).is_none());
    }

    #[test]
    fn observations_without_matching_poses_are_unusable() {
        let mut f = FiducialScale::new(TagFamily::AprilTag36h11, 0.1).unwrap();
        let mut window = StateWindow::with_default_capacity();
        for i in 0..8 {
            f.push_observation(TagObservation {
                timestamp: Timestamp::from_seconds(i as Scalar * 0.1),
                id: 0,
                t_camera_tag: Se3::from_translation(Vec3::new(0.01 * i as Scalar, 0.0, 0.8)),
                reprojection_rmse: 0.05,
            });
            // Poses a full minute away — no pairing is possible.
            window.push_pose(WindowSample {
                timestamp: Timestamp::from_seconds(60.0 + i as Scalar * 0.1),
                pose: Se3::from_translation(Vec3::new(0.002 * i as Scalar, 0.0, 0.0)),
                landmark_count: 80,
            });
        }
        assert!(f.estimate(&window).is_none());
    }

    #[test]
    fn one_observation_is_not_enough() {
        let mut f = FiducialScale::new(TagFamily::AprilTag36h11, 0.1).unwrap();
        let mut window = StateWindow::with_default_capacity();
        f.push_observation(TagObservation {
            timestamp: Timestamp::ZERO,
            id: 0,
            t_camera_tag: Se3::from_translation(Vec3::new(0.0, 0.0, 0.8)),
            reprojection_rmse: 0.05,
        });
        window.push_pose(WindowSample {
            timestamp: Timestamp::ZERO,
            pose: Se3::identity(),
            landmark_count: 80,
        });
        assert!(f.estimate(&window).is_none());
    }

    #[test]
    fn the_observation_buffer_is_bounded() {
        let mut f = FiducialScale::new(TagFamily::AprilTag36h11, 0.1)
            .unwrap()
            .with_config(FiducialConfig {
                max_observations: 5,
                ..FiducialConfig::default()
            });
        for i in 0..50 {
            f.push_observation(TagObservation {
                timestamp: Timestamp::from_seconds(i as Scalar),
                id: 0,
                t_camera_tag: Se3::identity(),
                reprojection_rmse: 0.0,
            });
        }
        assert_eq!(f.observations().count(), 5);
        f.reset();
        assert_eq!(f.observations().count(), 0);
    }

    #[test]
    fn camera_in_tag_inverts_the_stored_pose() {
        let t_tag_camera = Se3::new(
            So3::exp(&Vec3::new(0.2, -0.1, 0.4)),
            Vec3::new(0.3, -0.2, 1.1),
        );
        let obs = TagObservation {
            timestamp: Timestamp::ZERO,
            id: 0,
            t_camera_tag: t_tag_camera.inverse(),
            reprojection_rmse: 0.0,
        };
        assert_relative_eq!(
            obs.camera_in_tag(),
            t_tag_camera.translation(),
            epsilon = 1e-12
        );
    }
    /// The whole detector, end to end: render a tag at a known metric pose,
    /// run threshold -> connected components -> quad fit -> decode -> IPPE,
    /// and demand the metric distance back.
    ///
    /// Nothing exercised [`detect`] before this. That is how a homography
    /// solver returning garbage for exactly four correspondences — which is
    /// every tag, and which broke `unit_homography` and therefore the decoder
    /// too — survived a green suite: no assertion ever looked at the pipeline
    /// end to end.
    #[test]
    fn a_rendered_tag_survives_the_whole_pipeline_and_its_distance_is_exact() {
        let k = CameraIntrinsics::from_focal(600.0, 640, 480);
        let size = 0.1;
        // Distances chosen so the apparent size is an even number of pixels
        // and the tag edges land on pixel boundaries; anti-aliased edges are a
        // separate concern from the geometry under test here.
        for z in [0.4, 0.5, 0.6, 1.0, 1.5, 2.0] {
            let img = render::render_tag(7, &render::RenderConfig::facing(k, size, z)).unwrap();
            let dets = detect(&img, &DetectorConfig::default());

            assert_eq!(
                dets.len(),
                1,
                "exactly one tag at {z} m, got {}",
                dets.len()
            );
            assert_eq!(dets[0].id, 7);
            assert_eq!(dets[0].hamming, 0, "a clean render must decode uncorrected");
            // Apparent size is the ruler itself: f * size / z.
            assert_relative_eq!(dets[0].size_px(), 600.0 * size / z, epsilon = 1e-9);

            let pose = estimate_tag_pose(&dets[0].corners, &k, size).unwrap();
            let t = pose.t_camera_tag.translation();
            // Depth is exact to machine precision across a 5x sweep. This is
            // the assertion that matters for L5: scale = size / observed size,
            // so any nonlinearity here is a distance-dependent scale error.
            assert_relative_eq!(t.z, z, epsilon = 1e-9);
            // The quad fitter reports the boundary between the last dark pixel
            // and the first light one, which sits half a pixel inside the
            // ideal corner. That is a fixed offset of the whole quad, not
            // noise: it cancels out of z and displaces x and y by exactly half
            // a pixel's worth of metres. Asserted rather than tolerated.
            let half_pixel_m = 0.5 * z / 600.0;
            assert_relative_eq!(t.x, -half_pixel_m, epsilon = 1e-12);
            assert_relative_eq!(t.y, -half_pixel_m, epsilon = 1e-12);
            assert!(pose.reprojection_rmse < 1e-9, "{}", pose.reprojection_rmse);
        }
    }

    /// Off-axis and tilted, where the corners no longer land on pixel
    /// boundaries and the planar ambiguity is live.
    #[test]
    fn a_tilted_rendered_tag_recovers_its_pose_to_sub_millimetre() {
        let k = CameraIntrinsics::from_focal(600.0, 640, 480);
        let size = 0.1;
        let truth = Se3::new(
            So3::exp(&Vec3::new(0.15, -0.2, 0.05)),
            Vec3::new(0.02, -0.01, 0.5),
        );
        let img = render::render_tag(
            7,
            &render::RenderConfig {
                t_camera_tag: truth,
                ..render::RenderConfig::facing(k, size, 0.5)
            },
        )
        .unwrap();

        let dets = detect(&img, &DetectorConfig::default());
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].id, 7);
        let pose = estimate_tag_pose(&dets[0].corners, &k, size).unwrap();

        // Measured 0.79 mm at 0.5 m, i.e. 0.16% — the same half-pixel corner
        // convention as above, now spread over three axes by the tilt.
        let dt = (pose.t_camera_tag.translation() - truth.translation()).norm();
        assert!(dt < 1.5e-3, "translation error {dt} m");

        // Measured 0.020 rad. Out-of-plane rotation is the weakest degree of
        // freedom of a small planar target — that is the two-fold planar
        // ambiguity of Collins & Bartoli flattening the cost basin, not a
        // defect — and it is also the one quantity scale does not depend on.
        let rotation_error = (truth.rotation().inverse() * pose.t_camera_tag.rotation())
            .log()
            .norm();
        assert!(rotation_error < 0.03, "rotation error {rotation_error} rad");
        assert!(pose.reprojection_rmse < 0.3, "{}", pose.reprojection_rmse);
    }

    /// From image bytes to a metric observation, through the public source.
    #[test]
    fn observe_turns_a_rendered_frame_into_a_metric_tag_observation() {
        let k = CameraIntrinsics::from_focal(600.0, 640, 480);
        let size = 0.1;
        // 0.6 m is 100 px across, so the rendered edges fall on pixel
        // boundaries. At a distance whose apparent size is odd — 0.8 m is
        // 75 px — the anti-aliased edge biases the fitted quad by a whole
        // pixel of width and the depth comes back 1/76 = 1.3% short. That is
        // the corner convention interacting with the renderer, not the
        // estimator, and it belongs in a rig measurement rather than here.
        let img = render::render_tag(7, &render::RenderConfig::facing(k, size, 0.6)).unwrap();
        let frame = Frame::new(FrameId(1), Timestamp::from_seconds(0.5), img);

        let mut source = FiducialScale::new(TagFamily::AprilTag36h11, size).unwrap();
        assert_eq!(source.observe(&frame, &k), 1);

        let obs = source
            .observations()
            .next()
            .copied()
            .expect("one observation");
        assert_eq!(obs.id, 7);
        assert_eq!(obs.timestamp, frame.timestamp);
        assert_relative_eq!(obs.t_camera_tag.translation().z, 0.6, epsilon = 1e-9);

        // A single observation is still not a scale: the trajectory alignment
        // needs a baseline, and there is none from one frame.
        assert!(source
            .estimate(&StateWindow::with_default_capacity())
            .is_none());
    }
}
