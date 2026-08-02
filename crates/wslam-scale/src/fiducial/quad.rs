//! Quad fitting with line-fit refinement — stage three of the detector.
//!
//! Corner accuracy is the whole ballgame for a fiducial ruler: scale enters
//! through the tag's apparent size, so a half-pixel bias on opposite corners of
//! a 60 px tag is a 1.7% scale error. Taking corners straight from the
//! boundary-point extrema would leave exactly that kind of bias, quantised to
//! the pixel grid. So the extrema only *segment* the contour; the corners
//! themselves come from intersecting total-least-squares lines fitted to the
//! four sides, which averages the per-pixel quantisation over dozens of points
//! and lands well inside a tenth of a pixel.

use wslam_core::{Scalar, Vec2};

/// Four corners in image order, wound so that the shoelace area is positive
/// (clockwise on screen, since image `y` points down).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quad {
    /// Corners, consecutive entries joined by a side.
    pub corners: [Vec2; 4],
}

impl Quad {
    /// Signed area by the shoelace formula. Positive for our winding.
    #[must_use]
    pub fn signed_area(&self) -> Scalar {
        let c = &self.corners;
        0.5 * (0..4)
            .map(|i| {
                let a = c[i];
                let b = c[(i + 1) % 4];
                a.x * b.y - b.x * a.y
            })
            .sum::<Scalar>()
    }

    /// Length of the shortest side.
    #[must_use]
    pub fn min_side(&self) -> Scalar {
        (0..4)
            .map(|i| (self.corners[(i + 1) % 4] - self.corners[i]).norm())
            .fold(Scalar::INFINITY, Scalar::min)
    }

    /// Length of the longest side.
    #[must_use]
    pub fn max_side(&self) -> Scalar {
        (0..4)
            .map(|i| (self.corners[(i + 1) % 4] - self.corners[i]).norm())
            .fold(0.0, Scalar::max)
    }

    /// Whether every interior turn has the same sign — a self-intersecting or
    /// reflex "quad" is a fitting failure, not a tag.
    #[must_use]
    pub fn is_convex(&self) -> bool {
        let c = &self.corners;
        let mut sign = 0.0;
        for i in 0..4 {
            let a = c[(i + 1) % 4] - c[i];
            let b = c[(i + 2) % 4] - c[(i + 1) % 4];
            let cross = a.x * b.y - a.y * b.x;
            if cross.abs() < 1e-12 {
                return false;
            }
            if sign == 0.0 {
                sign = cross.signum();
            } else if cross.signum() != sign {
                return false;
            }
        }
        true
    }

    /// Centroid of the corners.
    #[must_use]
    pub fn centre(&self) -> Vec2 {
        self.corners.iter().sum::<Vec2>() / 4.0
    }
}

/// Acceptance limits for a fitted quad.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadConfig {
    /// Shortest side a quad may have, in pixels. Below roughly 8 px the 6x6
    /// payload cells are under a pixel each and cannot be sampled.
    pub min_side_px: Scalar,
    /// Largest ratio of longest to shortest side. A real tag under perspective
    /// stays well under this; a sliver is a mis-segmented contour.
    pub max_side_ratio: Scalar,
    /// Fraction of each side's points discarded at both ends before the line
    /// fit, so that corner rounding does not bend the line.
    pub corner_trim: Scalar,
    /// Minimum surviving points per side for the refinement to be used.
    pub min_points_per_side: usize,
}

impl Default for QuadConfig {
    fn default() -> Self {
        QuadConfig {
            min_side_px: 8.0,
            max_side_ratio: 8.0,
            corner_trim: 0.2,
            min_points_per_side: 4,
        }
    }
}

/// Fit a quadrilateral to a closed boundary contour.
///
/// Returns `None` when the points do not describe a convex quad of usable size.
#[must_use]
pub fn fit_quad(points: &[Vec2], config: &QuadConfig) -> Option<Quad> {
    if points.len() < 8 {
        return None;
    }
    let coarse = extreme_corners(points)?;
    let quad = refine_corners(points, &coarse, config).unwrap_or(coarse);

    if !quad
        .corners
        .iter()
        .all(|c| c.x.is_finite() && c.y.is_finite())
    {
        return None;
    }
    if !quad.is_convex() {
        return None;
    }
    if quad.min_side() < config.min_side_px {
        return None;
    }
    if quad.max_side() > config.max_side_ratio * quad.min_side() {
        return None;
    }
    Some(wind_positive(quad))
}

/// Four extreme points of a convex contour, in traversal order.
///
/// Farthest from the centroid, then farthest from that, then the two points
/// farthest from the line joining them on either side. For a square under
/// perspective these are the corners; the refinement afterwards is what makes
/// them accurate rather than merely correct.
fn extreme_corners(points: &[Vec2]) -> Option<Quad> {
    let centroid: Vec2 = points.iter().sum::<Vec2>() / points.len() as Scalar;
    let p0 = *points.iter().max_by(|a, b| {
        (*a - centroid)
            .norm_squared()
            .total_cmp(&(*b - centroid).norm_squared())
    })?;
    let p2 = *points.iter().max_by(|a, b| {
        (*a - p0)
            .norm_squared()
            .total_cmp(&(*b - p0).norm_squared())
    })?;

    let axis = p2 - p0;
    let len = axis.norm();
    if len < 1e-9 {
        return None;
    }
    let normal = Vec2::new(-axis.y, axis.x) / len;

    let mut best_pos = (0.0, p0);
    let mut best_neg = (0.0, p0);
    for p in points {
        let d = normal.dot(&(p - p0));
        if d > best_pos.0 {
            best_pos = (d, *p);
        }
        if d < best_neg.0 {
            best_neg = (d, *p);
        }
    }
    if best_pos.0 <= 1e-9 || best_neg.0 >= -1e-9 {
        return None; // degenerate: the contour is a line segment
    }
    Some(Quad {
        corners: [p0, best_pos.1, p2, best_neg.1],
    })
}

/// Refine corners by intersecting total-least-squares lines fitted to the four
/// sides.
fn refine_corners(points: &[Vec2], coarse: &Quad, config: &QuadConfig) -> Option<Quad> {
    let centroid: Vec2 = points.iter().sum::<Vec2>() / points.len() as Scalar;

    // Angular order around the centroid *is* boundary order for a convex
    // contour, which gives contiguous runs per side without tracing edges.
    let mut ordered: Vec<(Scalar, Vec2)> = points
        .iter()
        .map(|p| ((p.y - centroid.y).atan2(p.x - centroid.x), *p))
        .collect();
    ordered.sort_by(|a, b| a.0.total_cmp(&b.0));

    let corner_angles: Vec<Scalar> = coarse
        .corners
        .iter()
        .map(|c| (c.y - centroid.y).atan2(c.x - centroid.x))
        .collect();
    let mut corner_index: Vec<usize> = corner_angles
        .iter()
        .map(|&a| {
            ordered
                .iter()
                .enumerate()
                .min_by(|(_, x), (_, y)| (x.0 - a).abs().total_cmp(&(y.0 - a).abs()))
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect();
    corner_index.sort_unstable();
    corner_index.dedup();
    if corner_index.len() != 4 {
        return None;
    }

    let n = ordered.len();
    let mut lines = [Line::default(); 4];
    for (side, line) in lines.iter_mut().enumerate() {
        let start = corner_index[side];
        let end = corner_index[(side + 1) % 4];
        let count = if end > start {
            end - start
        } else {
            n - start + end
        };
        if count < 3 {
            return None;
        }
        let trim = ((count as Scalar) * config.corner_trim).floor() as usize;
        let keep = count.saturating_sub(2 * trim);
        if keep < config.min_points_per_side {
            return None;
        }
        let side_points: Vec<Vec2> = (0..keep)
            .map(|k| ordered[(start + trim + k) % n].1)
            .collect();
        *line = fit_line_tls(&side_points)?;
    }

    let mut corners = [Vec2::zeros(); 4];
    for (i, corner) in corners.iter_mut().enumerate() {
        // Corner i is where the side arriving at it meets the side leaving it.
        *corner = intersect(&lines[(i + 3) % 4], &lines[i])?;
    }
    Some(Quad { corners })
}

/// A line as a point and a unit direction.
#[derive(Debug, Clone, Copy, Default)]
struct Line {
    point: Vec2,
    direction: Vec2,
}

/// Total-least-squares line fit: the principal axis of the point scatter.
///
/// Ordinary least squares on `y = mx + c` blows up on vertical sides, which a
/// tag has two of whenever it is upright in frame.
fn fit_line_tls(points: &[Vec2]) -> Option<Line> {
    if points.len() < 2 {
        return None;
    }
    let mean: Vec2 = points.iter().sum::<Vec2>() / points.len() as Scalar;
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for p in points {
        let d = p - mean;
        sxx += d.x * d.x;
        sxy += d.x * d.y;
        syy += d.y * d.y;
    }
    // Principal eigenvector of the 2x2 scatter matrix, closed form.
    let trace = sxx + syy;
    if trace < 1e-12 {
        return None;
    }
    let diff = sxx - syy;
    let lambda = 0.5 * (trace + (diff * diff + 4.0 * sxy * sxy).sqrt());
    let dir = if (lambda - syy).abs() > (lambda - sxx).abs() {
        Vec2::new(lambda - syy, sxy)
    } else {
        Vec2::new(sxy, lambda - sxx)
    };
    let norm = dir.norm();
    if norm < 1e-12 {
        return None;
    }
    Some(Line {
        point: mean,
        direction: dir / norm,
    })
}

fn intersect(a: &Line, b: &Line) -> Option<Vec2> {
    let denom = a.direction.x * b.direction.y - a.direction.y * b.direction.x;
    // Near-parallel sides mean the segmentation split one edge in two.
    if denom.abs() < 1e-6 {
        return None;
    }
    let d = b.point - a.point;
    let t = (d.x * b.direction.y - d.y * b.direction.x) / denom;
    Some(a.point + a.direction * t)
}

fn wind_positive(quad: Quad) -> Quad {
    if quad.signed_area() >= 0.0 {
        quad
    } else {
        let c = quad.corners;
        Quad {
            corners: [c[0], c[3], c[2], c[1]],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Sample a quad's outline at sub-pixel spacing, then quantise each point
    /// to the half-integer grid the segmenter actually produces. The fit has to
    /// recover the true corners *through* that quantisation.
    fn outline(corners: [Vec2; 4], per_side: usize, quantise: bool) -> Vec<Vec2> {
        let mut pts = Vec::new();
        for i in 0..4 {
            let a = corners[i];
            let b = corners[(i + 1) % 4];
            for k in 0..per_side {
                let t = k as Scalar / per_side as Scalar;
                let p = a + (b - a) * t;
                pts.push(if quantise {
                    Vec2::new((p.x - 0.5).round() + 0.5, (p.y - 0.5).round() + 0.5)
                } else {
                    p
                });
            }
        }
        pts
    }

    #[test]
    fn recovers_an_axis_aligned_square_exactly() {
        let truth = [
            Vec2::new(10.0, 10.0),
            Vec2::new(50.0, 10.0),
            Vec2::new(50.0, 50.0),
            Vec2::new(10.0, 50.0),
        ];
        let q = fit_quad(&outline(truth, 40, false), &QuadConfig::default()).unwrap();
        for (got, want) in q.corners.iter().zip(truth.iter()) {
            assert_relative_eq!(got.x, want.x, epsilon = 1e-6);
            assert_relative_eq!(got.y, want.y, epsilon = 1e-6);
        }
    }

    /// The reason the line fit exists: quantised boundary points still yield
    /// sub-pixel corners, because dozens of samples average the grid away.
    #[test]
    fn line_refinement_beats_pixel_quantisation() {
        let truth = [
            Vec2::new(12.3, 9.7),
            Vec2::new(63.8, 14.1),
            Vec2::new(59.2, 66.4),
            Vec2::new(8.6, 61.0),
        ];
        let q = fit_quad(&outline(truth, 60, true), &QuadConfig::default()).unwrap();
        for (got, want) in q.corners.iter().zip(truth.iter()) {
            assert!(
                (got - want).norm() < 0.35,
                "corner {got:?} vs {want:?} — refinement did not beat the grid"
            );
        }
    }

    #[test]
    fn recovers_a_perspective_distorted_quad() {
        let truth = [
            Vec2::new(20.0, 30.0),
            Vec2::new(120.0, 12.0),
            Vec2::new(140.0, 95.0),
            Vec2::new(28.0, 110.0),
        ];
        let q = fit_quad(&outline(truth, 80, false), &QuadConfig::default()).unwrap();
        for (got, want) in q.corners.iter().zip(truth.iter()) {
            assert!((got - want).norm() < 1e-6, "{got:?} vs {want:?}");
        }
        assert!(q.is_convex());
        assert!(q.signed_area() > 0.0, "winding must be normalised");
    }

    /// Whatever order the contour arrives in, the fitted quad is the same set
    /// of corners — the caller must not depend on point ordering.
    #[test]
    fn corner_set_is_independent_of_the_starting_point() {
        let truth = [
            Vec2::new(15.0, 20.0),
            Vec2::new(85.0, 18.0),
            Vec2::new(88.0, 77.0),
            Vec2::new(12.0, 80.0),
        ];
        let base = fit_quad(&outline(truth, 50, false), &QuadConfig::default()).unwrap();
        for shift in [1usize, 37, 99] {
            let mut pts = outline(truth, 50, false);
            let n = pts.len();
            pts.rotate_left(shift % n);
            let q = fit_quad(&pts, &QuadConfig::default()).unwrap();
            for c in q.corners {
                assert!(
                    base.corners.iter().any(|b| (b - c).norm() < 1e-6),
                    "corner {c:?} not in the reference set"
                );
            }
        }
    }

    #[test]
    fn rejects_a_degenerate_line_of_points() {
        let pts: Vec<Vec2> = (0..40).map(|i| Vec2::new(i as Scalar, 5.0)).collect();
        assert!(fit_quad(&pts, &QuadConfig::default()).is_none());
    }

    #[test]
    fn rejects_a_quad_smaller_than_the_payload_can_be_sampled_in() {
        let truth = [
            Vec2::new(2.0, 2.0),
            Vec2::new(6.0, 2.0),
            Vec2::new(6.0, 6.0),
            Vec2::new(2.0, 6.0),
        ];
        assert!(fit_quad(&outline(truth, 12, false), &QuadConfig::default()).is_none());
    }

    #[test]
    fn rejects_a_sliver() {
        let truth = [
            Vec2::new(10.0, 10.0),
            Vec2::new(200.0, 10.0),
            Vec2::new(200.0, 19.0),
            Vec2::new(10.0, 19.0),
        ];
        let cfg = QuadConfig {
            max_side_ratio: 4.0,
            ..QuadConfig::default()
        };
        assert!(fit_quad(&outline(truth, 60, false), &cfg).is_none());
    }

    #[test]
    fn rejects_too_few_points() {
        assert!(fit_quad(&[], &QuadConfig::default()).is_none());
        assert!(fit_quad(&[Vec2::zeros(); 4], &QuadConfig::default()).is_none());
    }

    #[test]
    fn total_least_squares_handles_a_vertical_side() {
        // Ordinary least squares on y = mx + c would divide by zero here.
        let pts: Vec<Vec2> = (0..20).map(|i| Vec2::new(7.0, i as Scalar)).collect();
        let line = fit_line_tls(&pts).unwrap();
        assert_relative_eq!(line.direction.x.abs(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(line.direction.y.abs(), 1.0, epsilon = 1e-12);
        assert_relative_eq!(line.point.x, 7.0, epsilon = 1e-12);
    }

    #[test]
    fn convexity_and_area_helpers_agree_with_geometry() {
        let square = Quad {
            corners: [
                Vec2::new(0.0, 0.0),
                Vec2::new(4.0, 0.0),
                Vec2::new(4.0, 4.0),
                Vec2::new(0.0, 4.0),
            ],
        };
        assert_relative_eq!(square.signed_area(), 16.0, epsilon = 1e-12);
        assert!(square.is_convex());
        assert_relative_eq!(square.min_side(), 4.0, epsilon = 1e-12);
        assert_relative_eq!(square.centre(), Vec2::new(2.0, 2.0), epsilon = 1e-12);

        // Bow tie: sides cross, so the turn signs disagree.
        let bowtie = Quad {
            corners: [
                Vec2::new(0.0, 0.0),
                Vec2::new(4.0, 0.0),
                Vec2::new(0.0, 4.0),
                Vec2::new(4.0, 4.0),
            ],
        };
        assert!(!bowtie.is_convex());
    }
}
