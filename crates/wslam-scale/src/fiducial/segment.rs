//! Connected components and boundary clustering — stage two of the detector.
//!
//! Union-find over the binarised image labels every maximal black and white
//! region. The quads we want are not the regions themselves but the *interfaces
//! between them*: every pair of 4-adjacent pixels with opposite labels is a
//! point on some region boundary, and grouping those points by the pair of
//! regions they separate isolates each closed contour on its own. AprilTag 3
//! calls these gradient clusters, and the important property is that the outer
//! edge of a tag's black border against the surrounding white quiet zone lands
//! in exactly one cluster, uncontaminated by the tag's internal cell edges.

use super::threshold::{Binary, BLACK, SKIP};
use std::collections::BTreeMap;
use wslam_core::Vec2;

/// Disjoint-set forest with union by size and path halving.
#[derive(Debug, Clone)]
pub struct UnionFind {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl UnionFind {
    /// `n` singleton sets.
    #[must_use]
    pub fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n as u32).collect(),
            size: vec![1; n],
        }
    }

    /// Representative of `i`'s set.
    pub fn find(&mut self, mut i: u32) -> u32 {
        while self.parent[i as usize] != i {
            // Path halving: one pointer update per step, no second pass.
            let grand = self.parent[self.parent[i as usize] as usize];
            self.parent[i as usize] = grand;
            i = grand;
        }
        i
    }

    /// Merge the sets containing `a` and `b`.
    pub fn union(&mut self, a: u32, b: u32) {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.size[ra as usize] < self.size[rb as usize] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb as usize] = ra;
        self.size[ra as usize] += self.size[rb as usize];
    }

    /// Cardinality of `i`'s set.
    pub fn set_size(&mut self, i: u32) -> u32 {
        let r = self.find(i);
        self.size[r as usize]
    }
}

/// Boundary points separating one black region from one white region.
#[derive(Debug, Clone)]
pub struct Cluster {
    /// `(black region root, white region root)` — the interface identity.
    pub key: (u32, u32),
    /// Sub-pixel points on the interface, in image coordinates.
    pub points: Vec<Vec2>,
}

/// Label maximal 4-connected runs of equal binarisation.
///
/// [`SKIP`] pixels join nothing: a flat region is not evidence of a component,
/// and letting it merge two genuinely distinct regions would destroy the
/// interface identity the clusters depend on.
#[must_use]
pub fn connected_components(binary: &Binary) -> UnionFind {
    let (w, h) = (binary.width(), binary.height());
    let mut uf = UnionFind::new((w as usize) * (h as usize));
    for y in 0..h {
        for x in 0..w {
            let v = binary.at(x, y);
            if v == SKIP {
                continue;
            }
            let i = y * w + x;
            if x + 1 < w && binary.at(x + 1, y) == v {
                uf.union(i, i + 1);
            }
            if y + 1 < h && binary.at(x, y + 1) == v {
                uf.union(i, i + w);
            }
        }
    }
    uf
}

/// Group black/white interface points by the pair of regions they separate.
///
/// Clusters smaller than `min_points` are dropped: a quad with sides of even a
/// few pixels contributes dozens of interface points, so anything below that is
/// texture or sensor noise. Output is ordered by region key, so the detector is
/// deterministic regardless of hash iteration order (spec.md §6).
#[must_use]
pub fn gradient_clusters(binary: &Binary, uf: &mut UnionFind, min_points: usize) -> Vec<Cluster> {
    let (w, h) = (binary.width(), binary.height());
    let mut groups: BTreeMap<(u32, u32), Vec<Vec2>> = BTreeMap::new();

    for y in 0..h {
        for x in 0..w {
            let v = binary.at(x, y);
            if v == SKIP {
                continue;
            }
            for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                if nx >= w || ny >= h {
                    continue;
                }
                let n = binary.at(nx, ny);
                if n == SKIP || n == v {
                    continue;
                }
                let (black, white) = if v == BLACK {
                    (y * w + x, ny * w + nx)
                } else {
                    (ny * w + nx, y * w + x)
                };
                let key = (uf.find(black), uf.find(white));
                let mid = Vec2::new(0.5 * (x as f64 + nx as f64), 0.5 * (y as f64 + ny as f64));
                groups.entry(key).or_default().push(mid);
            }
        }
    }

    groups
        .into_iter()
        .filter(|(_, pts)| pts.len() >= min_points)
        .map(|(key, points)| Cluster { key, points })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::threshold::adaptive_threshold;
    use super::*;
    use wslam_core::GrayImage;

    fn filled_square(w: u32, h: u32, x0: u32, y0: u32, side: u32) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        img.data_mut().fill(230);
        for y in y0..(y0 + side) {
            for x in x0..(x0 + side) {
                img.data_mut()[(y * w + x) as usize] = 25;
            }
        }
        img
    }

    #[test]
    fn union_find_merges_and_counts() {
        let mut uf = UnionFind::new(6);
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(4, 5);
        assert_eq!(uf.find(0), uf.find(2));
        assert_ne!(uf.find(0), uf.find(3));
        assert_eq!(uf.set_size(2), 3);
        assert_eq!(uf.set_size(3), 1);
        assert_eq!(uf.set_size(5), 2);
    }

    #[test]
    fn union_find_is_idempotent() {
        let mut uf = UnionFind::new(4);
        uf.union(0, 1);
        uf.union(0, 1);
        uf.union(1, 0);
        assert_eq!(uf.set_size(0), 2);
    }

    #[test]
    fn a_single_black_square_yields_one_cluster_of_its_perimeter() {
        let img = filled_square(48, 48, 12, 12, 20);
        let b = adaptive_threshold(&img, 4, 25);
        let mut uf = connected_components(&b);
        let clusters = gradient_clusters(&b, &mut uf, 20);
        assert_eq!(
            clusters.len(),
            1,
            "one interface: square against background"
        );

        // Perimeter of a 20 px square is 4 * 20 interface crossings.
        let n = clusters[0].points.len();
        assert!(
            (72..=88).contains(&n),
            "expected ~80 boundary points, got {n}"
        );

        // Every point sits on the square's outline.
        for p in &clusters[0].points {
            let on_x = (p.x - 11.5).abs() < 0.01 || (p.x - 31.5).abs() < 0.01;
            let on_y = (p.y - 11.5).abs() < 0.01 || (p.y - 31.5).abs() < 0.01;
            assert!(on_x || on_y, "stray boundary point {p:?}");
        }
    }

    #[test]
    fn two_separated_squares_yield_two_clusters() {
        let mut img = filled_square(64, 32, 4, 6, 16);
        for y in 6..22u32 {
            for x in 40..56u32 {
                img.data_mut()[(y * 64 + x) as usize] = 25;
            }
        }
        let b = adaptive_threshold(&img, 4, 25);
        let mut uf = connected_components(&b);
        let clusters = gradient_clusters(&b, &mut uf, 20);
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn small_clusters_are_dropped() {
        // A 2x2 speck contributes 8 interface points; the threshold rejects it.
        let img = filled_square(32, 32, 15, 15, 2);
        let b = adaptive_threshold(&img, 4, 25);
        let mut uf = connected_components(&b);
        assert!(gradient_clusters(&b, &mut uf, 20).is_empty());
        assert_eq!(gradient_clusters(&b, &mut uf, 4).len(), 1);
    }

    #[test]
    fn clustering_is_deterministic() {
        let img = filled_square(48, 48, 10, 8, 24);
        let b = adaptive_threshold(&img, 4, 25);
        let first: Vec<(u32, u32)> = {
            let mut uf = connected_components(&b);
            gradient_clusters(&b, &mut uf, 20)
                .iter()
                .map(|c| c.key)
                .collect()
        };
        for _ in 0..5 {
            let mut uf = connected_components(&b);
            let keys: Vec<(u32, u32)> = gradient_clusters(&b, &mut uf, 20)
                .iter()
                .map(|c| c.key)
                .collect();
            assert_eq!(keys, first);
        }
    }

    #[test]
    fn a_blank_image_yields_no_clusters() {
        let mut img = GrayImage::new(32, 32);
        img.data_mut().fill(128);
        let b = adaptive_threshold(&img, 4, 25);
        let mut uf = connected_components(&b);
        assert!(gradient_clusters(&b, &mut uf, 20).is_empty());
    }
}
