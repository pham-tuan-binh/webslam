//! Temporary diagnostic probe: homography decomposition candidate ranking.
use wslam_core::*;
use wslam_track::init::*;
use wslam_track::motion_ba::pinhole_only;
use wslam_track::triangulate::{triangulate_two_view, TriangulationConfig};

fn intrinsics() -> CameraIntrinsics {
    CameraIntrinsics::from_focal(520.0, 640, 480)
}
fn relative() -> (So3, Vec3) {
    (
        So3::exp(&Vec3::new(0.03, -0.09, 0.02)),
        Vec3::new(-0.45, 0.06, 0.08),
    )
}
fn project(k: &CameraIntrinsics, p: &Vec3) -> Option<Vec2> {
    wslam_track::motion_ba::project_pinhole(&pinhole_only(k), p)
}

fn planar_scene(n: usize, sigma: f64, seed: u64) -> Vec<(Vec2, Vec2)> {
    let k = intrinsics();
    let (r, t) = relative();
    let mut rng = DeterministicRng::new("plane", seed);
    let mut out = Vec::new();
    while out.len() < n {
        let (x, y) = (rng.uniform_range(-1.6, 1.6), rng.uniform_range(-1.2, 1.2));
        let p = Vec3::new(x, y, 5.0 + 0.4 * x - 0.25 * y);
        let p2 = r.act(&p) + t;
        let (Some(a), Some(b)) = (project(&k, &p), project(&k, &p2)) else {
            continue;
        };
        if !(k.contains(a, 6.0) && k.contains(b, 6.0)) {
            continue;
        }
        out.push((
            a + Vec2::new(rng.normal() * sigma, rng.normal() * sigma),
            b + Vec2::new(rng.normal() * sigma, rng.normal() * sigma),
        ));
    }
    out
}

fn main() {
    let k = intrinsics();
    let kp = pinhole_only(&k);
    let (r, t) = relative();
    let tn = t.normalize();
    for s in [1u64, 3, 0] {
        let matches = planar_scene(140, 0.4, 3000 + s);
        let fit = estimate_homography_ransac(
            &matches,
            &kp,
            1.0,
            400,
            &mut DeterministicRng::new("init", 7000 + s).fork("init-homography", 0),
        )
        .unwrap();
        println!(
            "--- seed {s}: score {:.1} inliers {}",
            fit.score, fit.inlier_count
        );
        let cfg = TriangulationConfig {
            min_parallax_rad: 0.02_f64.to_radians(),
            max_reprojection_px: 4.0,
            pixel_sigma: 1.0,
            max_depth: 1.0e4,
        };
        for (i, d) in decompose_homography(&fit.model).into_iter().enumerate() {
            let rot = d.rotation.minus(&r).norm();
            let cos = d.translation.dot(&tn);
            let pose = Se3::new(d.rotation, d.translation).inverse();
            let mut count = 0;
            for (j, (a, b)) in matches.iter().enumerate() {
                if !fit.inliers[j] {
                    continue;
                }
                if triangulate_two_view(&Se3::identity(), *a, &pose, *b, &kp, &cfg).is_ok() {
                    count += 1;
                }
            }
            println!(
                "  cand {i}: rot_err {rot:.5} t_cos {cos:+.5} n {:?} tri {count}",
                d.plane_normal.as_slice()
            );
        }
    }
}
