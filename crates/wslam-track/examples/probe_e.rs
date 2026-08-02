//! Temporary diagnostic probe: essential-matrix RANSAC on the general scene.
use wslam_core::math::hat;
use wslam_core::*;
use wslam_track::init::*;
use wslam_track::motion_ba::pinhole_only;

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

fn general_scene(n: usize, sigma: f64, seed: u64) -> Vec<(Vec2, Vec2)> {
    let k = intrinsics();
    let (r, t) = relative();
    let mut rng = DeterministicRng::new("scene", seed);
    let mut out = Vec::new();
    while out.len() < n {
        let p = Vec3::new(
            rng.uniform_range(-1.6, 1.6),
            rng.uniform_range(-1.2, 1.2),
            rng.uniform_range(2.5, 8.0),
        );
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

/// Same gate the crate uses, replicated so the probe can see it.
fn score_e(e: &Mat3, norm: &[(Vec2, Vec2)], focal: f64, sigma: f64) -> (f64, Vec<bool>) {
    let inv_s2 = 1.0 / (sigma * sigma);
    let mut score = 0.0;
    let mut inl = vec![false; norm.len()];
    for (i, (x1, x2)) in norm.iter().enumerate() {
        let a = Vec3::new(x1.x, x1.y, 1.0);
        let b = Vec3::new(x2.x, x2.y, 1.0);
        let l2 = e * a;
        let l1 = e.transpose() * b;
        let num = b.dot(&l2);
        let d2 = l2.x * l2.x + l2.y * l2.y;
        let d1 = l1.x * l1.x + l1.y * l1.y;
        if d1 <= 1e-24 || d2 <= 1e-24 {
            continue;
        }
        let cf = num * num / d2 * focal * focal * inv_s2;
        let cb = num * num / d1 * focal * focal * inv_s2;
        if cf > 3.841 || cb > 3.841 {
            continue;
        }
        inl[i] = true;
        score += (5.991 - cf) + (5.991 - cb);
    }
    (score, inl)
}

fn main() {
    let k = intrinsics();
    let kp = pinhole_only(&k);
    let focal = (kp.fx * kp.fy).sqrt();
    let (r, t) = relative();
    let truth = hat(&t.normalize()) * r.matrix();
    for s in [0u64, 4] {
        let matches = general_scene(140, 0.4, 5000 + s);
        let norm: Vec<(Vec2, Vec2)> = matches
            .iter()
            .map(|(a, b)| (kp.unproject_normalized(*a), kp.unproject_normalized(*b)))
            .collect();
        let (ts, ti) = score_e(&truth, &norm, focal, 1.0);
        println!(
            "seed {s}: TRUTH score {ts:.1} inliers {}",
            ti.iter().filter(|b| **b).count()
        );
        let fit = estimate_essential_ransac(
            &matches,
            &kp,
            1.0,
            400,
            &mut DeterministicRng::new("init", 7000 + s).fork("init-essential", 1),
        )
        .unwrap();
        println!(
            "  ransac score {:.1} inliers {}",
            fit.score, fit.inlier_count
        );
        // Manual LO from the ransac consensus.
        let mut inl = fit.inliers.clone();
        for round in 0..10 {
            let cons: Vec<(Vec2, Vec2)> = norm
                .iter()
                .zip(&inl)
                .filter_map(|(m, &ok)| ok.then_some(*m))
                .collect();
            let Some(e) = estimate_essential_eight_point(&cons) else {
                break;
            };
            let (sc, ni) = score_e(&e, &norm, focal, 1.0);
            println!(
                "  LO round {round}: score {sc:.1} inliers {}",
                ni.iter().filter(|b| **b).count()
            );
            if ni == inl {
                break;
            }
            inl = ni;
        }
        // Also: seed LO from the truth's inliers to see if the global optimum is
        // reachable at all.
        let cons: Vec<(Vec2, Vec2)> = norm
            .iter()
            .zip(&ti)
            .filter_map(|(m, &ok)| ok.then_some(*m))
            .collect();
        if let Some(e) = estimate_essential_eight_point(&cons) {
            let (sc, ni) = score_e(&e, &norm, focal, 1.0);
            println!(
                "  refit-from-truth: score {sc:.1} inliers {}",
                ni.iter().filter(|b| **b).count()
            );
        }
    }
}
