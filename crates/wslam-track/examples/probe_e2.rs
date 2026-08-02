//! Temporary diagnostic probe: eight-point estimator quality on clean data.
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

fn residuals(e: &Mat3, norm: &[(Vec2, Vec2)], focal: f64) -> (f64, f64, usize) {
    let mut worst: f64 = 0.0;
    let mut sum = 0.0;
    let mut inl = 0;
    for (x1, x2) in norm {
        let a = Vec3::new(x1.x, x1.y, 1.0);
        let b = Vec3::new(x2.x, x2.y, 1.0);
        let l2 = e * a;
        let num = b.dot(&l2);
        let d = num.abs() / (l2.x * l2.x + l2.y * l2.y).sqrt() * focal;
        worst = worst.max(d);
        sum += d;
        if d < 1.96 {
            inl += 1;
        }
    }
    (sum / norm.len() as f64, worst, inl)
}

fn main() {
    let k = intrinsics();
    let kp = pinhole_only(&k);
    let focal = (kp.fx * kp.fy).sqrt();
    let (r, t) = relative();
    let truth = hat(&t.normalize()) * r.matrix();
    for s in [0u64, 4] {
        for sigma in [0.0, 0.4] {
            let matches = general_scene(140, sigma, 5000 + s);
            let norm: Vec<(Vec2, Vec2)> = matches
                .iter()
                .map(|(a, b)| (kp.unproject_normalized(*a), kp.unproject_normalized(*b)))
                .collect();
            let (m, w, i) = residuals(&truth, &norm, focal);
            println!("seed {s} sigma {sigma}: truth mean {m:.3}px worst {w:.3}px inl {i}");
            let e = estimate_essential_eight_point(&norm).unwrap();
            let (m, w, i) = residuals(&e, &norm, focal);
            println!("   8pt(all 140):     mean {m:.3}px worst {w:.3}px inl {i}");
            let mut svd = e.svd(false, false);
            svd.sort_by_singular_values();
            println!("      sv {:?}", svd.singular_values.as_slice());
            // How far is the fitted E from the truth, up to sign/scale?
            let a = e / e.norm();
            let b = truth / truth.norm();
            let d = (a - b).norm().min((a + b).norm());
            println!("      |E - Etruth| = {d:.5}");
        }
    }
}
