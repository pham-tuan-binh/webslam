//! Temporary diagnostic probe: where the eight-point estimator loses accuracy.
use nalgebra::DMatrix;
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

fn null9(rows: &[[f64; 9]]) -> Mat3 {
    let m = rows.len().max(9);
    let mut a = DMatrix::<f64>::zeros(m, 9);
    for (i, row) in rows.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            a[(i, j)] = *v;
        }
    }
    let svd = a.svd(false, true);
    let v_t = svd.v_t.unwrap();
    let idx = svd
        .singular_values
        .iter()
        .enumerate()
        .min_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    Mat3::new(
        v_t[(idx, 0)],
        v_t[(idx, 1)],
        v_t[(idx, 2)],
        v_t[(idx, 3)],
        v_t[(idx, 4)],
        v_t[(idx, 5)],
        v_t[(idx, 6)],
        v_t[(idx, 7)],
        v_t[(idx, 8)],
    )
}

fn rows_of(a: &[Vec2], b: &[Vec2]) -> Vec<[f64; 9]> {
    a.iter()
        .zip(b)
        .map(|(p, q)| {
            [
                q.x * p.x,
                q.x * p.y,
                q.x,
                q.y * p.x,
                q.y * p.y,
                q.y,
                p.x,
                p.y,
                1.0,
            ]
        })
        .collect()
}

fn project_essential(e: &Mat3) -> Mat3 {
    let mut svd = e.svd(true, true);
    svd.sort_by_singular_values();
    let (u, v_t) = (svd.u.unwrap(), svd.v_t.unwrap());
    u * Mat3::from_diagonal(&Vec3::new(1.0, 1.0, 0.0)) * v_t
}

fn residuals(e: &Mat3, norm: &[(Vec2, Vec2)], focal: f64) -> (f64, usize) {
    let mut sum = 0.0;
    let mut inl = 0;
    for (x1, x2) in norm {
        let a = Vec3::new(x1.x, x1.y, 1.0);
        let b = Vec3::new(x2.x, x2.y, 1.0);
        let l2 = e * a;
        let num = b.dot(&l2);
        let d = num.abs() / (l2.x * l2.x + l2.y * l2.y).sqrt() * focal;
        sum += d;
        if d < 1.96 {
            inl += 1;
        }
    }
    (sum / norm.len() as f64, inl)
}

fn err_vs(e: &Mat3, truth: &Mat3) -> f64 {
    let a = e / e.norm();
    let b = truth / truth.norm();
    (a - b).norm().min((a + b).norm())
}

fn main() {
    let k = intrinsics();
    let kp = pinhole_only(&k);
    let focal = (kp.fx * kp.fy).sqrt();
    let (r, t) = relative();
    let truth = hat(&t.normalize()) * r.matrix();
    for s in [0u64, 4, 1, 2, 3, 5] {
        let matches = general_scene(140, 0.4, 5000 + s);
        let norm: Vec<(Vec2, Vec2)> = matches
            .iter()
            .map(|(a, b)| (kp.unproject_normalized(*a), kp.unproject_normalized(*b)))
            .collect();
        let src: Vec<Vec2> = norm.iter().map(|c| c.0).collect();
        let dst: Vec<Vec2> = norm.iter().map(|c| c.1).collect();
        let (a, t1) = hartley_normalize(&src).unwrap();
        let (b, t2) = hartley_normalize(&dst).unwrap();

        let lin_norm = null9(&rows_of(&a, &b));
        let _lin_raw = null9(&rows_of(&src, &dst));

        let rank2 = |m: &Mat3| {
            let mut svd = m.svd(true, true);
            svd.sort_by_singular_values();
            let (u, v_t) = (svd.u.unwrap(), svd.v_t.unwrap());
            let s = svd.singular_values;
            u * Mat3::from_diagonal(&Vec3::new(s[0], s[1], 0.0)) * v_t
        };
        let equalize = |m: &Mat3| {
            let mut svd = m.svd(true, true);
            svd.sort_by_singular_values();
            let (u, v_t) = (svd.u.unwrap(), svd.v_t.unwrap());
            let s = svd.singular_values;
            let a = 0.5 * (s[0] + s[1]);
            u * Mat3::from_diagonal(&Vec3::new(a, a, s[2])) * v_t
        };
        let den = t2.transpose() * lin_norm * t1;
        {
            let mut svd = den.svd(false, false);
            svd.sort_by_singular_values();
            let sv = svd.singular_values;
            println!(
                "--- seed {s} linear sv ratios s2/s1 {:.4} s3/s1 {:.4}",
                sv[1] / sv[0],
                sv[2] / sv[0]
            );
        }
        let variants: [(&str, Mat3); 7] = [
            (
                "v6 rank2-in-norm then denorm",
                t2.transpose() * rank2(&lin_norm) * t1,
            ),
            ("v1 current: denorm then project", project_essential(&den)),
            ("v2 rank2-in-norm, denorm, equalise", {
                equalize(&(t2.transpose() * rank2(&lin_norm) * t1))
            }),
            ("v3 denorm, rank2 only", rank2(&den)),
            ("v4 denorm, equalise only", equalize(&den)),
            ("v5 no projection", den),
            ("truth", truth),
        ];
        for (name, e) in &variants {
            let (m, i) = residuals(e, &norm, focal);
            println!(
                "   {name:38} mean {m:.3}px inl {i:3}  |dE| {:.5}",
                err_vs(e, &truth)
            );
        }
    }
}
