//! Temporary diagnostic probe for the two-view init model selection.
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
    let (r, t) = relative();
    let truth = Se3::new(r, t).inverse();
    for planar in [false, true] {
        for s in 0..6u64 {
            let matches = if planar {
                planar_scene(140, 0.4, 3000 + s)
            } else {
                general_scene(140, 0.4, 5000 + s)
            };
            let mut rng = DeterministicRng::new("init", 7000 + s);
            let kp = pinhole_only(&k);
            let hf = estimate_homography_ransac(
                &matches,
                &kp,
                1.0,
                400,
                &mut DeterministicRng::new("init", 7000 + s).fork("init-homography", 0),
            );
            let ef = estimate_essential_ransac(
                &matches,
                &kp,
                1.0,
                400,
                &mut DeterministicRng::new("init", 7000 + s).fork("init-essential", 1),
            );
            let sh = hf.as_ref().map_or(0.0, |f| f.score);
            let se = ef.as_ref().map_or(0.0, |f| f.score);
            let hi = hf.as_ref().map_or(0, |f| f.inlier_count);
            let ei = ef.as_ref().map_or(0, |f| f.inlier_count);
            match initialize_two_view(&matches, &k, &InitConfig::default(), &mut rng) {
                Some(v) => {
                    let rot = v.pose.rotation().minus(&truth.rotation()).norm();
                    let cos = v
                        .pose
                        .translation()
                        .normalize()
                        .dot(&truth.translation().normalize());
                    println!(
                        "planar={planar} s={s} {:?} ratio={:.4} SH={sh:.1}({hi}) SE={se:.1}({ei}) rot={rot:.5} cos={cos:.6} lm={}",
                        v.model, v.homography_ratio, v.landmarks.len()
                    );
                }
                None => println!("planar={planar} s={s} REFUSED SH={sh:.1}({hi}) SE={se:.1}({ei})"),
            }
        }
    }
}
