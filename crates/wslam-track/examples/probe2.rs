use wslam_core::*;
use wslam_track::init::*;
fn main() {
    let k = CameraIntrinsics::from_focal(460.0, 480, 360);
    let mut rng = DeterministicRng::new("probe", 5);
    let pts: Vec<Vec3> = (0..200)
        .map(|_| {
            Vec3::new(
                rng.uniform_range(-3.4, 3.4),
                rng.uniform_range(-2.4, 2.4),
                rng.uniform_range(2.2, 6.5),
            )
        })
        .collect();
    for sigma in [0.0f64, 0.02, 0.05, 0.1, 0.3] {
        for deg in [2.0f64, 4.5, 9.0] {
            let r = So3::exp(&Vec3::new(0.0, deg.to_radians(), 0.0));
            let inv = Se3::from_rotation(r).inverse();
            let mut n = DeterministicRng::new("noise", 3);
            let mut m = Vec::new();
            for p in &pts {
                let (Some(a), Some(b)) = (k.project(p), k.project(&inv.act(p))) else {
                    continue;
                };
                if k.contains(a, 4.0) && k.contains(b, 4.0) {
                    m.push((
                        a + Vec2::new(n.normal() * sigma, n.normal() * sigma),
                        b + Vec2::new(n.normal() * sigma, n.normal() * sigma),
                    ));
                }
            }
            let mut rng2 = DeterministicRng::new("init", 42);
            match initialize_two_view(&m, &k, &InitConfig::default(), &mut rng2) {
            Some(v) => println!("sigma {sigma} rot {deg} deg -> ACCEPTED {:?} ratio {:.3} parallax {:.2} deg, {} lm",
                v.model, v.homography_ratio, v.median_parallax_rad.to_degrees(), v.landmarks.len()),
            None => println!("sigma {sigma} rot {deg} deg -> None (correct)"),
        }
        }
    }
}
