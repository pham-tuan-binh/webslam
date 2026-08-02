//! Temporary diagnostic probe: per-frame tracker state on the synthetic scene.
use wslam_core::*;
use wslam_track::*;

const WIDTH: u32 = 480;
const HEIGHT: u32 = 360;
const FOCAL: f64 = 460.0;

fn intrinsics() -> CameraIntrinsics {
    CameraIntrinsics::from_focal(FOCAL, WIDTH, HEIGHT)
}

#[derive(Debug, Clone, Copy)]
struct Landmark {
    position: Vec3,
    amplitude: f64,
    radius: f64,
}

fn scene(count: usize, seed: u64) -> Vec<Landmark> {
    let mut rng = DeterministicRng::new("test-scene", seed);
    (0..count)
        .map(|_| Landmark {
            position: Vec3::new(
                rng.uniform_range(-3.4, 3.4),
                rng.uniform_range(-2.4, 2.4),
                rng.uniform_range(2.2, 6.5),
            ),
            amplitude: rng.uniform_range(70.0, 140.0),
            radius: rng.uniform_range(2.2, 3.2),
        })
        .collect()
}

fn render(scene: &[Landmark], pose: &Se3, k: &CameraIntrinsics, gain: f64) -> GrayImage {
    let (w, h) = (k.width as usize, k.height as usize);
    let mut buf = vec![0f64; w * h];
    for (i, v) in buf.iter_mut().enumerate() {
        let (x, y) = ((i % w) as f64, (i / w) as f64);
        *v = 96.0 + 6.0 * (0.013 * x + 0.009 * y).sin() + 4.0 * (0.021 * y).cos();
    }
    let inv = pose.inverse();
    for lm in scene {
        let cam = inv.act(&lm.position);
        let Some(px) = k.project(&cam) else { continue };
        let r = lm.radius;
        let (x0, x1) = ((px.x - r).floor() as i64, (px.x + r).ceil() as i64);
        let (y0, y1) = ((px.y - r).floor() as i64, (px.y + r).ceil() as i64);
        for y in y0.max(0)..=y1.min(h as i64 - 1) {
            for x in x0.max(0)..=x1.min(w as i64 - 1) {
                let d2 = (x as f64 - px.x).powi(2) + (y as f64 - px.y).powi(2);
                let t = 1.0 - d2 / (r * r);
                if t > 0.0 {
                    buf[y as usize * w + x as usize] += lm.amplitude * t * t;
                }
            }
        }
    }
    let mut img = GrayImage::new(k.width, k.height);
    for (dst, v) in img.data_mut().iter_mut().zip(buf) {
        *dst = (v * gain).round().clamp(0.0, 255.0) as u8;
    }
    img
}

fn truth_pose(i: usize) -> Se3 {
    let t = i as f64;
    Se3::new(
        So3::exp(&Vec3::new(
            0.004 * (0.19 * t).sin(),
            0.020 * (0.11 * t).sin(),
            0.003 * (0.07 * t).cos(),
        )),
        Vec3::new(0.055 * t, 0.010 * (0.13 * t).sin(), 0.012 * t),
    )
}

fn frame(i: usize, image: GrayImage) -> Frame {
    Frame::new(
        FrameId(i as u64),
        Timestamp::from_seconds(i as f64 / 30.0),
        image,
    )
}

fn main() {
    let k = intrinsics();
    let sc = scene(180, 7);
    let cfg = TrackConfig::default();
    let mut tracker = Tracker::new(cfg, k);
    for i in 0..40 {
        let pose = truth_pose(i);
        let out = tracker.process(&frame(i, render(&sc, &pose, &k, 1.0)), None);
        let with_lm = tracker
            .features()
            .iter()
            .filter(|f| f.landmark.is_some())
            .count();
        println!(
            "f{i:3} {:>28} tracked {:3} inl {:3} feat {:3} with_lm {:3} map {:3} kf {}",
            format!("{:?}", out.state),
            out.tracked_count,
            out.inlier_count,
            tracker.features().len(),
            with_lm,
            tracker.local_map().len(),
            out.is_keyframe
        );
    }
}
