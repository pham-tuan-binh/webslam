//! Dataset loaders.
//!
//! EuRoC, and any other dataset in the EuRoC ASL directory layout. TUM-VI
//! publishes an export in this layout but is fisheye, and `load_sensor_yaml`
//! refuses it — see that function for why.
//!
//! ```text
//! MH_01_easy/mav0/
//!   cam0/data.csv          timestamp_ns, filename
//!   cam0/data/*.png
//!   imu0/data.csv          timestamp_ns, wx, wy, wz, ax, ay, az
//!   state_groundtruth_estimate0/data.csv   timestamp_ns, px..., qw, qx, qy, qz
//!   cam0/sensor.yaml       intrinsics
//! ```
//!
//! Loading is streaming-friendly: images are read on demand rather than all at
//! once, because a EuRoC sequence is a few thousand frames and holding them all
//! is gigabytes for no benefit.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use wslam_core::{
    CameraIntrinsics, Frame, FrameId, GrayImage, ImuSample, RadialTangential, Scalar, Se3, So3,
    Timestamp, Vec3,
};

use crate::metrics::Stamped;

/// A loaded sequence, with images left on disk.
#[derive(Debug)]
pub struct Sequence {
    /// Human-readable name, e.g. `MH_01_easy`.
    pub name: String,
    /// Frame timestamps and image paths.
    pub frames: Vec<(Timestamp, PathBuf)>,
    /// Inertial samples, time-ordered.
    pub imu: Vec<ImuSample>,
    /// Ground-truth poses, time-ordered. Empty when the sequence has none.
    pub truth: Vec<Stamped>,
    /// Intrinsics from `sensor.yaml`, when parseable.
    pub intrinsics: Option<CameraIntrinsics>,
    /// Full `T_body_camera` from cam0's `T_BS`, rotation **and** translation.
    ///
    /// The translation matters for evaluation, not just the rotation. EuRoC's
    /// ground truth is `T_world_body`; we estimate `T_world_camera`. The two
    /// differ by `p_world_camera = p_world_body + R_world_body * t_body_camera`,
    /// and because that offset rotates with the body a single global Sim(3)
    /// cannot absorb it. On cam0's 6.5 cm lever arm it leaves a hard error
    /// floor of roughly 0.038 m — the same size as the segment ATEs we are
    /// trying to measure.
    pub body_from_camera_se3: Option<Se3>,
    /// `R_body_camera` from cam0's `T_BS`.
    ///
    /// EuRoC publishes the camera-IMU extrinsic and it is **not** identity —
    /// cam0 sits about 90 degrees from the body axes. Feeding L1's body-frame
    /// attitude to L2 and L3 without it is a silent frame error.
    pub body_from_camera: Option<So3>,
}

impl Sequence {
    /// Frame count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Duration in seconds.
    #[must_use]
    pub fn duration(&self) -> Scalar {
        match (self.frames.first(), self.frames.last()) {
            (Some(a), Some(b)) => b.0.since(a.0),
            _ => 0.0,
        }
    }

    /// Load one frame's image from disk.
    pub fn load_frame(&self, index: usize) -> Result<Frame> {
        let (timestamp, path) = self
            .frames
            .get(index)
            .with_context(|| format!("frame {index} out of range"))?;
        let image = image::open(path)
            .with_context(|| format!("reading {}", path.display()))?
            .to_luma8();
        let (w, h) = image.dimensions();
        Ok(Frame::new(
            FrameId(index as u64),
            *timestamp,
            GrayImage::from_vec(w, h, image.into_raw()),
        ))
    }

    /// Inertial samples in `(from, to]`.
    #[must_use]
    pub fn imu_between(&self, from: Timestamp, to: Timestamp) -> &[ImuSample] {
        let start = self.imu.partition_point(|s| s.timestamp <= from);
        let end = self.imu.partition_point(|s| s.timestamp <= to);
        &self.imu[start..end.max(start)]
    }
}

/// Load a EuRoC-format sequence.
///
/// Accepts either the sequence root or the `mav0` directory inside it, because
/// both are what people actually have after unzipping.
pub fn load_euroc(path: &Path) -> Result<Sequence> {
    let mav0 = if path.join("mav0").is_dir() {
        path.join("mav0")
    } else if path.join("cam0").is_dir() {
        path.to_path_buf()
    } else {
        bail!(
            "{} does not look like a EuRoC sequence (no mav0/ or cam0/). \
             Run `cargo xtask fetch-datasets euroc`.",
            path.display()
        );
    };

    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "sequence".to_string());

    let frames = load_frame_index(&mav0.join("cam0"))?;
    if frames.is_empty() {
        bail!("{} contains no frames", mav0.display());
    }
    let imu = load_imu(&mav0.join("imu0/data.csv")).unwrap_or_default();
    let truth = load_groundtruth(&mav0.join("state_groundtruth_estimate0/data.csv"))
        .or_else(|_| load_groundtruth(&mav0.join("mocap0/data.csv")))
        .unwrap_or_default();
    // `?` rather than `.ok()`: an unsupported camera model must stop the run.
    // Falling through to L2's estimator would quietly replace a known-wrong
    // calibration with a guessed one and still report an ATE.
    let sensor_yaml = mav0.join("cam0/sensor.yaml");
    let intrinsics = if sensor_yaml.exists() {
        Some(load_sensor_yaml(&sensor_yaml)?)
    } else {
        None
    };
    let body_from_camera_se3 = load_t_bs(&sensor_yaml).ok();
    let body_from_camera = body_from_camera_se3.map(|t| t.rotation());

    log::info!(
        "{name}: {} frames, {} imu, {} ground truth, intrinsics {}, extrinsic {}",
        frames.len(),
        imu.len(),
        truth.len(),
        if intrinsics.is_some() { "yes" } else { "no" },
        match body_from_camera {
            Some(r) => format!("{:.1} deg", r.angle().to_degrees()),
            None => "none".to_string(),
        }
    );

    Ok(Sequence {
        name,
        frames,
        imu,
        truth,
        intrinsics,
        body_from_camera,
        body_from_camera_se3,
    })
}

/// Read the full `T_body_camera` out of a EuRoC `sensor.yaml`'s `T_BS` block.
///
/// `T_BS` is the 4x4 sensor-to-body transform, written row-major across four
/// continuation lines. Only its rotation block is needed here: the translation
/// is the lever arm, which L2 models separately.
fn load_t_bs(path: &Path) -> Result<Se3> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let start = text.find("T_BS:").context("no `T_BS` key")?;
    let block = &text[start..];
    let open = block.find('[').context("no `[` after T_BS")?;
    let close = block.find(']').context("unterminated T_BS")?;
    let values: Vec<Scalar> = block[open + 1..close]
        .split(',')
        .filter_map(|v| v.trim().parse::<Scalar>().ok())
        .collect();
    if values.len() < 12 {
        bail!("T_BS has {} numbers, expected at least 12", values.len());
    }
    // Row-major 4x4: upper-left 3x3 is R_body_sensor, column 3 is t_body_sensor.
    let r = wslam_core::Mat3::new(
        values[0], values[1], values[2], values[4], values[5], values[6], values[8], values[9],
        values[10],
    );
    let t = Vec3::new(values[3], values[7], values[11]);
    Ok(Se3::new(So3::from_matrix(&r), t))
}

fn load_frame_index(cam: &Path) -> Result<Vec<(Timestamp, PathBuf)>> {
    let csv = cam.join("data.csv");
    let text = fs::read_to_string(&csv).with_context(|| format!("reading {}", csv.display()))?;
    let dir = cam.join("data");
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split(',');
        let (Some(ts), Some(file)) = (fields.next(), fields.next()) else {
            continue;
        };
        let Ok(nanos) = ts.trim().parse::<i64>() else {
            continue;
        };
        out.push((Timestamp::from_nanos(nanos), dir.join(file.trim())));
    }
    out.sort_by_key(|(t, _)| *t);
    Ok(out)
}

fn load_imu(csv: &Path) -> Result<Vec<ImuSample>> {
    let text = fs::read_to_string(csv).with_context(|| format!("reading {}", csv.display()))?;
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split(',').map(str::trim).collect();
        if f.len() < 7 {
            continue;
        }
        let parse = |s: &str| s.parse::<Scalar>().ok();
        let (Ok(nanos), Some(wx), Some(wy), Some(wz), Some(ax), Some(ay), Some(az)) = (
            f[0].parse::<i64>(),
            parse(f[1]),
            parse(f[2]),
            parse(f[3]),
            parse(f[4]),
            parse(f[5]),
            parse(f[6]),
        ) else {
            continue;
        };
        out.push(ImuSample::new(
            Timestamp::from_nanos(nanos),
            Vec3::new(wx, wy, wz),
            Vec3::new(ax, ay, az),
        ));
    }
    out.sort_by_key(|s| s.timestamp);
    Ok(out)
}

fn load_groundtruth(csv: &Path) -> Result<Vec<Stamped>> {
    let text = fs::read_to_string(csv).with_context(|| format!("reading {}", csv.display()))?;
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split(',').map(str::trim).collect();
        if f.len() < 8 {
            continue;
        }
        let parse = |s: &str| s.parse::<Scalar>().ok();
        let Ok(nanos) = f[0].parse::<i64>() else {
            continue;
        };
        let (Some(px), Some(py), Some(pz), Some(qw), Some(qx), Some(qy), Some(qz)) = (
            parse(f[1]),
            parse(f[2]),
            parse(f[3]),
            // EuRoC orders the quaternion w-first, unlike almost everything
            // else. Getting this wrong produces a plausible-looking trajectory
            // that is rotated, which is very hard to spot in an ATE number.
            parse(f[4]),
            parse(f[5]),
            parse(f[6]),
            parse(f[7]),
        ) else {
            continue;
        };
        out.push(Stamped {
            timestamp: Timestamp::from_nanos(nanos),
            pose: Se3::new(So3::from_wxyz(qw, qx, qy, qz), Vec3::new(px, py, pz)),
            // Ground truth is one continuous frame by construction.
            epoch: 0,
        });
    }
    out.sort_by_key(|s| s.timestamp);
    Ok(out)
}

/// Parse the intrinsics out of a EuRoC-format `sensor.yaml`.
///
/// A hand parse rather than a YAML dependency: we need four keys and the file is
/// machine-generated with a stable shape.
///
/// **The `distortion_model` check is the important part.** EuRoC ships
/// `radial-tangential`, which is the Brown-Conrady model `CameraIntrinsics`
/// implements. TUM-VI's EuRoC export ships `equidistant` — the Kannala-Brandt
/// fisheye model, whose four coefficients mean something completely different.
/// Reading those four numbers into a `RadialTangential` compiles, runs, and
/// produces an undistortion that is wrong by tens of pixels at the image edge,
/// which would show up as a mediocre ATE rather than as an error. Refusing is
/// the only honest option until a fisheye model exists.
fn load_sensor_yaml(path: &Path) -> Result<CameraIntrinsics> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;

    let numbers = |key: &str| -> Option<Vec<Scalar>> {
        let line = text.lines().find(|l| l.trim_start().starts_with(key))?;
        let inner = line.split('[').nth(1)?.split(']').next()?;
        Some(
            inner
                .split(',')
                .filter_map(|s| s.trim().parse::<Scalar>().ok())
                .collect(),
        )
    };
    let word = |key: &str| -> Option<String> {
        let line = text.lines().find(|l| l.trim_start().starts_with(key))?;
        Some(
            line.split(':')
                .nth(1)?
                .trim()
                .trim_matches('"')
                .to_lowercase(),
        )
    };

    let intrinsics = numbers("intrinsics").context("no `intrinsics` key")?;
    if intrinsics.len() < 4 {
        bail!("`intrinsics` has {} entries, expected 4", intrinsics.len());
    }
    let distortion = numbers("distortion_coefficients").unwrap_or_default();

    // Only reject when there are coefficients to misinterpret: a sequence
    // declaring an exotic model with all-zero coefficients is just a pinhole.
    let model = word("distortion_model").unwrap_or_else(|| "none".to_string());
    let has_distortion = distortion.iter().any(|c| c.abs() > 1e-12);
    let supported = matches!(
        model.as_str(),
        "radial-tangential" | "radtan" | "plumb_bob" | "none" | "pinhole"
    );
    if has_distortion && !supported {
        bail!(
            "{} declares distortion_model `{model}`, which this build cannot represent.\n\
             CameraIntrinsics implements Brown-Conrady (radial-tangential) only; \
             `equidistant` / Kannala-Brandt fisheye needs a different projection model.\n\
             Loading its coefficients as radial-tangential would silently produce a wrong \
             undistortion, so the sequence is refused instead.",
            path.display()
        );
    }

    let resolution = numbers("resolution").unwrap_or_else(|| vec![752.0, 480.0]);
    Ok(CameraIntrinsics {
        fx: intrinsics[0],
        fy: intrinsics[1],
        cx: intrinsics[2],
        cy: intrinsics[3],
        width: resolution[0] as u32,
        height: *resolution.get(1).unwrap_or(&480.0) as u32,
        distortion: RadialTangential {
            k1: distortion.first().copied().unwrap_or(0.0),
            k2: distortion.get(1).copied().unwrap_or(0.0),
            p1: distortion.get(2).copied().unwrap_or(0.0),
            p2: distortion.get(3).copied().unwrap_or(0.0),
            k3: 0.0,
        },
    })
}

/// Find every sequence under a dataset root.
pub fn discover(root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir() && (p.join("mav0").is_dir() || p.join("cam0").is_dir()))
        .collect();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write(path: &Path, contents: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut f = fs::File::create(path).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
    }

    #[test]
    fn euroc_quaternions_are_read_w_first() {
        // EuRoC orders the quaternion w-first. Reading it xyzw produces a
        // trajectory that is silently rotated — plausible-looking, and very
        // hard to spot downstream.
        let dir = std::env::temp_dir().join("wslam-ds-quat");
        let _ = fs::remove_dir_all(&dir);
        let csv = dir.join("gt.csv");
        // 90 degrees about Z: w = cos(45) = 0.7071, z = sin(45) = 0.7071.
        write(
            &csv,
            "#ts,px,py,pz,qw,qx,qy,qz\n1000000000,1.0,2.0,3.0,0.70710678,0.0,0.0,0.70710678\n",
        );
        let poses = load_groundtruth(&csv).unwrap();
        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].timestamp.nanos(), 1_000_000_000);
        // The rotation must take +X onto +Y.
        let rotated = poses[0].pose.rotation().act(&Vec3::new(1.0, 0.0, 0.0));
        assert!(
            (rotated - Vec3::new(0.0, 1.0, 0.0)).norm() < 1e-6,
            "{rotated:?}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn sensor_yaml_intrinsics_are_parsed() {
        let dir = std::env::temp_dir().join("wslam-ds-yaml");
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("sensor.yaml");
        write(
            &path,
            "camera_model: pinhole\n\
             intrinsics: [458.654, 457.296, 367.215, 248.375]\n\
             distortion_coefficients: [-0.283408, 0.0739591, 0.00019359, 1.76187e-05]\n\
             resolution: [752, 480]\n",
        );
        let k = load_sensor_yaml(&path).unwrap();
        assert!((k.fx - 458.654).abs() < 1e-6);
        assert!((k.cy - 248.375).abs() < 1e-6);
        assert_eq!((k.width, k.height), (752, 480));
        // A real EuRoC lens is barrel-distorted; the sign must survive.
        assert!(k.distortion.k1 < 0.0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_camera_imu_extrinsic_is_read_and_is_not_identity() {
        // EuRoC cam0 sits ~90 degrees from the body axes. Assuming identity
        // here handed L1's body-frame attitude to L2 and L3 unchanged, which
        // produced a -53.7% focal error on real imagery and silently degraded
        // the flow prediction.
        let dir = std::env::temp_dir().join("wslam-ds-tbs");
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("sensor.yaml");
        write(
            &path,
            "sensor_type: camera\n\
             T_BS:\n\
             \x20 cols: 4\n\
             \x20 rows: 4\n\
             \x20 data: [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,\n\
             \x20        0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,\n\
             \x20       -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,\n\
             \x20        0.0, 0.0, 0.0, 1.0]\n\
             intrinsics: [458.654, 457.296, 367.215, 248.375]\n",
        );
        let r = load_t_bs(&path).expect("T_BS parses").rotation();
        let degrees = r.angle().to_degrees();
        assert!(
            (80.0..100.0).contains(&degrees),
            "EuRoC cam0 should sit near 90 deg from the body frame, got {degrees:.1}"
        );
        // And it must be a real rotation, not a projected mess.
        assert!((r.matrix().determinant() - 1.0).abs() < 1e-9);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_extrinsic_is_absent_rather_than_wrongly_identity() {
        let dir = std::env::temp_dir().join("wslam-ds-notbs");
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("sensor.yaml");
        write(&path, "intrinsics: [400.0, 400.0, 320.0, 240.0]\n");
        assert!(load_t_bs(&path).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_fisheye_sequence_is_refused_rather_than_misread() {
        // TUM-VI's EuRoC export declares `equidistant` (Kannala-Brandt). Its four
        // coefficients look exactly like radial-tangential ones and are not.
        // Reading them anyway produces a wrong undistortion that shows up as a
        // mediocre ATE, never as an error — the precise failure mode this
        // project exists to refuse.
        let dir = std::env::temp_dir().join("wslam-ds-fisheye");
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("sensor.yaml");
        write(
            &path,
            "camera_model: pinhole\n\
             intrinsics: [190.978, 190.973, 254.932, 256.897]\n\
             distortion_model: equidistant\n\
             distortion_coefficients: [0.00348, 0.000715, -0.00205, 0.000202]\n\
             resolution: [512, 512]\n",
        );
        let err = load_sensor_yaml(&path).unwrap_err().to_string();
        assert!(err.contains("equidistant"), "{err}");
        assert!(err.contains("Brown-Conrady"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_exotic_model_with_zero_coefficients_is_still_a_pinhole() {
        // Refusing here would reject a perfectly usable sequence: there are no
        // coefficients to misinterpret.
        let dir = std::env::temp_dir().join("wslam-ds-zero");
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("sensor.yaml");
        write(
            &path,
            "intrinsics: [400.0, 400.0, 320.0, 240.0]\n\
             distortion_model: equidistant\n\
             distortion_coefficients: [0.0, 0.0, 0.0, 0.0]\n\
             resolution: [640, 480]\n",
        );
        let k = load_sensor_yaml(&path).expect("zero coefficients are harmless");
        assert!(k.distortion.is_identity());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn euroc_radial_tangential_is_accepted() {
        let dir = std::env::temp_dir().join("wslam-ds-radtan");
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("sensor.yaml");
        write(
            &path,
            "intrinsics: [458.654, 457.296, 367.215, 248.375]\n\
             distortion_model: radial-tangential\n\
             distortion_coefficients: [-0.283408, 0.0739591, 0.00019359, 1.76187e-05]\n\
             resolution: [752, 480]\n",
        );
        let k = load_sensor_yaml(&path).expect("EuRoC is the supported model");
        assert!(k.distortion.k1 < 0.0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn malformed_rows_are_skipped_not_fatal() {
        // Datasets in the wild have trailing blank lines and comment headers.
        let dir = std::env::temp_dir().join("wslam-ds-bad");
        let _ = fs::remove_dir_all(&dir);
        let csv = dir.join("imu.csv");
        write(
            &csv,
            "#comment\n\
             1000,0.1,0.2,0.3,0.0,0.0,9.8\n\
             garbage,,,,,,\n\
             2000,0.1,0.2,0.3,0.0,0.0,9.8\n\
             \n",
        );
        let imu = load_imu(&csv).unwrap();
        assert_eq!(imu.len(), 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_sequence_gives_an_actionable_error() {
        let err = load_euroc(Path::new("/nonexistent/sequence")).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("EuRoC"), "{text}");
        assert!(text.contains("fetch-datasets"), "{text}");
    }

    #[test]
    fn discover_returns_nothing_for_a_missing_root() {
        assert!(discover(Path::new("/nonexistent/root")).is_empty());
    }

    #[test]
    fn imu_between_is_exclusive_at_the_lower_bound() {
        let seq = Sequence {
            name: "t".into(),
            frames: Vec::new(),
            imu: (0..10)
                .map(|i| {
                    ImuSample::new(
                        Timestamp::from_seconds(i as Scalar),
                        Vec3::zeros(),
                        Vec3::zeros(),
                    )
                })
                .collect(),
            truth: Vec::new(),
            intrinsics: None,
            body_from_camera: None,
            body_from_camera_se3: None,
        };
        let slice = seq.imu_between(Timestamp::from_seconds(2.0), Timestamp::from_seconds(5.0));
        assert_eq!(slice.len(), 3); // 3, 4, 5
        assert_eq!(slice[0].timestamp.seconds() as i64, 3);
    }
}
