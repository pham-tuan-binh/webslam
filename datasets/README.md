# datasets

Gitignored. Fetch with:

```sh
sh datasets/fetch.sh euroc          # or: cargo xtask fetch-datasets euroc
```

## Read this before choosing a dataset

| Dataset | Usable today? | Why |
|---|---|---|
| **EuRoC MAV** | **yes** | `radial-tangential` distortion — the Brown-Conrady model `CameraIntrinsics` implements. |
| **TUM VI** | **no** | `pinhole-equidistant` (Kannala-Brandt) **fisheye**. We have no fisheye projection model. |
| **7-Scenes** | for relocalization only | RGB-D, no IMU; used for the L4 public number. |

**TUM VI is fisheye.** Its EuRoC-format export ships four `distortion_coefficients`
that look exactly like radial-tangential ones and mean something entirely
different. Loading them anyway compiles, runs, and produces an undistortion wrong
by tens of pixels at the image edge — visible as a mediocre ATE, never as an
error. `load_sensor_yaml` therefore **refuses** any sequence declaring an
unsupported `distortion_model` with non-zero coefficients, and
`a_fisheye_sequence_is_refused_rather_than_misread` pins that.

Fetch TUM VI if you like — it is the better match for phone optics and worth
having ready — but it needs a Kannala-Brandt model before it produces a number.

**Only the `room*` TUM VI sequences have full ground truth.** corridor,
magistrale, outdoors and slides carry mocap for the start and end segments only,
so an ATE over them would score a short prologue and epilogue and report the rest
as unmatched. `fetch.sh` pulls room sequences for that reason.

## The EuRoC host is unreliable

`robotics.ethz.ch` was unreachable while this was written — DNS resolves, port 80
times out. `fetch.sh` detects that and prints manual instructions instead of
hanging or dying: download the "ASL Dataset Format" zips from

  https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

drop them into `datasets/euroc/`, and re-run — the script unpacks whatever it
finds. Any subset works; the harness reports per sequence and never pools.

## What each is for

| Dataset | Used by | Why this one |
|---|---|---|
| **EuRoC MAV** | Tier 2 replay, L3 ATE, L4b loop closure | Published reference ATE numbers, so it validates the *port* as well as the algorithm (spec.md §6 L3: "Any divergence is a port bug, not an algorithm result") |
| **TUM VI** | Tier 2, once fisheye lands | Longer sequences, harder lighting, fisheye — much closer to phone optics than EuRoC's global-shutter stereo rig. 20 Hz camera, 200 Hz IMU. |
| **7-Scenes** | L4 relocalization | spec.md §6 L4: "Also run 7-Scenes for a comparable public number" |

## What they are *not* for

They are not a substitute for device testing. EuRoC has metric ground truth, but
it was recorded with a global-shutter camera on a drone with a hardware-synced
IMU — none of which resembles a phone browser. It tells you the tracker is
correct; it tells you nothing about clock jitter, rolling shutter, or thermal
throttling.

## Licences

None of this is redistributed here. Each dataset carries its own terms:

- **EuRoC MAV** — CC BY 3.0. Burri et al., *The EuRoC micro aerial vehicle
  datasets*, IJRR 2016.
- **TUM VI** — CC BY 4.0 (data), BSD-2-Clause (code). Schubert et al., *The TUM VI
  Benchmark for Evaluating Visual-Inertial Odometry*, IROS 2018.
- **7-Scenes** — Microsoft research-use terms. Shotton et al., CVPR 2013.

Cite them if you publish numbers derived from them.
