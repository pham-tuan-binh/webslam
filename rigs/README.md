# Ground-truth rigs

Instruments for validating the layers that need a truth source independent of the
thing being measured (spec.md §6).

| Rig | Provides | Layer it grades | Build cost |
|---|---|---|---|
| `turntable.py` | exact angular velocity | L1 orientation, L0 (with strobe) | ~1 day |
| `strobe.py` | absolute camera frame timing | L0 clock | ~1 day |
| `charuco.py` | per-device intrinsics | L2 focal (validation only) | hours |

## What is deliberately absent

**The robot-arm rig.** spec.md §6 calls it the highest-value item, because it is
the only source of exact metric 6-DoF and therefore the only way to measure
**L5 scale error** and **L6 NEES** against truth. It was removed as out of scope:
this repository ships a pose library, not a robotics test bench.

The consequence is worth stating plainly rather than leaving implicit:

- **Scale error %** — the headline metric, and the Campos-curve comparison — is
  currently validated only against synthetic trajectories with known scale.
- **NEES and coverage** are validated only against synthetic trials where the
  error really is drawn from the claimed covariance.

Both machineries are real and tested (`wslam_core::covariance`,
`harness/replay/src/metrics.rs`), so adding a metric truth source later is a
matter of feeding it data. Until then, treat "calibrated covariance" as a
property the estimator is *built and unit-tested* for, not one measured on a
phone. See docs/VERIFICATION.md.

## Setup

```sh
cd rigs
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[vision]'          # or just `-e .` for the dry-run paths
```

Every script runs with `--dry-run` and no hardware. That path exists so the
capture format and the downstream analysis can be exercised in CI, and so a
broken harness is discovered before a rig session rather than during one.

```sh
python turntable.py --dry-run --duration 2
python strobe.py                     # writes strobe.html
python charuco.py board --out board.png
python -m pytest -q test_rigs.py
```

## Captures

Both hardware rigs write the same directory format (`capture.py`), and every
capture carries its conditions — device, OS, browser, lighting, thermal state,
git commit. spec.md §6 requires results reported **per cell** of the device × OS
matrix and never pooled; that is only possible if the conditions travel with the
measurement.

```
captures/2026-08-01T12-00-00Z_turntable-30dps_pixel8/
  manifest.json
  groundtruth.csv
  imu.csv
```

`captures/` is gitignored.

## Calibrating the instruments themselves

- **Turntable rate.** Time 100 revolutions with a stopwatch once and hard-code the
  result. Never derive the "known" rate from the phone's gyro; that is the sensor
  under test.
- **Strobe frequency.** Must be incommensurate with the capture rate.
  `check_incommensurate()` refuses aliasing ratios rather than producing a
  confidently wrong number.
- **ChArUco board.** Print at 100% and measure a square with callipers. Printer
  scaling is the most common source of a systematically wrong ground truth.
