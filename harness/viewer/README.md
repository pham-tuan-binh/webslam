# viewer

The dev viewer. spec.md §8: **internal, ugly, indispensable.**

Not the public demo — that is `packages/demo`, has a different audience, and
must not be conflated with this.

## Running it

```sh
# Write a session to scrub later (what CI does nightly)
cargo run -p wslam-replay --features rerun-viewer -- run datasets/euroc/MH_01_easy --rrd /tmp/run.rrd
rerun /tmp/run.rrd

# Or stream to a live viewer
rerun &
cargo run -p wslam-replay --features rerun-viewer -- run datasets/euroc/MH_01_easy
```

The feature is off by default because `rerun` is a large dependency and Tier 1
must stay under ten seconds. Without it every call becomes a no-op through
`NullSink`, so the instrumentation at the call sites cannot rot behind a
`#[cfg]`.

## What it shows

Everything spec.md §8 lists as required:

| Entity | What |
|---|---|
| `world/camera/image` | Camera feed |
| `world/camera/image/features` | Tracked features, **coloured by state** — new / tracked / outlier-rejected / lost |
| `world/landmarks` | Sparse landmark cloud |
| `world/keyframes/*` | Keyframe frusta |
| `world/camera` vs `world/ground_truth` | Estimated trajectory over truth |
| `world/camera/uncertainty` | **Covariance ellipsoid on the current pose** |
| `world/pose_graph/accepted` · `/rejected` | Pose-graph edges, **including loop candidates geometric verification rejected** |
| `status/scale` | Scale source badge and current variance |
| `status/tracking_state` | init / tracking / limited / lost / relocalizing |
| `timings/*` | Per-stage: upload / pyramid / corners / flow / pnp |
| `metrics/*` | Position error, scale σ%, uncertainty — as time series |

Two of those exist for specific reasons worth restating:

- **The covariance ellipsoid** is the differentiator, and the spec says we
  should be looking at it daily. If it is not on screen we have no daily signal
  that the L6 claim still holds.
- **Rejected loop candidates** are drawn because that is how the verification
  threshold gets tuned by eye rather than by guesswork. A graph view showing
  only accepted closures cannot answer "was the threshold too tight?".

## Two timelines

Everything is logged on both `frame` (integer sequence) and `time` (seconds in
the unified timebase). Scrub by frame when chasing a tracking failure; scrub by
time when correlating against IMU.

## Adding to it

Log through `ViewerSink`, never against `rerun` directly — that is what keeps
the no-feature build compiling and the call sites honest.
