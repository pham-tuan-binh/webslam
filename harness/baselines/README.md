# baselines

The Tier-2 regression wall. spec.md §6: *"Per-sequence ATE checked into
`harness/baselines/` as data; CI fails on regression beyond tolerance."*

**Data, not code.** A human should be able to read the diff and see exactly
which number moved and by how much.

## Files

| File | Recorded from |
|---|---|
| `euroc.toml` | `datasets/euroc/*`, via `wslam-replay regress --write` |
| `tum-vi.toml` | `datasets/tum-vi/*` |

Both are empty until someone records them, and an unbaselined sequence **fails**
the regression run rather than passing quietly — a sequence with no baseline has
no regression protection at all, and silently accepting it would be the worst of
both worlds.

## What each baseline guards

A single ATE number is easy to game, so four things are recorded:

| Metric | Tolerance | Why it is here |
|---|---|---|
| `ate_rmse` | +10% | The headline. |
| `lost_fraction` | +5 points | A tracker that gives up on the hard frames has a *better* ATE. Without this, dropping half the sequence looks like an improvement. |
| `frame_ms_p99` | +60% | spec.md §6 L4 asks for the tail, not the mean. Loose because CI-runner timing is genuinely noisy; it catches order-of-magnitude regressions. |
| `map_mb_per_min` | +25% | spec.md §9 lists unbounded map memory as a tab-killing risk. |

The 10% ATE tolerance is not slack for sloppy work: RANSAC is seeded and
deterministic on one machine, but floating-point reductions are not bit-identical
across architectures, and a wall that fires on x86-vs-arm noise gets disabled
within a week.

## Re-recording

```sh
cargo xtask regen-baselines --confirm
```

`--confirm` is mandatory. **The commit message must explain what moved and why.**
A baseline that changes without explanation is not a baseline.
