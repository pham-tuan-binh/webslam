# Contributing

Read [`spec.md`](./spec.md) first. Most of the surprising decisions in this
repository are load-bearing, and the spec says why. If a change contradicts the
spec, the change might still be right — but say so explicitly and update
[`docs/DECISIONS.md`](./docs/DECISIONS.md) rather than letting the two drift.

## Before you push

```sh
cargo xtask ci
```

That runs exactly what CI runs, in CI's order: fmt, clippy with `-D warnings`,
the workspace tests, the wasm target check, and the TypeScript suite.

## The four rules that are not style preferences

These are enforced mechanically, and a PR that trips one is not a near miss.

### 1. No wall clock on the estimation path

`Instant::now`, `SystemTime::now`, `Date.now`, `performance.now` — none of them,
anywhere below the shim. Every timestamp enters through
`wslam_core::TimeBase`. `HostClock` is a deliberately separate trait so that
"does this code read the clock?" is answerable with one grep, and CI greps.

Profiling is the single exception, and it is confined to `StageTimings`. A
`HostClock` that influences an estimate is a bug.

*Why:* a pipeline that reads wall-clock time cannot be regression-tested at all.

### 2. Every RNG is seeded, RANSAC included

`wslam_core::DeterministicRng` is the only randomness source, and it has no
entropy constructor. The seed is logged at construction so a red CI run reports
the seed that produced it.

CI runs the whole suite twice and diffs. One green run hides an unseeded RNG.

### 3. The shim stays stupid

`packages/web-slam/src/shim.ts` stamps arrival and forwards bytes. It may not
average, filter, reorder, deduplicate, or substitute a fabricated timestamp for
a missing one.

*Why:* L0 exists to measure event-loop jitter. A shim that helpfully cleans up
its inputs destroys the signal it is supposed to deliver.

### 4. Layer boundaries are crate boundaries

`wslam-track` cannot reach into `wslam-map`. If you find yourself wanting to,
the design is wrong, and the build will tell you before review does. The
dependency direction is in [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md).

## Tests

Tier 1 is the deliverable, not an afterthought. A test earns its place by
**recovering a known answer from synthetic input** — not by re-running the
implementation and asserting it did what it did.

Specifically:

- Prefer closed-form ground truth you generated yourself.
- Verify every analytic Jacobian against central finite differences.
- Property-test round-trips: `exp`/`log`, serialise/deserialise, pack/unpack.
- Write a named test for each degenerate case the spec calls out. When a test
  exists because the literature predicts a failure, name it after the finding —
  e.g. `barrel_distortion_overestimates_focal_without_the_model`.
- When a feature is supposed to earn its keep, test the comparison. "KLT with a
  pyramid succeeds and without one fails" is worth more than "KLT succeeds".

Tier 1 must stay under ~10 s for the whole workspace. If it creeps, the test
belongs in Tier 2.

## Baselines

`harness/baselines/` is the Tier-2 regression wall. `cargo xtask
regen-baselines` requires `--confirm` and the commit message must explain what
moved and why. A baseline that changes silently is not a baseline.

## Comments

Comment **why**, not what. Cite the spec section when a choice traces to one.
`crates/wslam-core/src/camera.rs` is the house style.

Do not write comments that restate the code, and do not leave a comment
explaining a workaround without saying what it works around.

## The public API

`packages/web-slam/src/types.ts` is the product. It was frozen at M0 against a
mock precisely so it would be expensive to change, and
`packages/web-slam/test/api.test.ts` asserts the properties the spec promises
about it.

The `debug` namespace is different: explicitly unstable, versioned separately
via `DEBUG_API_VERSION`, and free to move fast. That separation exists so the
viewer never has to pin us.

## Adding a ScaleSource

The bar is stated in spec.md §1: *"The library never silently guesses scale."*
A new source must:

1. Report an honest variance, including its own systematic error — not just its
   measurement noise.
2. Return `None` rather than a confident wrong answer when its assumptions do
   not hold. `InertialScale` returning nothing during a static hold is the
   reference behaviour.
3. Ship a test that drives it into its own degenerate case and asserts it
   declines.
4. Document its cost in the same terms as the table in spec.md §2: what the
   caller gives up to get it.
