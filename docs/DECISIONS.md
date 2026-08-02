# Decisions

Records the calls made while building this repository, including the two
spec.md §10 open decisions. Each entry states the decision, the reasoning, and
what would reverse it.

---

## D1 — Required ScaleSources: `none`, `declared`, `fiducial`, `map` at M4

**spec.md §10:** *"Which ScaleSources are we required to support? ... If
`declared` is acceptable to our consumers, most of the difficulty evaporates.
If we must be fully passive, M1 becomes load-bearing and the timeline roughly
doubles. Decide before M0."*

**Decision.** M4 ships `none`, `declared`, `fiducial` and `map`. `inertial` is
research track R1 and gates nothing. `learned` is opt-in, feature-gated, and
ships no weights.

**Reasoning.** This is exactly the milestone table in §8 — M4's deliverable is
"L5 ScaleSource — `fiducial`, `declared`, `map`" and R1 is explicitly parallel.
Choosing the passive-only path would contradict the milestone plan the same
document sets out. Three of the four shipped sources are *exact* rulers
(`declared` and `fiducial` measure a known length; `map` inherits one), so the
headline scale-accuracy claim does not depend on the hardest layer.

**Assumption flagged for the owner.** This is the one place the spec asks a
question it does not answer, and the answer depends on consumers we do not have
in the repository. If a consumer requires fully passive operation — no tap, no
marker, no prior map — then R1 becomes load-bearing and this decision must be
revisited before M4, not after.

**What reverses it.** A committed consumer who cannot accept any user
interaction or scene preparation.

---

## D2 — No COOP/COEP requirement by default; threads are opt-in

**spec.md §10:** *"Do we accept the COOP/COEP header requirement? Threads make
the L4 backend comfortable but block third-party embedding, which is frequently
the entire reason a team chooses WebAR over native. Decide before M4."*

**Decision.** The default build is **single-threaded and embeddable anywhere**.
The L4 backend runs on the main thread in a time-sliced budget, with loop
closure and pose-graph optimisation running at a reduced rate. A `threads`
Cargo feature plus a separate npm entry point (`web-slam/threaded`) builds the
`wasm-bindgen-rayon` variant for consumers who control their own headers.

**Reasoning.** spec.md §9 rates the impact of the header requirement as "breaks
third-party embedding — often the whole point of WebAR", against a cost of
"single-threaded backend optimisation at reduced rate". A reduced optimisation
rate degrades a metric; a header requirement removes a deployment mode
entirely. Reversible degradation beats irreversible exclusion, and the reduced
rate is measurable against the M5 exit criterion rather than hidden.

The backend work that actually needs the thread — pose-graph optimisation — is
also the work that tolerates latency best. Relocalization, which must be fast
because the user is standing there waiting, is a BoW query plus one PnP and does
not need a thread.

**Consequence to hold ourselves to.** The frontend must never stall on the
backend even single-threaded. That is a p99 frame-time claim, and spec.md §6 L4
already requires measuring it: "Backend latency distribution, and confirmation
the frontend never stalls on it. Measure frame-time tail (p99), not mean."
Single-threaded makes this the *primary* risk of the design rather than a
footnote, so the budget is enforced in code (`MapConfig::backend_budget_ms`),
not by convention.

**What reverses it.** Measured p99 frame time exceeding budget on the target
device matrix with the backend time-sliced as small as it usefully can be.

---

## D3 — Lie groups are implemented in-house, not taken from `sophus-rs`

**spec.md §7** lists `sophus-rs` for SE(3)/SO(3).

**Decision.** `wslam-core::math` implements SO(3), SE(3) and Sim(3) directly on
`nalgebra`.

**Reasoning.** These operations are the subject of the Tier-1 test suite
(spec.md §6: *"Lie group ops against known identities"*), they total ~400 lines
including the small-angle branches, and they sit underneath every other layer.
A dependency here converts an API-churn event in someone else's crate into a
break in the code we least want to touch. The tests we would have to write to
trust a third-party implementation are the same tests we write to trust our own.

**What reverses it.** Needing the automatic-differentiation or spline machinery
`sophus-rs` provides, which we currently do not.

---

## D4 — Map serialisation is hand-rolled little-endian, not serde

**Decision.** `wslam-map::serialize` writes an explicit versioned binary format.

**Reasoning.** The format is a compatibility surface: a map anchored today must
be readable by a build shipped later, and spec.md §4 L4c makes persistence the
mechanism that turns a map into a ScaleSource. An explicit format makes the
version check and the field layout reviewable, and makes
`Error::MapVersion` a real branch rather than a deserialisation failure. It also
keeps a derive-heavy dependency out of the wasm binary, which §7 asks us to
keep small.

**What reverses it.** The format growing enough optional structure that
hand-rolling becomes the larger risk.

---

## D5 — The GPU path is optional; the CPU path is the reference

**Decision.** `wslam-track` compiles and passes its full test suite without
`wslam-gpu`. The GPU path is behind a `gpu` feature and selected at runtime by
`TrackConfig::use_gpu`.

**Reasoning.** spec.md §6 L3 requires that WASM-vs-native divergence be
attributable: *"Any divergence is a port bug, not an algorithm result."* That
attribution needs a reference implementation that is definitionally correct and
runs everywhere, including CI machines with no GPU. It also means a WebGPU
failure on a given device degrades performance rather than breaking tracking,
which matters given the iOS 26 WebGPU floor in §9.

**What reverses it.** Nothing foreseeable; the reference costs little to keep.

---

## D6 — Dual MIT/Apache-2.0 licence

**Decision.** `MIT OR Apache-2.0`, the Rust ecosystem default.

**Reasoning.** spec.md §7 gives clean-room Rust as one of three reasons for the
toolchain choice, specifically to resolve the AlvaAR GPLv3 question (§9 marks
that risk resolved). Permissive licensing is the point of having done that work.
No GPL code is read, vendored or referenced in this repository.

---

## D7 — Vocabulary is trained data, checked in via git-lfs; code is ours

**Decision.** `vocab/` holds a trained vocabulary artifact. The tree-search and
training code is written here.

**Reasoning.** spec.md §7: *"DBoW2 is small. The vocabulary is the artifact; the
code is a tree search over binary descriptors ... the trained vocabulary file is
reusable as data."* Data is not derivative of the GPL implementation that
happened to train it, but the vocabulary shipped here is trained by our own
`xtask train-vocab` on our own descriptor definition, because our descriptor is
not bit-compatible with ORB's anyway.

---

## D8 — Timestamps are integer nanoseconds

**Decision.** `Timestamp` wraps `i64` nanoseconds rather than `f64` seconds.

**Reasoning.** spec.md §6 requires replay to be bit-for-bit reproducible.
Floating-point time accumulates representation error differently depending on
the magnitude of the origin, so a session that starts at `performance.now() =
3.2e6` and one that starts at `0` would not produce identical results from
identical inputs. Integer nanoseconds remove the failure mode; the conversion to
`f64` milliseconds happens once, at the JS boundary.

---

## D9 — Debug surface is versioned separately and gated

**Decision.** `DEBUG_API_VERSION` is distinct from `PUBLIC_API_VERSION`, and the
TypeScript `slam.debug` namespace is documented as unstable.

**Reasoning.** Straight from spec.md §3: the debug surface is *"explicitly
unstable — versioned separately from the core API so the viewer can move fast
without pinning us."* Recording it as two constants makes the promise
mechanical instead of aspirational.

---

## D10 — L2 refinement is multi-start, because the `(f, k1)` landscape is bimodal

**Decision.** `wslam_calib::refine_multistart` tries nine `(focal, k1)` starts and
keeps the lowest-cost solution. `FocalEstimator` uses it; the single-start
`refine` remains as the primitive.

**Reasoning — this was measured, not anticipated.** Focal length and radial
distortion both act radially, so the joint cost surface has a second basin in
which a too-long focal is traded against a too-weak barrel coefficient.
Single-start Levenberg-Marquardt on noise-free synthetic barrel data
(`k1 = -0.28`) converged to `k1 ~= -0.149` with the focal 3% high, at a residual
of 1-7 px where the true optimum reaches 1e-4 px — and reported
`converged: true`. Which basin it found flipped unpredictably with the rotation
magnitude and the Huber threshold.

A calibration that is right or wrong depending on how far the user happened to
pan is not a calibration. The basins differ by four orders of magnitude in final
cost, so selecting on cost is unambiguous rather than a coin flip.

`single_start_refinement_lands_in_the_wrong_basin` pins the finding, and says in
its doc comment that if it ever passes trivially the multi-start has become
redundant and should be deleted rather than left as cargo cult.

**Also added:** a field-of-view plausibility gate. At `k1 = -0.35` the
refinement converged, reported success, and returned a focal implying a
26-degree horizontal field of view. A wrong number that announces itself is
recoverable; one that claims success is not.

---

## D11 — Barrel distortion *shortens* our focal estimate, opposite to the cited paper

**Finding.** spec.md §5 cites Hayman & Murray (CVIU 2004) for barrel distortion
producing "a sharply increasing **overestimate** of focal length". Measured here,
the unmodelled arm consistently **under**estimates — 4.8% at `k1 = -0.10` rising
monotonically to 8.1% at `k1 = -0.35`, reproducible across seeds.

**Why they differ, and why the citation still stands.** Hayman & Murray analyse
full self-calibration, where the rotations are solved jointly with `f`; the
distortion error is partly absorbed by the rotation estimates and the residual
bias in `f` comes out positive. Here `R` comes from L1, so the error cannot hide
in the rotation and lands entirely on `f` — and barrel distortion compresses
peripheral flow, which the pinhole model can only explain with a shorter focal
length.

The citation supports what it actually supports: distortion is a first-order
failure mode for rotation-based calibration, which is why spec.md §6 L2 makes the
ablation a gate. It does not fix the sign for our setup, and the test asserts the
measured direction with the derivation in its doc comment.

**Why this is written down.** Anyone re-reading the spec will expect an
overestimate and may "fix" the test to match the paper. That would be a
regression disguised as a correction.

---

## D12 — Second-order tolerances are written in second-order form

**Decision.** Where a quantity is provably second order in another, its test
bound is quadratic rather than absolute or relative. Two instances:

- **L1 yaw leak during a gravity correction.** The gravity update's
  unobservable direction is the *estimated* vertical, so while the estimate is
  tilted by theta the null space is misaligned by theta and the leak into world
  yaw goes as theta squared. Measured over a 7-to-42 degree sweep, `leak/tilt`
  varies 5x while `leak/tilt^2` stays inside a factor of 2. The bound is
  `LEAK_COEFFICIENT * tilt^2`, and
  `the_yaw_leak_is_second_order_in_the_tilt_removed` asserts the *scaling* — a
  first-order bug (a Jacobian that genuinely observes yaw) shows a constant
  ratio and fails there while passing any absolute threshold picked at small
  tilt.

- **L3 pose after relocalization.** The rebuilt map sits exactly one bootstrap
  baseline from the anchor, because L3 normalises `|t| = 1` — it claims no scale.
  So the only legitimate distance is 1.0 (asserted to 1e-6), and everything
  after that drifts in arbitrary units. The durable property is relative: the
  trajectory stays nearer the anchor than the origin.

**Reasoning.** An absolute bound on a second-order quantity measures the test's
own setup. Both of these originally failed not because the estimator was wrong
but because the threshold encoded a magnitude the scenario happened to produce,
and the honest fix was to write the bound in the shape the physics has.

---

## D13 — GPU tests share one device, because per-test devices deadlock

**Decision.** `wslam-gpu`'s test module holds a single `GpuContext` in a
`OnceLock`. Pipelines stay per-test; the device does not.

**Reasoning — found by running on a second machine, not by reading the code.**
Every test originally built its own `GpuContext`, so `cargo test` issued one
`request_adapter` + `request_device` per test across its thread pool. On
macOS/Metal that was merely wasteful and the suite passed in 0.33 s. On an x86_64
Linux box with an NVIDIA card it wedged: 14 tests sat past 60 s each, while the
same tests run with `--test-threads=1` finished in 3.26 s and any single test
alone finished in 0.37 s.

Measured, on the same machine and commit:

| | wall time |
|---|---|
| device per test, parallel | **hangs past 60 s** |
| device per test, `--test-threads=1` | 3.26 s |
| **shared device, parallel** | **0.27 s** |

A `wgpu::Device` is `Send + Sync` and intended to be shared, so this is also the
idiomatic shape — the original was wrong on portability *and* on cost.

**Why it matters beyond the test suite.** spec.md §6 Tier 1 must run on every
commit, and CI runners are Linux. A suite that passes on the author's laptop and
hangs in CI is worse than one that fails, because the first symptom is a timeout
with no failing assertion to read.

**What this says about the rest of the repo.** Every number in it was, until
now, measured on one machine and one GPU backend. This is the first evidence of
a platform-specific defect, and it was in the layer the spec singles out for
exactly this concern (§6 L3: "Any divergence is a port bug, not an algorithm
result"). Treat single-platform green as weaker evidence than it looks.
