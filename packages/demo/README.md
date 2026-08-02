# demo

The public demo. spec.md §8: *"A single page, no install, QR-scannable ... The
demo is also the adoption artifact. Nobody adopts a tracking library from a
README."*

```sh
pnpm --filter web-slam-demo dev     # then scan the LAN URL with a phone
pnpm --filter web-slam-demo build
```

`vite --host` is on by default because a phone cannot reach `localhost`.
`getUserMedia` needs a secure context, so on a LAN address you need HTTPS —
run vite behind a local certificate (`mkcert` works) or tunnel it.

## It runs against the stub backend

This is deliberate, and it is the point.

spec.md §3: *"The public demo is written at M0 against a stub implementation
that returns synthetic poses. No tracking, no map, canned data. The point is to
validate the API with a real consumer before any of it exists, and to make the
interface expensive to change afterwards rather than cheap to drift. If the demo
is awkward to write against the mock, the API is wrong, and M0 is when that
costs nothing."*

So **read `src/main.ts` as a review of the public API**, not just as a demo.
Every awkwardness in it is an API bug.

Switching to the real backend is a two-line change in `createBackend()`:

```ts
const wasm = await import('web-slam/wasm');
return await wasm.WasmBackend.load(await import('../pkg/wslam_wasm.js'));
```

Nothing else in the file changes. That invariance is the property the mock
exists to prove, so if a future change breaks it, the API drifted.

## What it shows

- Live camera with the landmark cloud and trajectory composited over it
- Keyframe frusta accumulating as you walk
- The **covariance ellipsoid** on the current pose (spec.md §8 — the
  differentiator, so it belongs on the public page too)
- A scale badge reporting the source *and* its 1σ uncertainty, or an honest
  "up to scale" when there is none
- Per-stage frame timings against the 16.6 ms budget
- A visible relocalization event when you cover and uncover the lens
- Loop closures **including rejections**, because the false-positive rate is a
  first-class metric

The stub scripts a tracking loss and recovery around frame 300, and proposes two
loop candidates — one accepted, one rejected. A demo that never loses tracking
would let us ship an API whose recovery path nobody had used.
