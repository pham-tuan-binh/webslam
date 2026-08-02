"""Strobe rig: a second display flashing at a known frequency.

spec.md §6 ground-truth table: provides *absolute camera frame timing*, build
cost ~1 day. Point the phone at this page; the flash pattern in the recorded
frames tells you exactly when each frame was exposed, independent of anything
the browser reports.

**Incommensurate is the whole trick.** If the strobe rate divides the capture
rate, every frame sees the same phase and you learn nothing. Choose a frequency
whose ratio to the capture rate is irrational-ish — the default 71.3 Hz against
60 fps advances the phase by a different amount every frame and unwraps to an
unambiguous absolute time over a short window.

This module generates a self-contained HTML page and, separately, decodes a
recorded frame sequence back to exposure times.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

DEFAULT_FREQUENCY_HZ = 71.3


def check_incommensurate(strobe_hz: float, capture_hz: float, tolerance: float = 0.02) -> None:
    """Raise if the strobe would alias against the capture rate.

    The quantity that matters is the **phase advance per captured frame**,
    ``frac(strobe_hz / capture_hz)``. Aliasing happens whenever that lands on a
    simple rational: at 0 the phase never moves, at 1/2 it alternates between
    two values, at 1/3 it cycles through three. In every case the observed
    pattern repeats after a handful of frames and the unwrap is ambiguous.

    Testing only for near-integer ratios misses the sub-harmonics — 30 Hz
    against 60 fps has ratio 0.5, which is not near an integer but pins the
    phase to two values forever.

    Refusing is deliberate. An aliased rig does not fail loudly; it produces a
    plausible timing number that is wrong, which is the worst possible outcome
    for the layer whose entire job is to measure timing.
    """
    if strobe_hz <= 0 or capture_hz <= 0:
        raise ValueError("frequencies must be positive")

    phase_advance = (strobe_hz / capture_hz) % 1.0
    # Denominators up to 4: beyond that the pattern takes long enough to repeat
    # that a short capture still unwraps cleanly.
    simple_rationals = [n / d for d in (1, 2, 3, 4) for n in range(d + 1)]
    worst = min(simple_rationals, key=lambda p: abs(phase_advance - p))
    if abs(phase_advance - worst) < tolerance:
        suggestion = capture_hz * (round(strobe_hz / capture_hz) + 0.187)
        raise ValueError(
            f"strobe {strobe_hz:g} Hz against {capture_hz:g} fps gives a phase advance of "
            f"{phase_advance:.4f} cycles/frame, within {tolerance} of {worst:.4f} — the "
            f"pattern repeats and the unwrap is ambiguous. Try {suggestion:.2f} Hz."
        )


def generate_page(frequency_hz: float, path: Path) -> Path:
    """Write a standalone strobe page.

    Drives the flash from `requestAnimationFrame` against `performance.now()`
    rather than a CSS animation: CSS timing is compositor-scheduled and can slip
    by a frame without telling anyone, which would corrupt the very measurement
    this rig exists to make.
    """
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>web-slam strobe {frequency_hz:g} Hz</title>
<style>
  html, body {{ margin: 0; height: 100%; background: #000; overflow: hidden; }}
  #panel {{ position: fixed; inset: 0; background: #000; }}
  #hud {{ position: fixed; bottom: 1rem; left: 1rem; color: #666;
          font: 12px ui-monospace, monospace; mix-blend-mode: difference; }}
</style>
</head>
<body>
<div id="panel"></div>
<div id="hud"></div>
<script>
  // A square wave at a known frequency, plus a slow "frame marker" that flips
  // every 64 strobe periods so the decoder can unwrap absolute time rather than
  // only phase.
  const HZ = {frequency_hz!r};
  const panel = document.getElementById('panel');
  const hud = document.getElementById('hud');
  const t0 = performance.now();
  let flips = 0, last = -1;
  function tick(now) {{
    const t = (now - t0) / 1000;
    const phase = (t * HZ) % 1;
    const on = phase < 0.5;
    const marker = Math.floor(t * HZ / 64) % 2 === 1;
    // Green channel carries the strobe, red carries the coarse marker: two
    // independent signals in one frame, separable by the decoder.
    panel.style.background = `rgb(${{marker ? 255 : 0}}, ${{on ? 255 : 0}}, 0)`;
    if (on !== (last === 1)) {{ flips++; last = on ? 1 : 0; }}
    hud.textContent = `${{HZ}} Hz  t=${{t.toFixed(2)}}s  flips=${{flips}}`;
    requestAnimationFrame(tick);
  }}
  requestAnimationFrame(tick);
</script>
</body>
</html>
"""
    path.write_text(html)
    return path


def decode_exposure_times(
    green_levels: list[float],
    red_levels: list[float],
    strobe_hz: float,
    capture_hz: float,
) -> list[float]:
    """Recover absolute exposure time per frame from sampled strobe intensity.

    ``green_levels`` and ``red_levels`` are the mean intensity of the strobe
    panel in each recorded frame, normalised to [0, 1].

    The green channel gives phase within one strobe period; the red marker
    resolves which 64-period block we are in. Together they give an absolute
    time modulo 128 strobe periods, which at 71.3 Hz is 1.8 s — far longer than
    any plausible camera-IMU offset, so the answer is unambiguous.
    """
    if len(green_levels) != len(red_levels):
        raise ValueError("green and red level series must be the same length")
    period = 1.0 / strobe_hz
    block = 64.0 * period
    out: list[float] = []
    for i, (g, r) in enumerate(zip(green_levels, red_levels)):
        # Duty cycle observed over one exposure maps to phase: a frame that
        # integrates the rising half sees more light than one integrating the
        # falling half. asin gives the sub-period position.
        phase = math.asin(max(-1.0, min(1.0, 2.0 * g - 1.0))) / math.tau + 0.25
        coarse = block if r > 0.5 else 0.0
        # Nominal frame time anchors the unwrap; the strobe supplies the
        # correction, which is the quantity of interest.
        nominal = i / capture_hz
        blocks = math.floor((nominal - coarse) / (2 * block) + 0.5)
        out.append(coarse + blocks * 2 * block + phase * period)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hz", type=float, default=DEFAULT_FREQUENCY_HZ)
    parser.add_argument("--capture-hz", type=float, default=60.0)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "strobe.html")
    args = parser.parse_args()

    check_incommensurate(args.hz, args.capture_hz)
    path = generate_page(args.hz, args.out)
    print(f"wrote {path}")
    print(f"open it full-screen on a second display at {args.hz:g} Hz, point the phone at it,")
    print(f"and record while the turntable spins. Capture rate assumed {args.capture_hz:g} fps.")


if __name__ == "__main__":
    main()
