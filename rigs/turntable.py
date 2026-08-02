"""Turntable rig: phone on a motor at a programmable constant rate.

spec.md §6 ground-truth table: provides *exact angular velocity*, build cost
~1 day. It is the ground truth for L1 (integrated angle error over 60 s) and,
paired with the strobe, for L0 (cross-correlating gyro rate against
image-derived rotation rate).

A constant rate is the whole point. Any stepper driven open-loop at a fixed
step interval gives you one; what you must not do is derive the "known" rate
from the phone's own gyro, which is the sensor under test.

Two drivers ship here: a GRBL-compatible serial controller (any 3D-printer
board), and a dry run. Add your own by satisfying `TurntableDriver`.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from capture import CaptureWriter, Conditions, DeviceInfo, PoseSample


class TurntableDriver(Protocol):
    """Set a constant angular rate, and report the commanded angle."""

    def set_rate(self, deg_per_s: float) -> None: ...

    def angle_deg(self, elapsed_s: float) -> float: ...

    def stop(self) -> None: ...


@dataclass
class DryRunTurntable:
    """Perfect constant rate. Exercises the capture path with no hardware."""

    rate: float = 0.0

    def set_rate(self, deg_per_s: float) -> None:
        self.rate = deg_per_s

    def angle_deg(self, elapsed_s: float) -> float:
        return self.rate * elapsed_s

    def stop(self) -> None:
        self.rate = 0.0


class GrblTurntable:
    """A GRBL board driving a stepper through a belt reduction.

    The rate is commanded, not measured: GRBL executes a feed rate open-loop and
    a stepper either keeps up or stalls audibly. Verify with a stopwatch over
    100 revolutions once, then trust it — that calibration is more accurate than
    anything we could measure per-run.
    """

    def __init__(self, port: str, steps_per_rev: float, baud: int = 115200) -> None:
        try:
            import serial  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - hardware path
            raise SystemExit("pyserial is required for the GRBL driver: pip install pyserial") from exc
        self._serial = serial.Serial(port, baud, timeout=1.0)
        self._steps_per_rev = steps_per_rev
        self._rate = 0.0
        self._t0 = time.perf_counter()
        time.sleep(2.0)  # GRBL resets on port open
        self._write("$X")  # clear alarm
        self._write("G91")  # relative moves

    def _write(self, line: str) -> None:
        self._serial.write(f"{line}\n".encode())
        self._serial.readline()

    def set_rate(self, deg_per_s: float) -> None:
        self._rate = deg_per_s
        self._t0 = time.perf_counter()
        # One very long relative move at the requested feed rate; GRBL
        # interpolates it at a constant velocity, which is exactly what we want.
        revolutions = 10_000.0
        mm_per_rev = self._steps_per_rev  # GRBL's "mm" are our steps
        feed = deg_per_s / 360.0 * mm_per_rev * 60.0  # units/min
        self._write(f"G1 X{revolutions * mm_per_rev:.3f} F{feed:.3f}")

    def angle_deg(self, elapsed_s: float) -> float:
        return self._rate * elapsed_s

    def stop(self) -> None:
        self._write("!")  # feed hold
        self._write("\x18")  # soft reset
        self._serial.close()


def run(
    driver: TurntableDriver,
    rate_deg_s: float,
    duration_s: float,
    writer: CaptureWriter,
    *,
    sample_hz: float = 100.0,
    dry_run: bool,
) -> None:
    """Spin at a constant rate, logging the commanded angle as ground truth."""
    driver.set_rate(rate_deg_s)
    n = int(duration_s * sample_hz)
    t0 = time.perf_counter()
    try:
        for i in range(n):
            t = i / sample_hz
            if not dry_run:
                target = t0 + t
                now = time.perf_counter()
                if target > now:
                    time.sleep(target - now)
            angle = math.radians(driver.angle_deg(t))
            # Rotation about Z only; the phone lies flat on the platter.
            writer.add_pose(
                PoseSample(
                    t_ns=int(t * 1e9),
                    px=0.0,
                    py=0.0,
                    pz=0.0,
                    qx=0.0,
                    qy=0.0,
                    qz=math.sin(angle / 2),
                    qw=math.cos(angle / 2),
                )
            )
    finally:
        driver.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=30.0, help="degrees per second")
    parser.add_argument("--duration", type=float, default=60.0, help="seconds; 60 s is the L1 bar")
    parser.add_argument("--device-label", default="unknown")
    parser.add_argument("--os-name", default="unknown")
    parser.add_argument("--os-version", default="unknown")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--steps-per-rev", type=float, default=3200.0)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "captures")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    driver: TurntableDriver = (
        DryRunTurntable() if args.dry_run else GrblTurntable(args.port, args.steps_per_rev)
    )
    device = DeviceInfo(label=args.device_label, os_name=args.os_name, os_version=args.os_version)
    with CaptureWriter(
        args.out,
        "turntable",
        f"turntable-{args.rate:g}dps",
        1,
        device,
        Conditions(notes=f"constant {args.rate:g} deg/s"),
    ) as writer:
        writer.manifest.extra["rate_deg_per_s"] = args.rate
        print(f"spinning at {args.rate:g} deg/s for {args.duration:g}s -> {writer.path.name}")
        run(driver, args.rate, args.duration, writer, dry_run=args.dry_run)
    print("done")


if __name__ == "__main__":
    main()
