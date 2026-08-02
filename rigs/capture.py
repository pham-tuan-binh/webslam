"""The capture record: what a rig run writes, and what the harness reads.

One format for every rig, because spec.md §6 wants per-layer metrics computed
against ground truth by the same code path regardless of which rig produced the
truth. A capture is a directory:

    captures/2026-08-01T12-00-00Z_revisit-loop_pixel8/
      manifest.json      # this module's Manifest, including the rig revision
      groundtruth.csv    # t_ns, px, py, pz, qx, qy, qz, qw
      imu.csv            # t_ns, gx, gy, gz, ax, ay, az   (device, if logged)
      frames/000000.png  # optional; usually the phone records its own video
      notes.md

The manifest carries the *conditions*, not just the data. spec.md §6 asks for
results reported per device and per cell of the device x OS matrix, never
pooled — that is only possible if the conditions travel with the measurement.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

CAPTURE_FORMAT_VERSION = 1


@dataclass
class DeviceInfo:
    """Which phone, running what. Never aggregate across two of these."""

    label: str
    """Short stable id, e.g. "pixel8" or "iphone15pro"."""
    os_name: str
    os_version: str
    browser: str = ""
    browser_version: str = ""
    camera_id: str = ""
    """Which physical camera, if the device exposes more than one."""


@dataclass
class Conditions:
    """Everything that could plausibly move a number."""

    lighting_lux: float | None = None
    scene: str = ""
    """Free text: "office wall, low texture", "textured poster", ..."""
    thermal_state: str = "cold"
    """"cold" | "warm" | "throttled". spec.md §6 wants time-to-degradation."""
    elapsed_since_session_start_s: float = 0.0
    notes: str = ""


@dataclass
class Manifest:
    """The header of a capture."""

    format_version: int
    created_utc: str
    rig: str
    """"arm" | "turntable" | "strobe" | "charuco" | "handheld"."""
    trajectory: str
    trajectory_revision: int
    device: DeviceInfo
    conditions: Conditions
    groundtruth_frame: str = "arm_base"
    """Frame the ground-truth poses are expressed in."""
    groundtruth_rate_hz: float = 0.0
    imu_rate_hz: float = 0.0
    frame_count: int = 0
    git_commit: str = ""
    extra: dict[str, object] = field(default_factory=dict)


def git_commit() -> str:
    """Current commit, or empty. Recorded so a number can be re-derived."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


@dataclass
class PoseSample:
    """One ground-truth pose."""

    t_ns: int
    px: float
    py: float
    pz: float
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass
class ImuRecord:
    """One inertial sample, as logged by the device."""

    t_ns: int
    gx: float
    gy: float
    gz: float
    ax: float
    ay: float
    az: float


class CaptureWriter:
    """Writes one capture directory.

    Use as a context manager; the manifest is written on exit so that a crashed
    run leaves an obviously-incomplete directory rather than a plausible one.
    """

    def __init__(
        self,
        root: Path,
        rig: str,
        trajectory: str,
        trajectory_revision: int,
        device: DeviceInfo,
        conditions: Conditions | None = None,
    ) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        self.path = Path(root) / f"{stamp}_{trajectory}_{device.label}"
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "frames").mkdir(exist_ok=True)
        self.manifest = Manifest(
            format_version=CAPTURE_FORMAT_VERSION,
            created_utc=stamp,
            rig=rig,
            trajectory=trajectory,
            trajectory_revision=trajectory_revision,
            device=device,
            conditions=conditions or Conditions(),
            git_commit=git_commit(),
            extra={"host": platform.platform()},
        )
        self._poses: list[PoseSample] = []
        self._imu: list[ImuRecord] = []

    def add_pose(self, sample: PoseSample) -> None:
        self._poses.append(sample)

    def add_imu(self, sample: ImuRecord) -> None:
        self._imu.append(sample)

    def _write_csv(self, name: str, rows: Sequence[object], header: Iterable[str]) -> None:
        if not rows:
            return
        with (self.path / name).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for row in rows:
                writer.writerow(asdict(row).values())  # type: ignore[arg-type]

    def close(self) -> None:
        self._write_csv(
            "groundtruth.csv",
            self._poses,
            ["t_ns", "px", "py", "pz", "qx", "qy", "qz", "qw"],
        )
        self._write_csv("imu.csv", self._imu, ["t_ns", "gx", "gy", "gz", "ax", "ay", "az"])
        if len(self._poses) > 1:
            span_s = (self._poses[-1].t_ns - self._poses[0].t_ns) / 1e9
            if span_s > 0:
                self.manifest.groundtruth_rate_hz = (len(self._poses) - 1) / span_s
        if len(self._imu) > 1:
            span_s = (self._imu[-1].t_ns - self._imu[0].t_ns) / 1e9
            if span_s > 0:
                self.manifest.imu_rate_hz = (len(self._imu) - 1) / span_s
        self.manifest.frame_count = len(list((self.path / "frames").glob("*.png")))
        (self.path / "manifest.json").write_text(
            json.dumps(asdict(self.manifest), indent=2) + "\n"
        )

    def __enter__(self) -> "CaptureWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def read_manifest(capture: Path) -> Manifest:
    """Load a manifest, rejecting a format this code does not understand."""
    data = json.loads((Path(capture) / "manifest.json").read_text())
    version = data.get("format_version")
    if version != CAPTURE_FORMAT_VERSION:
        raise ValueError(
            f"{capture}: capture format v{version}, this code reads v{CAPTURE_FORMAT_VERSION}"
        )
    data["device"] = DeviceInfo(**data["device"])
    data["conditions"] = Conditions(**data["conditions"])
    return Manifest(**data)


def read_poses(capture: Path) -> list[PoseSample]:
    """Load ground-truth poses."""
    with (Path(capture) / "groundtruth.csv").open() as handle:
        return [
            PoseSample(
                t_ns=int(row["t_ns"]),
                **{k: float(row[k]) for k in ("px", "py", "pz", "qx", "qy", "qz", "qw")},
            )
            for row in csv.DictReader(handle)
        ]
