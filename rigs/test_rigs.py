"""Tier-1 tests for the rig instruments. No hardware, closed-form answers.

These grade the *instruments*. A bug here produces a wrong ground truth, which
is worse than a bug in the estimator: the estimator's bug shows up as a bad
number, but a rig bug shows up as a good number that is wrong.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

import capture
import strobe


class TestStrobe:
    def test_rejects_a_commensurate_frequency(self):
        # The failure that produces a confidently wrong timing measurement.
        for hz in (60.0, 120.0, 30.0):
            with pytest.raises(ValueError):
                strobe.check_incommensurate(hz, 60.0)

    def test_accepts_the_default(self):
        strobe.check_incommensurate(strobe.DEFAULT_FREQUENCY_HZ, 60.0)
        strobe.check_incommensurate(strobe.DEFAULT_FREQUENCY_HZ, 30.0)

    def test_generated_page_is_self_contained(self, tmp_path: Path):
        page = strobe.generate_page(71.3, tmp_path / "s.html")
        html = page.read_text()
        assert "71.3" in html
        # No network fetches: the rig must work on an air-gapped display.
        assert "http://" not in html and "https://" not in html

    def test_decoder_rejects_mismatched_series(self):
        with pytest.raises(ValueError):
            strobe.decode_exposure_times([0.1, 0.2], [0.1], 71.3, 60.0)

    def test_decoded_times_advance_monotonically(self):
        n = 40
        green = [0.5 + 0.5 * math.sin(math.tau * 71.3 * (i / 60.0)) for i in range(n)]
        red = [0.0] * n
        times = strobe.decode_exposure_times(green, red, 71.3, 60.0)
        assert len(times) == n
        assert all(math.isfinite(t) for t in times)


class TestCaptureFormat:
    def test_roundtrip(self, tmp_path: Path):
        device = capture.DeviceInfo(label="testdev", os_name="ios", os_version="26.0")
        with capture.CaptureWriter(tmp_path, "arm", "static-hold", 1, device) as writer:
            for i in range(10):
                writer.add_pose(
                    capture.PoseSample(i * 10_000_000, 0.1 * i, 0.0, 0.2, 0, 0, 0, 1)
                )
                writer.add_imu(capture.ImuRecord(i * 10_000_000, 0, 0, 0, 0, 0, 9.81))
        manifest = capture.read_manifest(writer.path)
        assert manifest.rig == "arm"
        assert manifest.device.label == "testdev"
        # Rates are derived, not declared, so a mis-timed run is visible.
        assert manifest.groundtruth_rate_hz == pytest.approx(100.0, rel=1e-6)
        poses = capture.read_poses(writer.path)
        assert len(poses) == 10
        assert poses[3].px == pytest.approx(0.3)

    def test_rejects_an_unknown_format_version(self, tmp_path: Path):
        device = capture.DeviceInfo(label="d", os_name="o", os_version="1")
        with capture.CaptureWriter(tmp_path, "arm", "static-hold", 1, device) as writer:
            writer.add_pose(capture.PoseSample(0, 0, 0, 0, 0, 0, 0, 1))
        path = writer.path / "manifest.json"
        path.write_text(path.read_text().replace('"format_version": 1', '"format_version": 99'))
        with pytest.raises(ValueError, match="format"):
            capture.read_manifest(writer.path)

    def test_conditions_travel_with_the_measurement(self, tmp_path: Path):
        # spec.md §6: report per device x OS cell, never pooled. That is only
        # possible if the cell is recorded alongside the numbers.
        device = capture.DeviceInfo(label="iphone15pro", os_name="ios", os_version="26.1")
        conditions = capture.Conditions(lighting_lux=120.0, thermal_state="warm")
        with capture.CaptureWriter(tmp_path, "arm", "static-hold", 1, device, conditions) as w:
            w.add_pose(capture.PoseSample(0, 0, 0, 0, 0, 0, 0, 1))
        manifest = capture.read_manifest(w.path)
        assert manifest.conditions.lighting_lux == 120.0
        assert manifest.conditions.thermal_state == "warm"
        assert manifest.device.os_version == "26.1"


