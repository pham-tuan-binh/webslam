"""ChArUco rig: per-device intrinsics ground truth.

spec.md §6 ground-truth table: provides *per-device intrinsics ground truth
(validation only, never runtime)*, build cost hours.

The parenthetical is the important part and it is enforced here rather than
merely stated: nothing in `crates/` reads this output. These numbers exist to
score L2's estimate, and if the pipeline could consult them, L2's error would be
unmeasurable — we would be grading the answer sheet against itself.

Requires OpenCV. It is not a dependency of anything that ships.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

SQUARES_X = 8
SQUARES_Y = 11
SQUARE_LENGTH_M = 0.024
MARKER_LENGTH_M = 0.018
DICTIONARY = "DICT_5X5_100"


@dataclass
class IntrinsicsTruth:
    """Ground-truth intrinsics for one device and one camera."""

    device_label: str
    camera_id: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float
    rms_reprojection_px: float
    image_count: int
    board: str

    def hfov_degrees(self) -> float:
        import math

        return 2.0 * math.degrees(math.atan(self.width * 0.5 / self.fx))


def _cv():
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - optional tool path
        raise SystemExit("opencv-python is required: pip install 'opencv-python>=4.10'") from exc
    return cv2


def generate_board(path: Path, dpi: int = 300) -> Path:
    """Render the board to PNG at a physically correct size.

    Print it at 100% scale and **measure a square with callipers afterwards**.
    Printer scaling is the single most common source of a systematically wrong
    "ground truth", and a 2% scale error here becomes a 2% focal error that
    looks exactly like an L2 bug.
    """
    cv2 = _cv()
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH_M, MARKER_LENGTH_M, dictionary)
    px_per_m = dpi / 0.0254
    size = (int(SQUARES_X * SQUARE_LENGTH_M * px_per_m), int(SQUARES_Y * SQUARE_LENGTH_M * px_per_m))
    image = board.generateImage(size, marginSize=int(0.01 * px_per_m))
    cv2.imwrite(str(path), image)
    return path


def calibrate(images: list[Path], device_label: str, camera_id: str = "") -> IntrinsicsTruth:
    """Run a ChArUco calibration over a directory of captures."""
    cv2 = _cv()
    import numpy as np

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH_M, MARKER_LENGTH_M, dictionary)
    detector = cv2.aruco.CharucoDetector(board)

    all_corners: list = []
    all_ids: list = []
    size: tuple[int, int] | None = None

    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        size = (image.shape[1], image.shape[0])
        corners, ids, _, _ = detector.detectBoard(image)
        # Fewer than 8 corners gives a pose that is technically solvable and
        # practically noise; including such views inflates the reported RMS and
        # biases the distortion terms.
        if corners is not None and ids is not None and len(corners) >= 8:
            all_corners.append(corners)
            all_ids.append(ids)

    if len(all_corners) < 8 or size is None:
        raise SystemExit(
            f"need at least 8 usable views, got {len(all_corners)}. "
            "Cover the frame corners — distortion is only observable where it is large."
        )

    rms, k, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, size, None, None
    )
    d = np.asarray(dist).ravel()
    return IntrinsicsTruth(
        device_label=device_label,
        camera_id=camera_id,
        width=size[0],
        height=size[1],
        fx=float(k[0, 0]),
        fy=float(k[1, 1]),
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
        k1=float(d[0]) if len(d) > 0 else 0.0,
        k2=float(d[1]) if len(d) > 1 else 0.0,
        p1=float(d[2]) if len(d) > 2 else 0.0,
        p2=float(d[3]) if len(d) > 3 else 0.0,
        k3=float(d[4]) if len(d) > 4 else 0.0,
        rms_reprojection_px=float(rms),
        image_count=len(all_corners),
        board=f"charuco-{SQUARES_X}x{SQUARES_Y}-{SQUARE_LENGTH_M}m-{DICTIONARY}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("board", help="render the board for printing")
    gen.add_argument("--out", type=Path, default=Path("charuco-board.png"))
    gen.add_argument("--dpi", type=int, default=300)

    cal = sub.add_parser("calibrate", help="calibrate from a directory of images")
    cal.add_argument("images", type=Path)
    cal.add_argument("--device-label", required=True)
    cal.add_argument("--camera-id", default="")
    cal.add_argument("--out", type=Path, default=Path("intrinsics-truth.json"))

    args = parser.parse_args()

    if args.command == "board":
        print(f"wrote {generate_board(args.out, args.dpi)}")
        print("Print at 100% scale, then MEASURE a square with callipers before trusting it.")
        return

    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    truth = calibrate(paths, args.device_label, args.camera_id)
    args.out.write_text(json.dumps(asdict(truth), indent=2) + "\n")
    print(f"wrote {args.out}")
    print(f"  f = {truth.fx:.2f} x {truth.fy:.2f} px   hfov = {truth.hfov_degrees():.1f} deg")
    print(f"  k1 = {truth.k1:+.4f}  k2 = {truth.k2:+.4f}   rms = {truth.rms_reprojection_px:.3f} px")
    if truth.k1 < -0.1:
        print("  note: strong barrel distortion. This is the Hayman & Murray case;")
        print("        expect the no-distortion L2 ablation arm to overestimate focal length.")


if __name__ == "__main__":
    main()
