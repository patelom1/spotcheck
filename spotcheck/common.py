"""Shared constants and small helpers used across SpotCheck modules."""
from __future__ import annotations

from pathlib import Path

# Class order matches PKLot (0 = empty, 1 = occupied) and Roboflow's
# alphabetical export order, so label files from every source agree.
CLASS_NAMES = ["empty", "occupied"]
EMPTY, OCCUPIED = 0, 1

# BGR colors for drawing
COLORS = {EMPTY: (80, 200, 60), OCCUPIED: (40, 40, 220)}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
LIGHTING = ("day", "dusk", "night")

REPO_ROOT = Path(__file__).resolve().parents[1]


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def label_path_for(image_path: str | Path) -> Path:
    """YOLO convention: .../images/x.jpg -> .../labels/x.txt"""
    p = Path(image_path)
    parts = list(p.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    """Pixel box -> normalized (cx, cy, bw, bh), clipped to the image."""
    x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def write_yolo_labels(path: str | Path, rows: list[tuple[int, float, float, float, float]]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c, cx, cy, bw, bh in rows:
            if bw > 0 and bh > 0:
                f.write(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def read_yolo_labels(path: str | Path) -> list[tuple[int, float, float, float, float]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            rows.append((int(parts[0]), *map(float, parts[1:5])))
    return rows
