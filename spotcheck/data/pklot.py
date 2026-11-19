"""Convert PKLot (de Almeida et al., 2015) to YOLO detection format.  Owner: Ridham

PKLot layout:  PKLot/PKLot/<LOT>/<Weather>/<YYYY-MM-DD>/<image>.jpg (+ same-name .xml)
Each XML lists <space id=.. occupied=0|1> with a <rotatedRect> and a <contour>.
We turn every space into an axis-aligned box (min/max of its contour points).

Splitting is by *capture day* (a "session"), never by image, so near-duplicate
frames from the same day cannot leak between train and test.

Shortcut: Roboflow Universe also hosts PKLot already in YOLO format; see
`download_roboflow.py`. This script is for building the split ourselves.

Usage:
    python -m spotcheck.data.pklot --root /data/PKLot/PKLot --out datasets/pklot
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

from spotcheck.common import CLASS_NAMES, xyxy_to_yolo, write_yolo_labels

WEATHER_DIRS = {"sunny", "cloudy", "rainy"}


def parse_pklot_xml(xml_path: str | Path) -> list[tuple[int, float, float, float, float]]:
    """Return [(cls, x1, y1, x2, y2)] in pixels. Spaces without a label are skipped."""
    root = ET.parse(xml_path).getroot()
    boxes = []
    for space in root.iter("space"):
        occ = space.get("occupied")
        if occ not in ("0", "1"):
            continue
        pts = [(float(p.get("x")), float(p.get("y"))) for p in space.iter("point")]
        if len(pts) < 3:  # fall back to the rotated rectangle
            rr = space.find("rotatedRect")
            if rr is None:
                continue
            import cv2

            c, s, a = rr.find("center"), rr.find("size"), rr.find("angle")
            rect = ((float(c.get("x")), float(c.get("y"))),
                    (float(s.get("w")), float(s.get("h"))), float(a.get("d")))
            pts = cv2.boxPoints(rect).tolist()
        arr = np.asarray(pts)
        x1, y1 = arr.min(0)
        x2, y2 = arr.max(0)
        boxes.append((int(occ), x1, y1, x2, y2))
    return boxes


def find_samples(root: Path):
    """Yield dicts with image/xml paths plus lot, weather and session metadata."""
    for xml in sorted(root.rglob("*.xml")):
        img = xml.with_suffix(".jpg")
        if not img.exists():
            continue
        rel = img.relative_to(root).parts
        weather = next((p.lower() for p in rel if p.lower() in WEATHER_DIRS), "unknown")
        yield {
            "image": img,
            "xml": xml,
            "lot": rel[0],
            "weather": weather,
            "session": f"{rel[0]}_{img.parent.name}",  # lot + capture date
        }


def split_sessions(sessions, ratios=(0.7, 0.15, 0.15), seed=0):
    sessions = sorted(set(sessions))
    random.Random(seed).shuffle(sessions)
    n = len(sessions)
    n_tr = max(1, round(ratios[0] * n))
    n_va = max(1, round(ratios[1] * n)) if n > 2 else 0
    out = {}
    for i, s in enumerate(sessions):
        out[s] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    return out


def convert(root: Path, out: Path, max_per_session: int | None = None, seed: int = 0):
    samples = list(find_samples(root))
    if not samples:
        raise SystemExit(f"No PKLot image/xml pairs found under {root}")
    by_session = defaultdict(list)
    for s in samples:
        by_session[s["session"]].append(s)
    if max_per_session:  # PKLot is huge and highly redundant; subsample per day
        rng = random.Random(seed)
        for k in by_session:
            rng.shuffle(by_session[k])
            by_session[k] = sorted(by_session[k][:max_per_session], key=lambda d: d["image"])
    split_of = split_sessions(by_session.keys(), seed=seed)

    import cv2

    manifest = []
    for session, items in by_session.items():
        split = split_of[session]
        for s in items:
            h, w = cv2.imread(str(s["image"])).shape[:2]
            name = f"{session}_{s['image'].stem}".replace(" ", "_")
            rows = [(c, *xyxy_to_yolo(x1, y1, x2, y2, w, h))
                    for c, x1, y1, x2, y2 in parse_pklot_xml(s["xml"])]
            dst = out / "images" / split / f"{name}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s["image"], dst)
            write_yolo_labels(out / "labels" / split / f"{name}.txt", rows)
            manifest.append({"image": f"{name}.jpg", "split": split, "source": "pklot",
                             "lot": s["lot"], "session": session, "weather": s["weather"],
                             "lighting": "day"})
    write_manifest(out / "manifest.csv", manifest)
    write_data_yaml(out)
    counts = defaultdict(int)
    for m in manifest:
        counts[m["split"]] += 1
    print(f"PKLot -> {out}: {dict(counts)} images from {len(by_session)} sessions")


def write_manifest(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_data_yaml(out: Path):
    out = out.resolve()
    lines = [f"path: {out}", "train: images/train", "val: images/val", "test: images/test",
             "names:"] + [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    (out / "data.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("datasets/pklot"))
    ap.add_argument("--max-per-session", type=int, default=60,
                    help="cap images per capture day (0 = keep all)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    convert(a.root, a.out, a.max_per_session or None, a.seed)
