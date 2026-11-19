"""Convert CNRPark-EXT (Amato et al., 2017) full images to YOLO format.  Owner: Ridham

CNRPark-EXT ships labels per *patch*, plus per-camera CSVs with each slot's box
in full-resolution (2592x1944) coordinates. We join them back onto the
1000x750 full images:

    CNR-EXT_FULL_IMAGE_1000x750/
        FULL_IMAGE_1000x750/<WEATHER>/<YYYY-MM-DD>/camera<N>/<YYYY-MM-DD_HHMM>.jpg
        camera<N>.csv                    # SlotId,X,Y,W,H  (2592x1944 coords)
    LABELS/all.txt                       # PATCHES/<W>/<date>/camera<N>/<W>_<date>_<HH.MM>_C<NN>_<slot>.jpg <0|1>

Check these paths against your download before running; the archive layout is
the only thing this script assumes. Slots with no label for a frame are skipped.

Usage:
    python -m spotcheck.data.cnrpark --full-root /data/CNR-EXT_FULL_IMAGE_1000x750 \
        --labels /data/LABELS/all.txt --out datasets/cnrpark
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from spotcheck.common import xyxy_to_yolo, write_yolo_labels
from spotcheck.data.pklot import split_sessions, write_data_yaml, write_manifest

FULL_W, FULL_H = 2592, 1944
PATCH_RE = re.compile(r"([A-Z])_(\d{4}-\d{2}-\d{2})_(\d{2})\.(\d{2})_C(\d+)_(\d+)\.jpg$")
WEATHER = {"S": "sunny", "O": "overcast", "R": "rainy"}


def load_slot_boxes(full_root: Path) -> dict[int, dict[int, tuple]]:
    cams = {}
    for f in full_root.glob("camera*.csv"):
        cam = int(re.findall(r"\d+", f.stem)[0])
        with open(f) as fh:
            cams[cam] = {int(r["SlotId"]): (float(r["X"]), float(r["Y"]), float(r["W"]), float(r["H"]))
                         for r in csv.DictReader(fh)}
    if not cams:
        raise SystemExit(f"No camera*.csv files in {full_root}")
    return cams


def load_patch_labels(labels_txt: Path):
    """-> {(weather, date, HHMM, cam): {slot: label}}"""
    frames = defaultdict(dict)
    for line in labels_txt.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        m = PATCH_RE.search(parts[0])
        if not m:
            continue
        w, date, hh, mm, cam, slot = m.groups()
        frames[(WEATHER.get(w, w), date, hh + mm, int(cam))][int(slot)] = int(parts[1])
    return frames


def convert(full_root: Path, labels_txt: Path, out: Path, max_per_session: int = 40, seed: int = 0):
    import cv2

    slots = load_slot_boxes(full_root)
    frames = load_patch_labels(labels_txt)
    # index full images by (camera, date, HHMM) regardless of weather folder spelling
    images = {}
    for p in (full_root / "FULL_IMAGE_1000x750").rglob("*.jpg"):
        if p.parent.name.startswith("camera"):
            cam = int(re.findall(r"\d+", p.parent.name)[0])
            images[(cam, p.stem.replace("_", "").replace("-", ""))] = p
    samples = []
    for (weather, date, hhmm, cam), labels in frames.items():
        img = images.get((cam, date.replace("-", "") + hhmm))
        if img is None or cam not in slots:
            continue
        samples.append(dict(image=img, cam=cam, labels=labels, weather=weather,
                            session=f"cnr_cam{cam}_{date}"))
    if not samples:
        raise SystemExit("Could not match any labels to full images; check --full-root/--labels paths.")

    by_session = defaultdict(list)
    for s in samples:
        by_session[s["session"]].append(s)
    rng = random.Random(seed)
    for k in by_session:
        rng.shuffle(by_session[k])
        by_session[k] = by_session[k][:max_per_session]
    split_of = split_sessions(by_session.keys(), seed=seed)

    manifest = []
    for session, items in by_session.items():
        split = split_of[session]
        for s in items:
            h, w = cv2.imread(str(s["image"])).shape[:2]
            sx, sy = w / FULL_W, h / FULL_H
            rows = []
            for slot, lab in s["labels"].items():
                if slot not in slots[s["cam"]]:
                    continue
                x, y, bw, bh = slots[s["cam"]][slot]
                rows.append((lab, *xyxy_to_yolo(x * sx, y * sy, (x + bw) * sx, (y + bh) * sy, w, h)))
            name = f"{session}_{s['image'].stem}"
            dst = out / "images" / split / f"{name}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s["image"], dst)
            write_yolo_labels(out / "labels" / split / f"{name}.txt", rows)
            manifest.append({"image": f"{name}.jpg", "split": split, "source": "cnrpark",
                             "lot": f"cnr_cam{s['cam']}", "session": session,
                             "weather": s["weather"], "lighting": "day"})
    write_manifest(out / "manifest.csv", manifest)
    write_data_yaml(out)
    print(f"CNRPark-EXT -> {out}: {len(manifest)} images, {len(by_session)} sessions")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full-root", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("datasets/cnrpark"))
    ap.add_argument("--max-per-session", type=int, default=40)
    a = ap.parse_args()
    convert(a.full_root, a.labels, a.out, a.max_per_session)
