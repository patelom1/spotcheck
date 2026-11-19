"""Build the UT Dallas campus dataset from a Roboflow YOLO export.  Owner: Ridham

Workflow
  1. Upload photos to Roboflow, label boxes as `empty` / `occupied`.
  2. Export "YOLOv8" format (no augmentation, no Roboflow split -- put everything in train).
  3. Fill in `campus_manifest.csv` (one row per photo):
        image,lot,session,lighting,occlusion
        IMG_0412.jpg,lot_j,2026-11-19_am,day,low
     `session` = one trip to one lot (date + time block). `lighting` in {day, dusk, night}.
  4. python -m spotcheck.data.campus --export roboflow_export --manifest campus_manifest.csv \
         --out datasets/campus [--holdout-lot lot_m]

Splits are made by session (70/15/15). With --holdout-lot, every image of that
lot goes to a separate `test_crosslot` split instead (cross-lot generalization).
Roboflow renames files (`IMG_0412_jpg.rf.<hash>.jpg`); we match on the original stem.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

from spotcheck.common import CLASS_NAMES, IMG_EXTS, label_path_for
from spotcheck.data.pklot import split_sessions, write_manifest


def original_stem(roboflow_name: str) -> str:
    stem = Path(roboflow_name).stem
    if ".rf." in stem:
        stem = stem.split(".rf.")[0]
        for ext in ("_jpg", "_jpeg", "_png", "_JPG", "_JPEG", "_PNG", "_heic", "_HEIC"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
    return stem


def build(export: Path, manifest_csv: Path, out: Path, holdout_lot: str | None = None, seed: int = 0):
    meta = {}
    with open(manifest_csv) as f:
        for r in csv.DictReader(f):
            meta[Path(r["image"]).stem] = r
    imgs = [p for p in export.rglob("*") if p.suffix.lower() in IMG_EXTS and "images" in p.parts]
    if not imgs:
        raise SystemExit(f"No images found under {export}/**/images/")

    matched, missing = [], []
    for p in imgs:
        m = meta.get(original_stem(p.name))
        (matched if m else missing).append((p, m))
    if missing:
        print(f"WARNING: {len(missing)} images have no manifest row and are skipped, e.g. "
              f"{[original_stem(p.name) for p, _ in missing[:5]]}")

    pool = [(p, m) for p, m in matched if m["lot"] != holdout_lot]
    split_of = split_sessions({m["session"] for _, m in pool}, seed=seed)

    rows = []
    for p, m in matched:
        split = "test_crosslot" if m["lot"] == holdout_lot else split_of[m["session"]]
        name = original_stem(p.name)
        dst = out / "images" / split / f"{name}{p.suffix.lower()}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        lab_dst = out / "labels" / split / f"{name}.txt"
        lab_dst.parent.mkdir(parents=True, exist_ok=True)
        src_lab = label_path_for(p)
        lab_dst.write_text(src_lab.read_text() if src_lab.exists() else "")
        rows.append({"image": dst.name, "split": split, "source": "campus", "lot": m["lot"],
                     "session": m["session"], "weather": m.get("weather", ""),
                     "lighting": m.get("lighting", "day"), "occlusion": m.get("occlusion", "")})
    write_manifest(out / "manifest.csv", rows)

    out = out.resolve()
    lines = [f"path: {out}", "train: images/train", "val: images/val", "test: images/test"]
    if holdout_lot:
        lines.append("test_crosslot: images/test_crosslot")
    lines += ["names:"] + [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    (out / "data.yaml").write_text("\n".join(lines) + "\n")

    print("images per split:", dict(Counter(r["split"] for r in rows)))
    print("images per lighting:", dict(Counter(r["lighting"] for r in rows)))
    n_boxes = Counter()
    for f in (out / "labels").rglob("*.txt"):
        for line in f.read_text().splitlines():
            if line.strip():
                n_boxes[CLASS_NAMES[int(line.split()[0])]] += 1
    print("labeled spaces:", dict(n_boxes))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("datasets/campus"))
    ap.add_argument("--holdout-lot", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.export, a.manifest, a.out, a.holdout_lot, a.seed)
