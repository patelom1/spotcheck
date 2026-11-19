"""Cross-site split of PKLot: train at some lots, adapt to and test at a new one.  Owner: Ridham

PKLot has three real parking lots, each with its own fixed camera:
    UFPR04, UFPR05  (Federal University of Parana)   PUCPR  (Pontifical Catholic University of Parana)
Photos were taken every 5 minutes for ~30 days in sunny, cloudy and rainy weather, and each
file name carries its timestamp, so every image has real weather + time-of-day metadata.

This script treats one lot as the "new site" (default PUCPR) and the others as the
public "source" data, mirroring our proposal's question: does a detector trained on
other lots transfer, and how many labeled local images does it take to adapt?

Output (datasets/xsite/):
    images|labels/source_train, source_val      all non-target lots, split by capture day
    images|labels/target_pool, target_val, target_test   target lot, split by capture day
    target_train_<N>.txt     nested random subsets of target_pool (N = 25, 50, 100, 200 ...)
    source.yaml              train on source, validate on source
    target_<N>.yaml          train on N target images, validate/test on target
    combined_<N>.yaml        source + N target images
    manifest.csv             lot, session (day), weather, hour, time_of_day for every image

Usage:
    python -m spotcheck.data.crosssite --root /data/PKLot/PKLot --target PUCPR --out datasets/xsite
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from spotcheck.common import CLASS_NAMES, write_yolo_labels, xyxy_to_yolo
from spotcheck.data.pklot import find_samples, parse_pklot_xml, write_manifest

TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})")


def time_of_day(hour: int) -> str:
    if hour < 10:
        return "morning"
    if hour < 14:
        return "midday"
    if hour < 17:
        return "afternoon"
    return "evening"


def parse_time(stem: str):
    m = TS_RE.search(stem)
    return (int(m.group(2)), int(m.group(3))) if m else (None, None)


def split_days(days, fractions, seed):
    days = sorted(days)
    random.Random(seed).shuffle(days)
    out, start = [], 0
    for i, f in enumerate(fractions):
        n = len(days) - start if i == len(fractions) - 1 else max(1, round(f * len(days)))
        out.append(set(days[start:start + n]))
        start += n
    return out


def _yaml(path: Path, root: Path, train, val, test=None):
    def fmt(v):
        if isinstance(v, list):
            return "\n" + "\n".join(f"  - {x}" for x in v)
        return f" {v}"
    lines = [f"path: {root}", "train:" + fmt(train), "val:" + fmt(val)]
    if test:
        lines.append("test:" + fmt(test))
    lines += ["names:"] + [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    path.write_text("\n".join(lines) + "\n")


def convert(root: Path, out: Path, target: str = "PUCPR", sizes=(25, 50, 100, 200),
            max_per_session: int = 40, seed: int = 0):
    samples = list(find_samples(root))
    lots = sorted({s["lot"] for s in samples})
    if target not in lots:
        raise SystemExit(f"target lot '{target}' not found; lots: {lots}")

    by_session = defaultdict(list)
    for s in samples:
        by_session[s["session"]].append(s)
    rng = random.Random(seed)
    for k in by_session:  # consecutive 5-min frames are near-duplicates: subsample per day
        rng.shuffle(by_session[k])
        by_session[k] = by_session[k][:max_per_session]

    src_days = [k for k in by_session if not k.startswith(target + "_")]
    tgt_days = [k for k in by_session if k.startswith(target + "_")]
    s_tr, s_va = split_days(src_days, [0.85, 0.15], seed)
    t_pool, t_va, t_te = split_days(tgt_days, [0.35, 0.15, 0.5], seed)
    split_of = {**{d: "source_train" for d in s_tr}, **{d: "source_val" for d in s_va},
                **{d: "target_pool" for d in t_pool}, **{d: "target_val" for d in t_va},
                **{d: "target_test" for d in t_te}}

    manifest = []
    for session, items in sorted(by_session.items()):
        split = split_of[session]
        for s in items:
            img = cv2.imread(str(s["image"]))
            if img is None:
                continue
            h, w = img.shape[:2]
            name = f"{session}_{s['image'].stem}".replace(" ", "_")
            dst = out / "images" / split / f"{name}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s["image"], dst)
            rows = [(c, *xyxy_to_yolo(x1, y1, x2, y2, w, h)) for c, x1, y1, x2, y2 in parse_pklot_xml(s["xml"])]
            write_yolo_labels(out / "labels" / split / f"{name}.txt", rows)
            hour, minute = parse_time(s["image"].stem)
            manifest.append({"image": dst.name, "split": split, "source": "pklot", "lot": s["lot"],
                             "session": session, "weather": s["weather"],
                             "hour": "" if hour is None else hour,
                             "time_of_day": "unknown" if hour is None else time_of_day(hour),
                             "lighting": s["weather"]})
    write_manifest(out / "manifest.csv", manifest)

    out = out.resolve()
    pool = sorted((out / "images" / "target_pool").glob("*.jpg"))
    rng.shuffle(pool)
    written = []
    for n in sizes:
        if n > len(pool):
            print(f"skip N={n}: only {len(pool)} target-pool images")
            continue
        lst = out / f"target_train_{n}.txt"
        lst.write_text("\n".join(str(p) for p in pool[:n]) + "\n")  # nested: N=25 ⊂ N=50 ⊂ ...
        _yaml(out / f"target_{n}.yaml", out, str(lst), "images/target_val", "images/target_test")
        _yaml(out / f"combined_{n}.yaml", out, [str(out / "images" / "source_train"), str(lst)],
              "images/target_val", "images/target_test")
        written.append(n)
    _yaml(out / "source.yaml", out, "images/source_train", "images/source_val", "images/target_test")

    c = Counter(m["split"] for m in manifest)
    print(f"lots: source={[l for l in lots if l != target]} target={target}")
    print("images per split:", dict(c))
    print("target test by weather:", dict(Counter(m["weather"] for m in manifest if m["split"] == "target_test")))
    print("target test by time of day:", dict(Counter(m["time_of_day"] for m in manifest if m["split"] == "target_test")))
    print(f"target-train subsets: {written}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="folder containing PUCPR/, UFPR04/, UFPR05/")
    ap.add_argument("--out", type=Path, default=Path("datasets/xsite"))
    ap.add_argument("--target", default="PUCPR")
    ap.add_argument("--sizes", type=int, nargs="+", default=[25, 50, 100, 200])
    ap.add_argument("--max-per-session", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    convert(a.root, a.out, a.target, a.sizes, a.max_per_session, a.seed)
