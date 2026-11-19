"""Tiny synthetic parking-lot dataset for smoke-testing the pipeline (NOT for results).

    python -m scripts.make_synthetic_dataset --out datasets/synthetic --n 60

Draws a gray lot with rows of painted spaces; some contain colored "cars".
Lets anyone run train -> evaluate -> plots -> API -> demo video in minutes on CPU
before the real data is ready.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from spotcheck.common import CLASS_NAMES, write_yolo_labels, xyxy_to_yolo
from spotcheck.data.pklot import write_data_yaml, write_manifest


def make_image(rng, w=640, h=480):
    base = rng.randint(70, 130)
    noise = np.random.default_rng(rng.randint(0, 1 << 30)).integers(0, 12, (h, w, 3))
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    rows = []
    sw, sh = rng.randint(48, 60), rng.randint(80, 100)
    for r, y0 in enumerate(range(30, h - sh, sh + 50)):
        for x0 in range(20, w - sw, sw + 6):
            cv2.rectangle(img, (x0, y0), (x0 + sw, y0 + sh), (235, 235, 235), 2)
            occ = rng.random() < 0.55
            if occ:
                color = tuple(int(c) for c in rng.choice([(30, 30, 160), (160, 60, 20), (20, 20, 20),
                                                          (200, 200, 200), (40, 120, 40), (90, 90, 110)]))
                m = 6
                cv2.rectangle(img, (x0 + m, y0 + m), (x0 + sw - m, y0 + sh - m), color, -1)
                cv2.rectangle(img, (x0 + m + 6, y0 + m + 14), (x0 + sw - m - 6, y0 + m + 32), (60, 60, 60), -1)
            rows.append((int(occ), *xyxy_to_yolo(x0, y0, x0 + sw, y0 + sh, w, h)))
    lighting = rng.choice(["day", "dusk", "night"])
    if lighting == "dusk":
        img = (img * np.array([0.7, 0.8, 1.0])).astype(np.uint8)
    elif lighting == "night":
        img = (img * 0.4).astype(np.uint8)
    return img, rows, lighting


def main(out: Path, n: int, seed: int = 0):
    rng = random.Random(seed)
    manifest = []
    for i in range(n):
        split = "train" if i < 0.7 * n else "val" if i < 0.85 * n else "test"
        img, rows, light = make_image(rng)
        name = f"syn_{i:04d}"
        p = out / "images" / split / f"{name}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img)
        write_yolo_labels(out / "labels" / split / f"{name}.txt", rows)
        manifest.append({"image": p.name, "split": split, "source": "synthetic", "lot": f"lot_{i % 3}",
                         "session": f"s{i // 5}", "weather": "", "lighting": light})
    write_manifest(out / "manifest.csv", manifest)
    write_data_yaml(out)
    print(f"synthetic dataset ({len(CLASS_NAMES)} classes, {n} images) -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("datasets/synthetic"))
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    main(a.out, a.n)
