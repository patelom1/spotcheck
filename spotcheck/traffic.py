"""Real parking "traffic": how occupancy changes through the day.  Owner: Krishna

Uses the ground-truth labels and photo timestamps (no model needed), and optionally
a model's predictions, to plot the share of occupied spaces by hour of day:

    python -m spotcheck.traffic --data datasets/xsite --out results/figures
    python -m spotcheck.traffic --data datasets/xsite --per-image results/test_per_image.csv \
        --model finetune --out results/figures

Outputs occupancy_by_hour.png (+ .csv). With --per-image it overlays the model's
predicted occupancy on the target test images, so you can see whether it tracks the
real morning fill-up and evening emptying.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from spotcheck.common import read_yolo_labels  # noqa: E402
from spotcheck.counting import count_labels  # noqa: E402


def occupancy_by_hour(root: Path):
    with open(root / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("hour", "") == "":
            continue
        lab = root / "labels" / r["split"] / (Path(r["image"]).stem + ".txt")
        c = count_labels(read_yolo_labels(lab))
        if c.total:
            per[r["lot"]][int(r["hour"])].append(c.occupancy_rate)
    return per


def main(root: Path, out: Path, per_image: Path | None = None, model: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    per = occupancy_by_hour(root)
    if not per:
        raise SystemExit("no timestamped images in manifest.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = ["#4C6EF5", "#F59F00", "#12B886", "#E64980"]
    table = []
    for i, (lot, hours) in enumerate(sorted(per.items())):
        hs = sorted(hours)
        m = [100 * np.mean(hours[h]) for h in hs]
        ax.plot(hs, m, "-o", ms=4, lw=2, color=colors[i % 4], label=f"{lot} (labels)")
        table += [{"lot": lot, "hour": h, "occupancy_pct": round(v, 1), "n_images": len(hours[h])} for h, v in zip(hs, m)]
    if per_image and model:
        with open(per_image) as f:
            pr = [r for r in csv.DictReader(f) if r["model"] == model and r.get("hour", "") != ""]
        by_h = defaultdict(list)
        for r in pr:
            tot = int(r["pred_empty"]) + int(r["pred_occupied"])
            if tot:
                by_h[int(r["hour"])].append(int(r["pred_occupied"]) / tot)
        hs = sorted(by_h)
        ax.plot(hs, [100 * np.mean(by_h[h]) for h in hs], "--s", ms=4, color="#343A40", label=f"{model} (predicted)")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("occupied spaces (%)")
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, ncols=2, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(out / "occupancy_by_hour.png", dpi=150)
    with open(out / "occupancy_by_hour.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lot", "hour", "occupancy_pct", "n_images"])
        w.writeheader()
        w.writerows(table)
    print(f"-> {out / 'occupancy_by_hour.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="converted dataset root with manifest.csv")
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    ap.add_argument("--per-image", type=Path)
    ap.add_argument("--model")
    a = ap.parse_args()
    main(a.data, a.out, a.per_image, a.model)
