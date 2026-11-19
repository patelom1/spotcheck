"""Figures and tables for the slides/report.  Owner: Krishna

    python -m spotcheck.plots --metrics results/test_metrics.json --out results/figures
    python -m spotcheck.plots --qualitative runs/source_only runs/finetune \
        --images datasets/xsite/images/target_test --out results/figures

Produces:
  map_by_model.png         mAP@0.5 and mAP@0.5:0.95 per experiment
  mae_by_weather.png       empty-space count MAE per weather condition
  mae_by_time_of_day.png   empty-space count MAE per time of day
  label_efficiency.png     count error vs number of new-site training images
  speed_vs_accuracy.png    FPS vs mAP@0.5 (model-size comparison)
  results_table.md         markdown table to paste into slides / README
  qualitative.png          same images, one row per model (domain-gap visual)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = ["#4C6EF5", "#F59F00", "#12B886", "#E64980", "#7950F2", "#868E96"]
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})

PRETTY = {"source_only": "Source only", "target_only": "New site only",
          "finetune": "Source → FT (ours)", "combined": "Source + new site",
          "finetune_s": "FT (YOLOv8s)", "source_only_s": "Source only (s)",
          "finetune_25": "FT, 25 imgs", "finetune_50": "FT, 50 imgs", "finetune_200": "FT, 200 imgs"}
MAIN = ["source_only", "target_only", "finetune", "combined", "finetune_s"]
EFFICIENCY = {"source_only": 0, "finetune_25": 25, "finetune_50": 50, "finetune": 100, "finetune_200": 200}
TOD_ORDER = ["morning", "midday", "afternoon", "evening"]


def pretty(n):
    return PRETTY.get(n, n)


def main_models(res):
    names = [n for n in MAIN if n in res]
    return names or list(res)


def bar_map(res, out):
    names = main_models(res)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    a = [res[n]["all"]["map50"] for n in names]
    b = [res[n]["all"]["map50_95"] for n in names]
    ax.bar(x - 0.2, a, 0.4, label="mAP@0.5", color=PALETTE[0])
    ax.bar(x + 0.2, b, 0.4, label="mAP@0.5:0.95", color=PALETTE[1])
    for xi, v in zip(x, a):
        ax.text(xi - 0.2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, [pretty(n) for n in names], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score on new-site test")
    ax.legend(frameon=False, ncols=2, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(out / "map_by_model.png")
    plt.close(fig)


def bar_group(res, out, field):
    names = main_models(res)
    conds = sorted({k.split("=", 1)[1] for n in names for k in res[n] if k.startswith(field + "=")},
                   key=lambda c: (TOD_ORDER.index(c) if c in TOD_ORDER else 99, c))
    if not conds:
        return
    x = np.arange(len(conds))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, n in enumerate(names):
        vals = [res[n].get(f"{field}={c}", {}).get("count_mae_image", np.nan) for c in conds]
        ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, w, label=pretty(n), color=PALETTE[i % 6])
    ax.set_xticks(x, conds)
    ax.set_ylabel("empty-space count MAE (per image)")
    ax.legend(frameon=False, fontsize=9, ncols=min(3, len(names)), loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(out / f"mae_by_{field}.png")
    plt.close(fig)


def label_efficiency(res, out):
    pts = sorted((k, res[n]["all"]["count_mae_image"]) for n, k in EFFICIENCY.items() if n in res)
    if len(pts) < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    xs, ys = zip(*pts)
    ax.plot(xs, ys, "-o", color=PALETTE[0], lw=2)
    for x, y in pts:
        ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    if "target_only" in res:
        ax.axhline(res["target_only"]["all"]["count_mae_image"], ls="--", color=PALETTE[1], label="new site only (100 imgs)")
        ax.legend(frameon=False, loc="upper right")
    ax.set_xlabel("labeled new-site images used for fine-tuning")
    ax.set_ylabel("empty-space count error / image")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out / "label_efficiency.png")
    plt.close(fig)


def speed_accuracy(res, out):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for i, n in enumerate(main_models(res)):
        a = res[n]["all"]
        ax.scatter(a["fps"], a["map50"], s=70, color=PALETTE[i % 6])
        ax.annotate(pretty(n), (a["fps"], a["map50"]), xytext=(5, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("FPS")
    ax.set_ylabel("mAP@0.5")
    fig.tight_layout()
    fig.savefig(out / "speed_vs_accuracy.png")
    plt.close(fig)


def table(res, out):
    hdr = "| Model | mAP@0.5 | mAP@0.5:0.95 | P (empty) | R (empty) | Count MAE / img | Count MAE / lot | FPS |"
    lines = [hdr, "|" + "---|" * 8]
    for n, r in res.items():
        a = r["all"]
        pe = a["per_class"].get("empty", {})
        lines.append(f"| {pretty(n)} | {a['map50']:.3f} | {a['map50_95']:.3f} | {pe.get('precision', float('nan')):.3f} "
                     f"| {pe.get('recall', float('nan')):.3f} | {a['count_mae_image']:.2f} | {a['count_mae_lot']:.2f} "
                     f"| {a['fps']:.1f} |")
    (out / "results_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def qualitative(models, images_dir, out, n=3, conf=0.25):
    import cv2
    from ultralytics import YOLO

    from spotcheck.common import list_images
    from spotcheck.infer import draw, train_imgsz

    imgs = list_images(images_dir)
    if not imgs:
        return
    pick = [imgs[i] for i in np.linspace(0, len(imgs) - 1, min(n, len(imgs))).astype(int)]
    fig, axes = plt.subplots(len(models), len(pick), figsize=(4 * len(pick), 3 * len(models)), squeeze=False)
    for r, m in enumerate(models):
        w = Path(m) / "weights" / "best.pt" if Path(m).is_dir() else Path(m)
        model = YOLO(str(w))
        imgsz = train_imgsz(model)
        for c, p in enumerate(pick):
            res = model.predict(str(p), imgsz=imgsz, conf=conf, verbose=False, agnostic_nms=True)[0]
            vis = draw(cv2.imread(str(p)), res.boxes.xyxy.cpu().numpy(),
                       res.boxes.cls.cpu().numpy().astype(int), res.boxes.conf.cpu().numpy(), hud=False)
            axes[r, c].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].grid(False)
        axes[r, 0].set_ylabel(pretty(Path(m).name), fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "qualitative.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", type=Path)
    ap.add_argument("--qualitative", nargs="+")
    ap.add_argument("--images", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.metrics:
        res = json.loads(a.metrics.read_text())
        bar_map(res, a.out)
        bar_group(res, a.out, "weather")
        bar_group(res, a.out, "time_of_day")
        label_efficiency(res, a.out)
        speed_accuracy(res, a.out)
        table(res, a.out)
    if a.qualitative:
        qualitative(a.qualitative, a.images, a.out)
    print(f"figures -> {a.out}")
