"""Evaluate trained models on the new-site test set.  Owner: Krishna

For each model it reports, on one split of one dataset:
  * detection: mAP@0.5, mAP@0.5:0.95, per-class precision / recall / AP50
  * counting:  MAE of empty-space counts per image, and per lot (summed per session)
  * speed:     mean inference ms/image and FPS (after warm-up)
  * breakdown: all of the above per weather and per time of day (from manifest.csv)

    python -m spotcheck.evaluate --models runs/source_only runs/finetune \
        --data datasets/xsite/target_100.yaml --split test --out results/

Writes results/<split>_metrics.json and results/<split>_metrics.csv; plots.py reads these.
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from spotcheck.common import CLASS_NAMES, list_images, read_yolo_labels, label_path_for
from spotcheck.counting import LotAggregator, count_detections, count_labels


def resolve_weights(m: str) -> Path:
    p = Path(m)
    if p.is_dir():
        p = p / "weights" / "best.pt"
    if not p.exists():
        raise SystemExit(f"weights not found: {p}")
    return p


def split_images(data_yaml: Path, split: str) -> list[Path]:
    d = yaml.safe_load(data_yaml.read_text())
    root = Path(d.get("path", data_yaml.parent))
    entry = d[split]
    entries = entry if isinstance(entry, list) else [entry]
    imgs = []
    for e in entries:
        e = Path(e) if Path(e).is_absolute() else root / e
        imgs += list_images(e) if e.is_dir() else [Path(l) for l in e.read_text().split()]
    return imgs


def load_manifest(data_yaml: Path) -> dict:
    d = yaml.safe_load(data_yaml.read_text())
    mf = Path(d.get("path", data_yaml.parent)) / "manifest.csv"
    if not mf.exists():
        return {}
    with open(mf) as f:
        return {Path(r["image"]).stem: r for r in csv.DictReader(f)}


def subset_yaml(data_yaml: Path, images: list[Path], tmpdir: Path, tag: str) -> Path:
    """data.yaml whose `val` is just these images (Ultralytics accepts a .txt list)."""
    d = yaml.safe_load(data_yaml.read_text())
    lst = tmpdir / f"{tag}.txt"
    lst.write_text("\n".join(str(p.resolve()) for p in images) + "\n")
    out = tmpdir / f"{tag}.yaml"
    out.write_text(yaml.safe_dump({"path": str(tmpdir), "train": str(lst), "val": str(lst),
                                   "names": d["names"]}))
    return out


def detection_metrics(model, data: Path, imgsz: int, device=None) -> dict:
    r = model.val(data=str(data), split="val", imgsz=imgsz, plots=False, verbose=False,
                  device=device, batch=8)
    box = r.box
    per_class = {}
    for i, ci in enumerate(box.ap_class_index.tolist()):
        per_class[CLASS_NAMES[ci]] = {"precision": float(box.p[i]), "recall": float(box.r[i]),
                                      "ap50": float(box.ap50[i])}
    return {"map50": float(box.map50), "map50_95": float(box.map), "per_class": per_class}


def counting_and_speed(model, images, manifest, imgsz, conf, device=None, warmup=3):
    for p in images[:warmup]:
        model.predict(str(p), imgsz=imgsz, conf=conf, verbose=False, device=device)
    rows, times = [], []
    agg_pred, agg_gt = LotAggregator(), LotAggregator()
    for p in images:
        t0 = time.perf_counter()
        r = model.predict(str(p), imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
        times.append(time.perf_counter() - t0)
        b = r.boxes
        pred = count_detections(b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int),
                                b.conf.cpu().numpy(), conf=conf)
        gt = count_labels(read_yolo_labels(label_path_for(p)))
        m = manifest.get(p.stem, {})
        lot, sess = m.get("lot", "unknown"), m.get("session", "unknown")
        agg_pred.add(lot, sess, pred)
        agg_gt.add(lot, sess, gt)
        rows.append({"image": p.name, "weather": m.get("weather", ""), "time_of_day": m.get("time_of_day", ""),
                     "hour": m.get("hour", ""), "lot": lot,
                     "gt_empty": gt.empty, "pred_empty": pred.empty,
                     "gt_occupied": gt.occupied, "pred_occupied": pred.occupied})
    err = [abs(r["gt_empty"] - r["pred_empty"]) for r in rows]
    lot_err = [abs(agg_pred.per_key[k].empty - agg_gt.per_key[k].empty) for k in agg_gt.per_key]
    ms = 1000 * float(np.mean(times)) if times else float("nan")
    return rows, {
        "count_mae_image": float(np.mean(err)) if err else float("nan"),
        "count_mae_lot": float(np.mean(lot_err)) if lot_err else float("nan"),
        "exact_count_acc": float(np.mean([e == 0 for e in err])) if err else float("nan"),
        "ms_per_image": ms, "fps": 1000 / ms if ms else float("nan"),
    }


def evaluate(models, data_yaml: Path, split: str, out: Path, imgsz=None, conf=0.25, device=None,
             group_by=("weather", "time_of_day")):
    from ultralytics import YOLO

    out.mkdir(parents=True, exist_ok=True)
    images = split_images(data_yaml, split)
    if not images:
        raise SystemExit(f"No images in split '{split}' of {data_yaml}")
    manifest = load_manifest(data_yaml)
    sets = {"all": images}
    for field in group_by:
        groups = defaultdict(list)
        for p in images:
            groups[manifest.get(p.stem, {}).get(field) or "unknown"].append(p)
        if len(groups) > 1:
            sets.update({f"{field}={k}": v for k, v in sorted(groups.items())})

    results, per_image = {}, []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        yamls = {k: subset_yaml(data_yaml, v, td, k.replace("=", "_")) for k, v in sets.items()}
        for m in models:
            w = resolve_weights(m)
            name = w.parent.parent.name if w.parent.name == "weights" else w.stem
            model = YOLO(str(w))
            from spotcheck.infer import train_imgsz
            isz = imgsz or train_imgsz(model)
            results[name] = {}
            for k, imgs in sets.items():
                det = detection_metrics(model, yamls[k], isz, device)
                rows, cnt = counting_and_speed(model, imgs, manifest, isz, conf, device)
                results[name][k] = {"n_images": len(imgs), "imgsz": isz, **det, **cnt}
                if k == "all":
                    per_image += [{"model": name, **r} for r in rows]
            a = results[name]["all"]
            print(f"{name:22s} mAP50={a['map50']:.3f} mAP50-95={a['map50_95']:.3f} "
                  f"countMAE={a['count_mae_image']:.2f} FPS={a['fps']:.1f}")

    (out / f"{split}_metrics.json").write_text(json.dumps(results, indent=2))
    flat = []
    for name, by_set in results.items():
        for k, v in by_set.items():
            row = {"model": name, "subset": k, **{x: v[x] for x in v if x != "per_class"}}
            for c, pc in v["per_class"].items():
                row.update({f"{c}_{kk}": vv for kk, vv in pc.items()})
            flat.append(row)
    keys = sorted({k for r in flat for k in r}, key=lambda k: (k not in ("model", "subset"), k))
    with open(out / f"{split}_metrics.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(flat)
    if per_image:
        with open(out / f"{split}_per_image.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
            wr.writeheader()
            wr.writerows(per_image)
    print(f"saved -> {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=Path("datasets/xsite/target_100.yaml"))
    ap.add_argument("--group-by", nargs="*", default=["weather", "time_of_day"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--imgsz", type=int, default=None, help="default: the size each model was trained at")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device")
    a = ap.parse_args()
    evaluate(a.models, a.data, a.split, a.out, a.imgsz, a.conf, a.device, a.group_by)
