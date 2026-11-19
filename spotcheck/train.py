"""Train one experiment from configs/experiments.yaml.  Owner: Om

    python -m spotcheck.train --exp source_only
    python -m spotcheck.train --exp finetune --epochs 5      # quick override
    python -m spotcheck.train --list

Runs land in runs/<exp>/ with weights/best.pt. A copy of the resolved config is
saved next to them so every number in the report traces back to its settings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from spotcheck.common import REPO_ROOT

CONFIG = REPO_ROOT / "configs" / "experiments.yaml"
RUNS = REPO_ROOT / "runs"
TRAIN_KEYS = {"imgsz", "epochs", "batch", "patience", "optimizer", "seed", "hsv_v", "hsv_s",
              "degrees", "fliplr", "flipud", "mosaic", "close_mosaic", "lr0", "device", "workers",
              "warmup_epochs"}


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text())


def resolve(exp: str, cfg: dict, overrides: dict | None = None) -> dict:
    if exp not in cfg["experiments"]:
        raise SystemExit(f"Unknown experiment '{exp}'. Options: {list(cfg['experiments'])}")
    spec = {**cfg.get("defaults", {}), **cfg["experiments"][exp], **(overrides or {})}
    init = spec["init"]
    if init in cfg["experiments"]:  # chain from another run
        best = RUNS / init / "weights" / "best.pt"
        if not best.exists():
            raise SystemExit(f"'{exp}' starts from '{init}', but {best} doesn't exist. Train '{init}' first.")
        spec["init"] = str(best)
    data = Path(spec["data"])
    spec["data"] = str(data if data.is_absolute() else REPO_ROOT / data)
    return spec


def train(exp: str, overrides: dict | None = None):
    from ultralytics import YOLO

    spec = resolve(exp, load_config(), overrides)
    print(json.dumps(spec, indent=2))
    model = YOLO(spec["init"])
    kwargs = {k: v for k, v in spec.items() if k in TRAIN_KEYS}
    model.train(data=spec["data"], project=str(RUNS), name=exp, exist_ok=True, plots=True, **kwargs)
    out = RUNS / exp
    out.mkdir(parents=True, exist_ok=True)
    (out / "spotcheck_config.json").write_text(json.dumps(spec, indent=2))
    print(f"done -> {out / 'weights' / 'best.pt'}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--device", help="e.g. 0 or cpu")
    a = ap.parse_args()
    if a.list or not a.exp:
        for name, s in load_config()["experiments"].items():
            print(f"{name:20s} data={s['data']}  init={s['init']}")
    else:
        ov = {k: v for k, v in dict(epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device).items()
              if v is not None}
        train(a.exp, ov)
