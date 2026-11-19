"""Write a data.yaml that trains on several converted datasets at once.  Owner: Ridham

Ultralytics accepts lists of image dirs, so no files are copied.
Val/test come from the --eval dataset: the domain we report on.
Each --train root must have images/train (pklot.py, cnrpark.py, campus.py outputs).
For the cross-site setup, crosssite.py already writes combined_<N>.yaml; to add CNRPark-EXT
there, append its images/train path to that file's `train:` list.

Usage:
    python -m spotcheck.data.combine --train datasets/pklot datasets/cnrpark \
        --eval datasets/pklot --out datasets/public.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from spotcheck.common import CLASS_NAMES


def write_combined(train_roots: list[Path], eval_root: Path, out: Path):
    tr = [str((r / "images" / "train").resolve()) for r in train_roots]
    ev = eval_root.resolve()
    lines = ["train:"] + [f"  - {t}" for t in tr] + [
        f"val: {ev / 'images' / 'val'}", f"test: {ev / 'images' / 'test'}", "names:"
    ] + [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, nargs="+", required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    write_combined(a.train, a.eval, a.out)
