"""Download datasets through the Roboflow API.  Owner: Ridham

Two uses:
  * our own labeled campus project (after labeling in the Roboflow UI)
  * the public PKLot mirror on Roboflow Universe, already in YOLO format,
    as a fast alternative to running pklot.py on the 12k-image original

Set ROBOFLOW_API_KEY (Roboflow -> Settings -> API Keys). Never commit the key.

Usage:
    python -m spotcheck.data.download_roboflow --workspace <ws> --project <proj> \
        --version 1 --out roboflow_export
"""
from __future__ import annotations

import argparse
import os


def download(workspace: str, project: str, version: int, out: str, fmt: str = "yolov8"):
    try:
        from roboflow import Roboflow
    except ImportError as e:
        raise SystemExit("pip install roboflow") from e
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("Set ROBOFLOW_API_KEY first.")
    rf = Roboflow(api_key=key)
    ds = rf.workspace(workspace).project(project).version(version).download(fmt, location=out)
    print(f"downloaded to {ds.location}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--out", default="roboflow_export")
    a = ap.parse_args()
    download(a.workspace, a.project, a.version, a.out)
