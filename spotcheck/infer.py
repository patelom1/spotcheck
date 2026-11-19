"""Inference wrapper shared by the API, web app and demo video.  Owner: Kevin

    from spotcheck.infer import SpotChecker
    sc = SpotChecker("runs/finetune/weights/best.pt")
    result = sc.predict(bgr_image)      # dict with counts + detections
    vis = sc.annotate(bgr_image, result)
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from spotcheck.common import CLASS_NAMES, COLORS, EMPTY
from spotcheck.counting import Counts, count_detections


def draw(img, boxes, classes, scores, hud=True, counts: Counts | None = None, title: str | None = None):
    img = img.copy()
    h, w = img.shape[:2]
    t = max(1, round(min(h, w) / 400))
    for (x1, y1, x2, y2), c in zip(np.asarray(boxes).astype(int), classes):
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[int(c)], t + 1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # center dot reads well even when boxes touch
        cv2.circle(img, (cx, cy), 3 * t + 1, COLORS[int(c)], -1)
    if hud:
        counts = counts or Counts(int(np.sum(np.asarray(classes) == EMPTY)),
                                  int(np.sum(np.asarray(classes) != EMPTY)))
        draw_hud(img, counts, title)
    return img


def draw_hud(img, counts: Counts, title: str | None = None):
    h, w = img.shape[:2]
    s = max(0.7, w / 1000)
    pw, ph = int(330 * s), int((120 if title else 92) * s)
    x0, y0 = int(16 * s), int(16 * s)
    sub = img[y0:y0 + ph, x0:x0 + pw]
    img[y0:y0 + ph, x0:x0 + pw] = cv2.addWeighted(sub, 0.25, np.zeros_like(sub), 0.75, 0)
    y = y0 + int(30 * s)
    if title:
        cv2.putText(img, title, (x0 + int(12 * s), y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (220, 220, 220), 1, cv2.LINE_AA)
        y += int(30 * s)
    cv2.putText(img, f"{counts.empty} open", (x0 + int(12 * s), y + int(6 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1 * s, COLORS[EMPTY], 2, cv2.LINE_AA)
    cv2.putText(img, f"/ {counts.total} spaces", (x0 + int(150 * s), y + int(6 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (235, 235, 235), 1, cv2.LINE_AA)
    bar_y, bar_w = y + int(26 * s), pw - int(24 * s)
    cv2.rectangle(img, (x0 + int(12 * s), bar_y), (x0 + int(12 * s) + bar_w, bar_y + int(10 * s)), (70, 70, 70), -1)
    fill = int(bar_w * counts.occupancy_rate)
    cv2.rectangle(img, (x0 + int(12 * s), bar_y), (x0 + int(12 * s) + fill, bar_y + int(10 * s)), (40, 40, 220), -1)
    return img


def train_imgsz(model, default: int = 960) -> int:
    """Image size the checkpoint was trained at; predicting at another size hurts accuracy."""
    try:
        v = (getattr(model, "ckpt", None) or {}).get("train_args", {}).get("imgsz") or model.overrides.get("imgsz")
        return int(v[0] if isinstance(v, (list, tuple)) else v) if v else default
    except Exception:
        return default


class SpotChecker:
    def __init__(self, weights: str, imgsz: int | None = None, conf: float = 0.25, iou: float = 0.6, device=None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        imgsz = imgsz or train_imgsz(self.model)
        self.weights, self.imgsz, self.conf, self.iou, self.device = weights, imgsz, conf, iou, device

    def predict(self, img: np.ndarray, conf: float | None = None) -> dict:
        conf = self.conf if conf is None else conf
        t0 = time.perf_counter()
        r = self.model.predict(img, imgsz=self.imgsz, conf=conf, iou=self.iou, agnostic_nms=True,
                               verbose=False, device=self.device)[0]
        ms = 1000 * (time.perf_counter() - t0)
        boxes = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        sc = r.boxes.conf.cpu().numpy()
        counts = count_detections(boxes, cls, sc, conf=conf, iou_thr=self.iou)
        return {
            "counts": counts.as_dict(),
            "detections": [{"box": [round(float(v), 1) for v in b], "label": CLASS_NAMES[c],
                            "confidence": round(float(s), 3)} for b, c, s in zip(boxes, cls, sc)],
            "inference_ms": round(ms, 1),
            "image_size": list(img.shape[:2][::-1]),
            "_raw": (boxes, cls, sc),
        }

    def annotate(self, img, result: dict, title: str | None = None):
        boxes, cls, sc = result["_raw"]
        c = result["counts"]
        return draw(img, boxes, cls, sc, counts=Counts(c["empty"], c["occupied"]), title=title)


def public(result: dict) -> dict:
    """Strip private numpy arrays before returning JSON."""
    return {k: v for k, v in result.items() if not k.startswith("_")}
