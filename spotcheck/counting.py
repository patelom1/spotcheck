"""Turn detections into space counts, per image and per lot.  Owner: Krishna

A space can't be both empty and occupied, so overlapping boxes of different
classes are resolved with class-agnostic NMS (keep the more confident one).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from spotcheck.common import EMPTY, OCCUPIED


@dataclass
class Counts:
    empty: int = 0
    occupied: int = 0

    @property
    def total(self) -> int:
        return self.empty + self.occupied

    @property
    def occupancy_rate(self) -> float:
        return self.occupied / self.total if self.total else 0.0

    def __add__(self, o: "Counts") -> "Counts":
        return Counts(self.empty + o.empty, self.occupied + o.occupied)

    def as_dict(self) -> dict:
        return {"empty": self.empty, "occupied": self.occupied, "total": self.total,
                "occupancy_rate": round(self.occupancy_rate, 4)}


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between (N,4) and (M,4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def agnostic_nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.6) -> np.ndarray:
    """Indices kept after class-agnostic NMS."""
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < iou_thr]
    return np.asarray(keep, dtype=int)


def count_detections(boxes, classes, scores, conf: float = 0.25, iou_thr: float = 0.6) -> Counts:
    boxes, classes, scores = map(np.asarray, (boxes, classes, scores))
    if boxes.size == 0:
        return Counts()
    m = scores >= conf
    boxes, classes, scores = boxes[m].reshape(-1, 4), classes[m], scores[m]
    keep = agnostic_nms(boxes, scores, iou_thr) if len(boxes) else []
    cls = classes[keep] if len(keep) else np.array([])
    return Counts(int((cls == EMPTY).sum()), int((cls == OCCUPIED).sum()))


def count_labels(rows) -> Counts:
    """Ground-truth counts from YOLO label rows [(cls, cx, cy, w, h), ...]."""
    cls = [r[0] for r in rows]
    return Counts(cls.count(EMPTY), cls.count(OCCUPIED))


@dataclass
class LotAggregator:
    """Sum counts over all images that cover a lot during one session."""
    per_key: dict = field(default_factory=lambda: defaultdict(Counts))

    def add(self, lot: str, session: str, counts: Counts):
        self.per_key[(lot, session)] = self.per_key[(lot, session)] + counts

    def table(self) -> dict:
        return {f"{lot}|{sess}": c.as_dict() for (lot, sess), c in sorted(self.per_key.items())}
