"""Blur people and license plates before images leave our machines.  Owner: Ridham

People: COCO-pretrained YOLOv8 `person` detections.
Plates: OpenCV's bundled Haar cascade, searched only inside detected vehicles
(car/motorcycle/bus/truck) to cut false positives on signs and stripes.
Run this on raw photos BEFORE uploading to Roboflow.

Usage:
    python -m spotcheck.data.privacy --src raw_photos --dst blurred_photos
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from spotcheck.common import list_images

PERSON, VEHICLES = 0, {2, 3, 5, 7}


def blur_region(img, x1, y1, x2, y2, k=51):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    k = max(3, (min(k, (x2 - x1) | 1, (y2 - y1) | 1)) | 1)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (k, k), 0)


def anonymize(img, model, plate_cascade, conf=0.25):
    res = model.predict(img, conf=conf, verbose=False)[0]
    n_people = n_plates = 0
    for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.tolist(), res.boxes.cls.int().tolist()):
        if c == PERSON:
            blur_region(img, x1, y1, x2, y2, k=71)
            n_people += 1
        elif c in VEHICLES:
            crop = cv2.cvtColor(img[int(y1):int(y2), int(x1):int(x2)], cv2.COLOR_BGR2GRAY)
            if crop.size == 0:
                continue
            for (px, py, pw, ph) in plate_cascade.detectMultiScale(crop, 1.1, 4, minSize=(16, 6)):
                pad = int(0.15 * pw)
                blur_region(img, x1 + px - pad, y1 + py - pad, x1 + px + pw + pad, y1 + py + ph + pad)
                n_plates += 1
    return img, n_people, n_plates


def main(src: Path, dst: Path, weights: str = "yolov8n.pt"):
    from ultralytics import YOLO

    model = YOLO(weights)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    tot_p = tot_l = 0
    for p in list_images(src):
        img = cv2.imread(str(p))
        if img is None:
            continue
        img, n_p, n_l = anonymize(img, model, cascade)
        tot_p, tot_l = tot_p + n_p, tot_l + n_l
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img)
    print(f"blurred {tot_p} people and {tot_l} plate candidates -> {dst}")
    print("Spot-check a few images by eye: the plate cascade misses some plates.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--weights", default="yolov8n.pt")
    a = ap.parse_args()
    main(a.src, a.dst, a.weights)
