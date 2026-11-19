"""Render the demo video (mp4, H.264) required for submission.  Owner: Kevin

Recommended (what docs/sample_demo.mp4 shows): one real test day at the new lot as a
time-lapse, baseline vs. ours side by side, with a clock, title card and results card.

    python -m scripts.make_demo_video --images datasets/xsite/images/target_test --glob "*2012-10-11_*" \\
        --weights runs/finetune/weights/best.pt --compare runs/source_only/weights/best.pt \\
        --hold 0.4 --clock \\
        --title "SpotCheck" --subtitle "One unseen day at a new parking lot (PUCPR)" \\
        --outro "Fine-tuning on 100 new-site images" "Count error per image: 6.1 -> 0.9" \\
        --out demo/spotcheck_demo.mp4

Other modes:
    --video clips/lot.mp4            any video file instead of a folder of stills
    (no --compare)                   single model with the counter only

Counts are smoothed with an exponential moving average so the on-screen number
doesn't flicker; it resets whenever a new still photo starts.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from spotcheck.common import list_images
from spotcheck.counting import Counts

TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})")


class EMA:
    def __init__(self, alpha=0.3):
        self.a, self.v = alpha, None

    def __call__(self, x):
        x = np.asarray(x, float)
        self.v = x if self.v is None else self.a * x + (1 - self.a) * self.v
        return self.v


# ---------- text rendering (PIL for clean fonts; falls back to OpenCV) ----------
def _font(size, bold=False):
    try:
        from matplotlib import font_manager
        from PIL import ImageFont

        path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans",
                                                                 weight="bold" if bold else "normal"))
        return ImageFont.truetype(path, size)
    except Exception:
        return None


def put_text(img, text, xy, size=28, color=(240, 240, 240), bold=False, anchor="la", max_w=None):
    f = _font(size, bold)
    while f is not None and max_w and size > 10 and f.getlength(text) > max_w:
        size = int(size * 0.92)
        f = _font(size, bold)
    if f is None:
        cv2.putText(img, text, (int(xy[0]), int(xy[1]) + size), cv2.FONT_HERSHEY_SIMPLEX, size / 32,
                    color[::-1], 2 if bold else 1, cv2.LINE_AA)
        return img
    from PIL import Image, ImageDraw

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(xy, text, font=f, fill=tuple(color), anchor=anchor)
    img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return img


def card(w, h, title=None, lines=(), accent=(42, 114, 232)):
    """Dark title/end card. accent is RGB."""
    img = np.full((h, w, 3), (31, 24, 20), np.uint8)  # BGR of #14181F
    s = w / 1280
    cv2.rectangle(img, (int(80 * s), int(h / 2 - 120 * s)), (int(92 * s), int(h / 2 + 120 * s)), accent[::-1], -1)
    y = h / 2 - (60 * s if lines else 0)
    if title:
        put_text(img, title, (int(120 * s), y), int(64 * s), (244, 242, 237), bold=True, anchor="lm",
                 max_w=w - 200 * s)
        y += 80 * s
    for i, line in enumerate(lines):
        put_text(img, line, (int(120 * s), y + i * 44 * s), int(30 * s), (200, 205, 214), anchor="lm",
                 max_w=w - 200 * s)
    return img


def compose(panels, names, w=1280, clock=None):
    """Place 1-2 panels on a dark 16:9 canvas with readable labels above and the clock below."""
    h = w * 9 // 16
    gap = 24 if len(panels) > 1 else 0
    pw = (w - 48 - gap * (len(panels) - 1)) // len(panels)
    resized = [cv2.resize(p, (pw, int(p.shape[0] * pw / p.shape[1]))) for p in panels]
    ph = max(r.shape[0] for r in resized)
    if ph > h - 150:  # very tall panels: shrink to leave room for labels and clock
        k = (h - 150) / ph
        resized = [cv2.resize(r, (int(r.shape[1] * k), int(r.shape[0] * k))) for r in resized]
        pw, ph = resized[0].shape[1], max(r.shape[0] for r in resized)
    canvas = np.full((h, w, 3), (31, 24, 20), np.uint8)
    total_w = pw * len(resized) + gap * (len(resized) - 1)
    x, y0 = (w - total_w) // 2, (h - ph) // 2 + 10
    colors = [(200, 205, 214), (255, 170, 90)] if len(resized) > 1 else [(255, 170, 90)]
    for r, name, col in zip(resized, names, colors):
        canvas[y0:y0 + r.shape[0], x:x + pw] = r
        put_text(canvas, name, (x + pw // 2, y0 - 22), 26, col, bold=True, anchor="mm", max_w=pw)
        x += pw + gap
    if clock:
        put_text(canvas, clock, (w // 2, y0 + ph + 36), 30, (255, 255, 255), bold=True, anchor="mm")
    return canvas


def stamp(img, watermark=None):
    h, w = img.shape[:2]
    s = w / 1280
    if watermark:
        put_text(img, watermark, (w - int(14 * s), int(h - 14 * s)), int(16 * s), (255, 214, 102), anchor="rd")
    return img


# ---------- frames ----------
def _encode_h264(raw: str, out: str, fps: float):
    """Re-encode OpenCV's mp4v output to H.264 so it plays in PowerPoint/browsers."""
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-preset", "veryfast",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", f"{fps:.2f}", out], check=True)
    else:
        shutil.copy(raw, out)
        print("ffmpeg not found: wrote mp4v; it may not play in PowerPoint.")


def _clock_from_name(name: str):
    m = TS_RE.search(name)
    if not m:
        return None
    h, mi = int(m.group(2)), int(m.group(3))
    return f"{(h - 1) % 12 + 1}:{mi:02d} {'am' if h < 12 else 'pm'}"


def _frame_source(video, images, hold_s, glob=None):
    """Yields (frame, is_new_still, clock_text)."""
    if video:
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        def gen():
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield f, False, None
            cap.release()
        return gen(), fps
    paths = [p for p in list_images(images) if not glob or p.match(glob)]
    if not paths:
        raise SystemExit(f"no images in {images} matching {glob}")
    fps = 10.0

    def gen():
        for p in paths:
            f = cv2.imread(str(p))
            if f is None:
                continue
            for k in range(max(1, int(hold_s * fps))):
                yield f, k == 0, _clock_from_name(p.name)
    return gen(), fps


def _fit(img, max_w):
    h, w = img.shape[:2]
    return cv2.resize(img, (max_w, int(h * max_w / w))) if w > max_w else img


def render_video(video, out, main, compare=None, conf=0.25, every=2, images=None, hold_s=2.0, max_w=1280,
                 glob=None, title=None, subtitle=None, outro=(), clock=False, watermark=None, card_s=3.0,
                 names=("Baseline (source lots only)", "SpotCheck (fine-tuned on new site)")):
    from spotcheck.infer import draw

    frames, fps = _frame_source(video, images, hold_s, glob)
    models = [(main, names[1])]
    if compare:
        models = [(compare, names[0])] + models  # baseline on the left
    state = [{"ema": EMA(), "last": None} for _ in models]
    writer, tmp = None, tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    size = None

    def write(img):
        writer.write(img[: size[1], : size[0]])

    for i, (frame, new_still, clk) in enumerate(frames):
        frame = _fit(frame, max_w)
        panels = []
        for (m, _name), st in zip(models, state):
            if new_still:
                st["ema"] = EMA()
            if st["last"] is None or new_still or (not images and i % every == 0):
                st["last"] = m.predict(frame, conf=conf)
            r = st["last"]
            e, o = st["ema"]([r["counts"]["empty"], r["counts"]["occupied"]])
            boxes, cls, sc = r["_raw"]
            panels.append(draw(frame, boxes, cls, sc, counts=Counts(int(round(e)), int(round(o)))))
        canvas = compose(panels, [n for _, n in models], max_w, clk if clock else None)
        if writer is None:
            h, w = canvas.shape[:2]
            size = (w - w % 2, h - h % 2)
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if title or subtitle:
                lines = [subtitle] if subtitle else []
                if compare:
                    lines.append("Left: baseline   ·   Right: ours   ·   green = empty, red = occupied")
                c = card(w, h, title, lines)
                if watermark:
                    stamp(c, watermark=watermark)
                for _ in range(int(card_s * fps)):
                    write(c)
        stamp(canvas, watermark=watermark)
        write(canvas)
    if writer is None:
        raise SystemExit("no frames read")
    if outro:
        c = card(size[0], size[1], outro[0], list(outro[1:]), accent=(232, 114, 42))
        if watermark:
            stamp(c, watermark=watermark)
        for _ in range(int(card_s * fps)):
            write(c)
    writer.release()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    _encode_h264(tmp, out, fps)
    Path(tmp).unlink(missing_ok=True)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    from spotcheck.infer import SpotChecker

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--images")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--compare", help="baseline weights, shown on the left")
    ap.add_argument("--out", default="demo/demo.mp4")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--every", type=int, default=2, help="video mode: run detection every N frames")
    ap.add_argument("--hold", type=float, default=2.0, help="seconds per still image")
    ap.add_argument("--glob", help='only images whose name matches, e.g. "*2012-10-11_*" for one day')
    ap.add_argument("--clock", action="store_true", help="show the time parsed from file names")
    ap.add_argument("--title")
    ap.add_argument("--subtitle")
    ap.add_argument("--outro", nargs="+", default=[], help="end card: first item is the headline")
    ap.add_argument("--watermark")
    ap.add_argument("--card-seconds", type=float, default=3.0)
    a = ap.parse_args()
    render_video(a.video, a.out, SpotChecker(a.weights), SpotChecker(a.compare) if a.compare else None,
                 a.conf, a.every, a.images, a.hold, glob=a.glob, title=a.title, subtitle=a.subtitle,
                 outro=a.outro, clock=a.clock, watermark=a.watermark, card_s=a.card_seconds)
