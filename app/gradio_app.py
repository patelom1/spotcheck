"""Interactive web demo (image, video, model comparison).  Owner: Kevin

    python -m app.gradio_app --weights runs/finetune/weights/best.pt \
        --compare runs/source_only/weights/best.pt
    # add --share for a public link (handy for demoing from a phone)
"""
from __future__ import annotations

import argparse
import tempfile

import cv2
import gradio as gr

from spotcheck.infer import SpotChecker
from scripts.make_demo_video import render_video


def build(weights: str, compare: str | None = None):
    main = SpotChecker(weights)
    other = SpotChecker(compare) if compare else None

    def run_image(img_rgb, conf):
        if img_rgb is None:
            return None, None, ""
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        r = main.predict(bgr, conf=conf)
        vis = cv2.cvtColor(main.annotate(bgr, r, title="Fine-tuned on new site"), cv2.COLOR_BGR2RGB)
        cmp = None
        if other:
            r2 = other.predict(bgr, conf=conf)
            cmp = cv2.cvtColor(other.annotate(bgr, r2, title="Source lots only"), cv2.COLOR_BGR2RGB)
        c = r["counts"]
        md = (f"### {c['empty']} open of {c['total']} spaces\n"
              f"Occupancy **{100 * c['occupancy_rate']:.0f}%** · inference {r['inference_ms']:.0f} ms")
        return vis, cmp, md

    def run_video(path, conf, every):
        if not path:
            return None
        out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        render_video(path, out, main, conf=conf, every=int(every))
        return out

    with gr.Blocks(title="SpotCheck", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# SpotCheck — parking space occupancy\n"
                    "Green = empty space, red = occupied. YOLOv8 trained on two PKLot lots, fine-tuned on a new one.")
        with gr.Tab("Image"):
            with gr.Row():
                inp = gr.Image(label="Lot photo", type="numpy")
                with gr.Column():
                    out = gr.Image(label="SpotCheck")
                    stats = gr.Markdown()
            if other:
                cmp = gr.Image(label="Baseline: trained on source lots only")
            else:
                cmp = gr.Image(visible=False)
            conf = gr.Slider(0.05, 0.9, 0.25, step=0.05, label="Confidence threshold")
            gr.Button("Detect", variant="primary").click(run_image, [inp, conf], [out, cmp, stats])
        with gr.Tab("Video"):
            vin = gr.Video(label="Walk-by or drone video")
            with gr.Row():
                vconf = gr.Slider(0.05, 0.9, 0.25, step=0.05, label="Confidence")
                every = gr.Slider(1, 10, 2, step=1, label="Detect every N frames")
            vout = gr.Video(label="Annotated")
            gr.Button("Process video", variant="primary").click(run_video, [vin, vconf, every], vout)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--compare")
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    build(a.weights, a.compare).launch(share=a.share)
