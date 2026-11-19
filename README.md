# SpotCheck: Does a Parking-Space Detector Transfer to a New Lot?

CS 4391 Foundations of Computer Vision, Fall 2026 — **Ridham Gabani, Om Patel, Krishna Chundru, Kevin Gajera**

SpotCheck finds every parking space in a photo of a lot, classifies it as **empty** or
**occupied**, and reports how many are open. Our motivating use case is a campus like UT Dallas:
a lot with its own camera angle and layout and no existing labels. We study that with real data:
a COCO-pretrained YOLOv8 is trained on two PKLot lots (UFPR04, UFPR05) and moved to a third,
unseen lot (PUCPR). We measure the domain gap and how many labeled new-site images it takes to
close it, broken down by real weather (sunny / cloudy / rainy) and time of day.

> **Scope note.** Our proposal planned to collect UTD photos. Because of time, we instead
> use the held-out PKLot lot as the new site. All images and labels are real, and nothing is
> synthetic except the CPU smoke test. We told the instructor about this change.

```
PKLot (3 real lots, photos every 5 min, 30 days)
  ├─ UFPR04 + UFPR05 ─► train "source" model (E1)
  └─ PUCPR (new site) ─► N labeled images ─► fine-tune (E3) ─► test on unseen PUCPR days
                                                                  │
            counts per image / per day, by weather and hour ◄─────┘
```

## Repository map

| Area | Path | Owner |
|---|---|---|
| Data: PKLot conversion, cross-site split, time/weather metadata | `spotcheck/data/pklot.py`, `crosssite.py`, `cnrpark.py`, `combine.py` | Ridham |
| Experiments and training | `configs/experiments.yaml`, `spotcheck/train.py` | Om |
| Counting, evaluation, figures, occupancy-by-hour | `spotcheck/counting.py`, `evaluate.py`, `plots.py`, `traffic.py` | Krishna |
| Inference, REST API, web demo, demo video | `spotcheck/infer.py`, `app/`, `scripts/make_demo_video.py` | Kevin |
| Tests | `tests/` | everyone |

Optional, if you collect any UTD photos later: `spotcheck/data/campus.py` (Roboflow export → split) and
`spotcheck/data/privacy.py` (blur people and plates first).

Schedule and git workflow: [`docs/TEAM_PLAN.md`](docs/TEAM_PLAN.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```
**Colab:** clone, `pip install -r requirements.txt`, mount Drive, and symlink `datasets/` and `runs/` into Drive
(`!ln -s /content/drive/MyDrive/spotcheck/datasets datasets`) so runs survive disconnects.

## Pipeline

### 0. Smoke test (CPU, ~5 min, fake data — not a result)
```bash
python -m scripts.make_synthetic_dataset --out datasets/synthetic
python -m spotcheck.train --exp smoke --device cpu
```

### 1. Data (Ridham)
Download PKLot (the full-image version with XML labels, several GB) from the official UFPR page or a Kaggle mirror and unzip it so that
`PKLot/PKLot/{PUCPR,UFPR04,UFPR05}/{Sunny,Cloudy,Rainy}/<date>/*.jpg|xml` exists.
```bash
python -m spotcheck.data.crosssite --root /data/PKLot/PKLot --target PUCPR --out datasets/xsite
```
This splits by capture **day** (no near-duplicate frames leak into test), and writes `target_<N>.yaml` for
N = 25, 50, 100, 200 new-site training images (nested subsets) plus a manifest with weather, hour, and time of day.

### 2. Training (Om)
```bash
python -m spotcheck.train --list
python -m spotcheck.train --exp source_only      # E1  domain-gap baseline
python -m spotcheck.train --exp target_only      # E2  100 new-site images, no source data
python -m spotcheck.train --exp finetune         # E3  main method (E1 -> 100 new-site images)
python -m spotcheck.train --exp combined         # E4  source + 100 new-site images together
python -m spotcheck.train --exp source_only_s && python -m spotcheck.train --exp finetune_s    # E5 size
python -m spotcheck.train --exp finetune_25 && python -m spotcheck.train --exp finetune_50 \
  && python -m spotcheck.train --exp finetune_200                                             # label efficiency
```

### 3. Evaluation (Krishna)
```bash
python -m spotcheck.evaluate --models runs/source_only runs/target_only runs/finetune runs/combined runs/finetune_s \
    runs/finetune_25 runs/finetune_50 runs/finetune_200 --data datasets/xsite/target_100.yaml --split test --out results
python -m spotcheck.plots --metrics results/test_metrics.json --out results/figures
python -m spotcheck.plots --qualitative runs/source_only runs/finetune --images datasets/xsite/images/target_test --out results/figures
python -m spotcheck.traffic --data datasets/xsite --per-image results/test_per_image.csv --model finetune --out results/figures
```
Metrics: mAP@0.5, mAP@0.5:0.95, per-class P/R, empty-space count MAE (per image and per capture day), FPS,
each broken down by weather and time of day. `traffic.py` plots real occupancy by hour and whether the model tracks it.

### 4. Demo (Kevin)
```bash
# time-lapse of one real day at the new site, baseline vs ours side by side (file names are timestamps, so it plays in order)
# docs/sample_demo.mp4 shows the intended look (synthetic placeholder images)
python -m scripts.make_demo_video --images datasets/xsite/images/target_test --glob "*<one test day>_*" \
    --weights runs/finetune/weights/best.pt --compare runs/source_only/weights/best.pt --hold 0.45 --clock \
    --title "SpotCheck" --subtitle "One unseen day at a new parking lot (PUCPR), 7 am to 6 pm" \
    --outro "Fine-tuning on 100 new-site images" "Count error per image: <E1> → <E3>" "mAP@0.5: <E1> → <E3>  ·  <FPS> FPS" \
    --out demo/spotcheck_demo.mp4

# interactive web demo
python -m app.gradio_app --weights runs/finetune/weights/best.pt --compare runs/source_only/weights/best.pt

# REST API (Swagger at /docs)
SPOTCHECK_WEIGHTS=runs/finetune/weights/best.pt uvicorn app.api:app
curl -F file=@lot.jpg localhost:8000/predict
```

## Results

_Filled in from `results/figures/results_table.md` once the final runs finish._

## References
1. de Almeida et al. PKLot – a robust dataset for parking lot classification. ESWA, 2015.
2. Amato et al. Deep learning for decentralized parking lot occupancy detection. ESWA, 2017.
3. Carion et al. End-to-end object detection with transformers. ECCV, 2020.
4. Jocher, Chaurasia, Qiu. Ultralytics YOLOv8, 2023.
5. Redmon et al. You only look once: Unified, real-time object detection. CVPR, 2016.
6. Ren et al. Faster R-CNN. NeurIPS, 2015.
