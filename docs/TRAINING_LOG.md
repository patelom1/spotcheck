# SpotCheck training log (Thu 11/19/2026)

All numbers below are copied from the Colab training logs. **Validation** numbers only;
the final **test** numbers (360 unseen PUCPR images) come from `spotcheck.evaluate` and go on the slides.

## Dataset (PKLot, cross-site split by capture day, max 20 images per day)

| Split | Lot(s) | Images | Labeled spaces | Notes |
|---|---|---|---|---|
| source_train | UFPR04 + UFPR05 | 1,064 | | E1 training data |
| source_val | UFPR04 + UFPR05 | 164 | 5,588 (4,193 empty / 1,395 occupied) | E1 validation |
| target_pool | PUCPR | 236 | | fine-tuning subsets drawn from here (25 ⊂ 50 ⊂ 100 ⊂ 200) |
| target_val | PUCPR | 100 | 9,999 (3,079 empty / 6,920 occupied) | validation for all new-site runs |
| target_test | PUCPR | 360 | | final test; 215 sunny / 105 cloudy / 40 rainy; 127 morning / 100 midday / 86 afternoon / 47 evening |

## Common settings

| Setting | Value |
|---|---|
| Model | YOLOv8n, COCO-pretrained (3.0 M parameters, 8.2 GFLOPs), 2 classes: empty, occupied |
| Image size | 960 px |
| Batch | 16 |
| Early stopping | patience 10 epochs |
| Augmentation | brightness ±50%, saturation ±60%, rotation ±5°, horizontal flip, mosaic (off for last 5 epochs) |
| Hardware / software | Google Colab Tesla T4 (15 GB); Ultralytics 8.4.215; PyTorch 2.14.1; seed 0 |

## Runs

| Run | Trained on | Starts from | Optimizer, learning rate | Epochs | Train time |
|---|---|---|---|---|---|
| **E1** source_only | 1,064 source images | COCO | AdamW 0.00167 (auto), 3-epoch warm-up | 25 | 15.4 min |
| **E2** target_only | 100 PUCPR images | COCO | AdamW 0.00167 (auto), warm-up | 25 | 2.1 min |
| **E3** finetune | 100 PUCPR images | E1 | AdamW 0.0005, no warm-up | 25 | 2.3 min |
| finetune_25 | 25 PUCPR images | E1 | AdamW 0.0005, no warm-up | 75 | 3.8 min |
| finetune_50 | 50 PUCPR images | E1 | AdamW 0.0005, no warm-up | 40 | 2.8 min |
| finetune_200 | 200 PUCPR images | E1 | AdamW 0.0005, no warm-up | 25 | 4.0 min |
| **E4** combined | 1,064 source + 100 PUCPR | COCO | AdamW 0.00167 (auto), warm-up | 25 | 15.4 min |

Fine-tunes with 25 and 50 images got more epochs so every fine-tune sees a similar number of
training steps (~150–175 batches).

## Validation results

### On the new lot (PUCPR target_val, 100 images, 9,999 spaces)

| Run | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| **E1** source_only | 0.067 | 0.015 | **0.002** | 0.001 |
| finetune_25 | 0.909 | 0.917 | **0.968** | 0.645 |
| finetune_50 | 0.931 | 0.935 | **0.978** | 0.652 |
| **E3** finetune (100) | 0.916 | 0.907 | **0.968** | 0.641 |
| finetune_200 | 0.968 | 0.993 | **0.991** | 0.706 |
| **E2** target_only (100) | 0.896 | 0.964 | **0.974** | 0.663 |
| **E4** combined | 0.991 | 0.994 | **0.995** | 0.756 |

### E1 on its own lots (source_val, 164 images)

| Run | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| E1 source_only | 0.994 | 0.995 | 0.991 | 0.890 |

### Speed
Model inference ≈ 4–6 ms per 960 px image on a T4 (from the validation logs; pre/post-processing adds ~3–7 ms).
End-to-end FPS is measured by `spotcheck.evaluate`.

## Takeaways (validation)
1. **Domain gap:** E1 scores 0.991 on its own lots but 0.002 on the new lot.
2. **Little labeling needed:** 25 labeled new-lot images bring mAP@0.5 to 0.968; returns diminish after ~50.
3. **Starting point matters less than expected:** with 100 new-lot images, fine-tuning from E1 (0.968) ≈ training from COCO (0.974).
   Training on everything together (E4) is best (0.995).

## History (for Q&A)
- The first fine-tuning attempt used `optimizer: auto`, which silently ignored our lower learning rate,
  and Ultralytics' warm-up (≥100 steps) covered most of the tiny runs. We fixed both (explicit AdamW, lr 0.0005,
  no warm-up) and reran all fine-tunes. Only the fixed runs are reported.
- Every fine-tune starts near mAP 0 for ~25 training steps: that is E1 meeting the new lot, consistent with
  its 0.002 score.
