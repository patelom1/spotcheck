# SpotCheck team plan: Thu 11/19 → Thu 11/26

**Submission (eLearning, zipped):** slides (PDF + PPTX) and a demo video (MP4).
Prof. Tian downloads submissions **the day before** presentations, so treat **Wed 11/25 at 9 pm** as our real deadline
and Thu 11/26 as buffer.

**Scope change (send the email in `docs/EMAIL_TO_PROFESSOR.md` on Thu 11/19):** we no longer collect UTD photos.
The "new site" is PUCPR, the PKLot lot we hold out. Every image and label is real.

## Roles

| Person | Owns | Presents |
|---|---|---|
| **Ridham** | Data: PKLot download + cross-site conversion, dataset stats, label sanity checks, example-image figure | Intro + problem + data (≈1:05) |
| **Om** | Repo owner + training: configs, Colab runs, hyperparameters | Method + experiments (≈1:10) |
| **Krishna** | Evaluation: metrics, counting error, weather/time-of-day breakdown, occupancy-by-hour, figures, failure cases | Results (≈1:25) |
| **Kevin** | Inference wrapper, time-lapse demo video, Gradio web demo, REST API | Demo + takeaways (≈1:00) |

## What gets graded
Only the **slides** (PDF + PPTX) and a **demo video** (MP4) are submitted. The grade is the presentation:
content, clarity, and Q&A, with points lost if Intro + Method + Results run over **5 minutes**.
Code is not submitted, so keep git simple and spend the time on results, the video, and rehearsal.

## Git (lightweight)
* One GitHub repo owned by **Om**, used to share code and keep one working version.
* Everyone pushes to `main`; `git pull` before you start and push when something works. No PR reviews needed.
* Never commit datasets or weights. Share `best.pt` files and `results/` through the team Drive folder.
* Check the course policy on AI tools, and add a line to the README or final slide if it asks for disclosure.

## Day-by-day

| Day | Who | Goal | Done when |
|---|---|---|---|
| **Thu 11/19** | **Ridham** (data) | Download PKLot, run `crosssite.py --max-per-session 20`, spot-check labels, zip `datasets/xsite` to the team Drive, post dataset stats | `xsite.zip` is on the Drive and stats are in the group chat |
| **Thu 11/19** | **Om** (repo + training) | Create repo, add team, send scope email, smoke test; start the training chain as soon as data is ready; keep the Colab tab open overnight | Chain running; by Fri morning at least E1 + E3 are in `spotcheck/runs` on the Drive |
| **Fri 11/20** | **Krishna** (evaluation) | Run `evaluate.py`, `plots.py`, `traffic.py` on finished runs; pick 3 failure cases; re-run when the last runs land | `results/` (table + figures) on the Drive |
| **Sat 11/21** | **Kevin** (demo) | Pick a test day (lot fills up and empties); final demo MP4 with real numbers on the end card; web-demo screenshots | `demo/spotcheck_demo.mp4` (≤60 s) on the Drive |
| **Sun 11/22** | Buffer | Re-run anything that failed; E5 (YOLOv8s) only if there's time | |
| **Mon 11/23** | **Everyone** | Fill your own slides with real numbers; write your speaker script | No `[__]` left |
| **Tue 11/24** | **Everyone** | **Rehearsal #1, timed, target 4:40**; fix slides | Under 5:00 |
| **Wed 11/25** | **Everyone** | **Rehearsal #2** + Q&A drill; export PDF + PPTX; zip with MP4; **submit by 9 pm** | Submitted |
| **Thu 11/26** | — | Buffer | |

**Handoffs.** Each person posts in the group chat when their part is on the Drive. The next person starts as soon
as they see the message, without waiting for midnight. Kevin can start Friday night as soon as E1 and E3 exist, because the demo needs only those two models.

**If a day slips.** Everything after it slips too, so the earlier you finish, the more buffer everyone gets.
If Ridham hits a download problem, Om starts the smoke test and Krishna and Kevin practice the evaluation and
demo commands on the synthetic data.

**While waiting for your day:** read your own module and the README section for it, run your commands on the
synthetic smoke data, and draft your speaker script, so your day is just running the real thing.


## Likely Q&A
* *Why not your own campus photos?* Time. PKLot gives three real lots with different cameras, so we can measure
  transfer to a site the model never saw, which is the same question we'd ask at UTD.
* *Why split by day?* Frames 5 minutes apart are nearly identical; splitting by image would leak test into train.
* *Why fine-tune instead of training on the new site only?* Compare E2 and E3, plus the label-efficiency curve.
* *Night?* PKLot was captured in daylight (check the hours in `manifest.csv`). We report weather and time of day instead, and list night as future work.
* *Could it run live?* FPS on the results slide; the API takes a photo and returns counts.
