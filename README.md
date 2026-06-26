# ISTVT Reproduction — Course Project Guide

A scaled-down, Colab/Kaggle-friendly re-implementation of:
**ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection**
(Zhao et al., IEEE TIFS 2023)

## What's in this repo

```
istvt/
  model.py        - Xception entry-flow extractor + decomposed ST self-attention
                     + self-subtract mechanism + full ISTVT model
  dataset.py       - face-sequence dataset loader (train/val split)
  preprocess.py    - video -> aligned face-crop frames (MTCNN-based)
  train.py         - training loop with checkpoint resume (survives Colab disconnects)
  visualize.py      - simplified attention-rollout interpretability demo
requirements.txt
```

## 1. Why you MUST scale down (read this first)

The paper trains on **4× Tesla V100s for up to 100 epochs** on the **full**
FaceForensics++ (1000 real + 4000 fake videos × 270 frames each), plus
Celeb-DF (~6000 videos) and DFDC. That is hundreds of GPU-hours.

Free-tier Colab/Kaggle gives you a single T4 (16GB) for ~12-30 hrs/week with
session timeouts (~12h Colab, ~9h Kaggle, and disconnects on idle). **You
cannot reproduce Table I exactly — and that's fine.** Your project's value
is in faithfully implementing the *architecture* (decomposed ST-attention +
self-subtract) and the *interpretability story*, on a realistic subset, with
an honest discussion of what you scaled down and why.

### Recommended scaled config (already the defaults in this code)

| Setting | Paper | This project |
|---|---|---|
| Frame size | 300×300 | **128×128** |
| Sequence length T | 6 | 6 (keep this — it's cheap and central to the method) |
| Embed dim C | 728 | **256** |
| Transformer depth M | 12 | **6** |
| Dataset | full FF++ (5 datasets total) | **FF++ subset**: ~100 real + 100 fake videos, single manipulation method (Deepfakes is easiest visually) |
| Frames/video | 270 | **24-32** |
| Epochs | 100 | **15-25** (with early stopping on val AUROC) |
| Batch size | large, 4 GPUs | **4**, with `grad_accum_steps=4` (effective 16) |

This config trains a ~5-10M parameter model — should comfortably fit a T4
and finish an epoch over ~160 videos in a few minutes.

## 2. Getting the dataset

FaceForensics++ requires filling out a Google form for full access
(`https://github.com/ondyari/FaceForensics`), which can take a few days —
**do this on Day 1**. While waiting, you can prototype with:
- The small **FF++ sample videos** (publicly downloadable, no form needed,
  ~10 videos) to get your pipeline running end-to-end.
- **Celeb-DF** or **DFDC preview** as a backup if FF++ access is delayed.

Once you have videos:
```bash
python istvt/preprocess.py --videos_dir raw/real --out_dir data/real \
    --frames_per_video 32 --img_size 128 --max_videos 100
python istvt/preprocess.py --videos_dir raw/fake --out_dir data/fake \
    --frames_per_video 32 --img_size 128 --max_videos 100
```

## 3. Training

```bash
pip install -r requirements.txt
cd istvt
python train.py --data_root ../data --epochs 20 --batch_size 4 \
    --img_size 128 --seq_len 6 --embed_dim 256 --depth 6 \
    --ckpt_dir ../checkpoints
```
If your Colab session dies, just rerun the same command — it resumes from
`checkpoints/last.pt` automatically.

## 4. Day-by-day plan (2 weeks)

**Days 1-2 — Setup**
- Request FF++ access; meanwhile download FF++ sample videos
- Set up Colab/Kaggle notebook, mount Drive/dataset for persistence
  (critical: Colab wipes local disk on disconnect — preprocess once,
  save the cropped frames to Drive, don't redo face detection every session)
- Run `preprocess.py` on the small sample set, sanity check crops

**Days 3-4 — Get the model training**
- Drop in `model.py`, run the `__main__` shape-check, fix any shape bugs
  for your chosen img_size/embed_dim
- Run `train.py` for 1-2 epochs on a tiny subset (10 videos) just to confirm
  loss decreases and nothing crashes — this catches 90% of bugs cheaply

**Days 5-7 — Real training run**
- Once FF++ access arrives, build your ~100+100 video subset
- Kick off the full 15-25 epoch training run (checkpointed, resumable)
- While it trains, start writing the report's Method section (you already
  understand the architecture from this conversation)

**Days 8-9 — Evaluation**
- Compute accuracy/AUROC on held-out videos
- **Cross-manipulation test** (cheap but impressive): train on Deepfakes only,
  test on Face2Face/FaceSwap/NeuralTextures subsets — mirrors the paper's
  Table II generalization experiment and is a great talking point even at
  small scale
- Compare against a simple frame-based baseline (e.g. just the Xception
  extractor + classifier head, no transformer) to demonstrate the value of
  the temporal/spatial decomposition — this directly recreates the paper's
  core argument (Sec. IV-C.1)

**Days 10-11 — Interpretability**
- Run `visualize.py` on a few correctly-classified fake videos
- Show spatial heatmaps (should highlight face boundaries/blending) vs.
  temporal heatmaps (should highlight moving regions: lips, eyes)
- This is your strongest "wow" demo for grading — lean on it

**Days 12-14 — Ablations, polish, report**
- Quick ablation: self-subtract on vs. off (just retrain with `self_subtract`
  replaced by identity) — should show a measurable accuracy/robustness drop,
  reproducing paper's Table III finding
- Optional robustness test: re-run eval after JPEG-compressing or
  downscaling your test videos (mirrors Fig. 6)
- Write up: Method, scaled-down experimental setup (be explicit about what
  you changed from the paper and why), results, visualization examples,
  limitations vs. the original paper

## 5. What to honestly flag as "not reproduced" in your report

- Full LRP-based interpretability (Algorithm 1, Eq. 4-5) — we substitute
  Attention Rollout (a baseline the paper itself compares against in
  Fig. 11/12), which is a legitimate scoped-down choice to discuss.
- Multi-dataset cross-domain generalization (Table II) — limited to
  within-FF++ cross-manipulation instead of cross-dataset.
- Exact paper hyperparameters (embed dim 728, depth 12, 270 frames/video) —
  scaled down for single-GPU feasibility; cite the compute gap explicitly.

Being upfront about these tradeoffs (with reasoning) generally reads as more
rigorous to a grader than silently using a smaller setup.
