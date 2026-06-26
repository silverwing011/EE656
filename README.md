# ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection

> **Paper:** Zhao et al., *"ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection"*, IEEE Transactions on Information Forensics and Security, Vol. 18, 2023.  
> **Implementation:** Full end-to-end reproduction in PyTorch on Google Colab, trained on FaceForensics++ (C23 compression).

---

## What This Project Is

This is a complete from-scratch implementation of the ISTVT architecture — a novel video transformer for detecting AI-generated (Deepfake) videos. The model doesn't just classify a video as real or fake; it also **explains its own decision** by generating separate spatial and temporal attention heatmaps, letting you visually see *what* the model detected and *where*.

The project covers the full ML pipeline: raw video preprocessing → face detection and cropping → dataset construction → model architecture → training with AMP and gradient accumulation → evaluation → interpretability visualization.

---

## Project Structure

```
├── preprocess_.ipynb     # Raw FF++ videos → aligned face crops (OpenCV DNN detector)
├── dataset.ipynb         # FaceSequenceDataset + DataLoaders (video-level train/val split)
├── model.ipynb           # Full ISTVT architecture (Xception + Decomposed ST-Attention + Self-Subtract)
├── train.ipynb           # Training loop (AMP, gradient accumulation, auto-resume, cosine LR)
├── visualize.ipynb       # Spatial & temporal heatmap generation via Attention Rollout
└── 5.pdf                 # Original IEEE TIFS 2023 research paper
```

---

## The Core Problem: Why Standard Deepfake Detection Falls Short

Existing **frame-based detectors** (e.g., plain Xception classifiers) look at one frame at a time and hunt for spatial artifacts like blurry edges or texture inconsistencies. This works reasonably well in controlled settings, but has three critical weaknesses:

1. **Fails on high-quality fakes** — modern synthesis methods (e.g., NeuralTextures) produce visually realistic frames with no obvious spatial artifacts.
2. **Breaks on low-quality / compressed video** — spatial details get blurred by compression, destroying the signal the model relies on.
3. **Poor generalization** — models overfit to the specific artifact signature of the manipulation method they were trained on, and fail badly on unseen methods.

The key insight from the paper: **current Deepfake synthesis methods are all frame-based**. Each frame is forged independently, so the *temporal consistency* between frames is never explicitly preserved. A video-based model that looks at sequences of frames can exploit this temporal inconsistency — a signal that is largely invisible when looking at frames one at a time.

---

## The ISTVT Architecture — Three Key Innovations

### 1. Xception Entry-Flow Feature Extractor

Each frame is independently passed through the entry-flow of a Xception network (depthwise separable convolutions with residual connections). This extracts low-level texture features at 1/8th the spatial resolution. The features from all T frames are then flattened into a token sequence for the transformer.

The paper emphasizes that Deepfake detection is fundamentally a **low-level, mesoscopic feature** problem — deeper semantic understanding doesn't help and can even hurt (larger models overfit).

### 2. Decomposed Spatial-Temporal Self-Attention

Standard video transformers run self-attention over *all* T×H×W tokens simultaneously, which costs **O(T²H²W²)** — computationally prohibitive. ISTVT decomposes this into two separate attention operations run sequentially:

| Attention Type | Attends Over | Equation | Purpose |
|---|---|---|---|
| **Temporal SA** | T frames at each spatial position j | Eq. 1 | Capture inter-frame inconsistency |
| **Spatial SA** | H×W patches within each frame k | Eq. 2 | Capture within-frame spatial artifacts |

Combined cost: **O(T² + H²W²)** — a dramatic reduction.

The temporal self-attention is applied **first** (ablation studies in Table III confirm this ordering outperforms all alternatives).

The decomposition also unlocks interpretability: since spatial and temporal processing are separated, you can visualize *what each branch learned independently*.

### 3. Self-Subtract Mechanism

This is the most elegant contribution. Before computing queries and keys for the **temporal** self-attention, the token tensor is transformed as:

```
I' = cat( I[0:2],  I[2:] − I[1:-1],  dim=0 )
```

In plain English: every frame's tokens are replaced by their **difference from the previous frame**. The temporal CLS token and first frame are kept as-is.

- **Q and K** use these difference tokens (attend to *what changed*)
- **V** uses the original tokens (extract features from *what exists*)

This forces the temporal attention head to ignore static, unchanging content and focus exclusively on **inter-frame changes** — exactly where Deepfake inconsistencies appear (lip movements, jaw motion, eye blinks). The paper demonstrates this significantly improves robustness, especially when frames contain misleading static artifacts like reflections or skin blemishes.

---

## Data Pipeline

### Preprocessing (`preprocess_.ipynb`)

- **Source:** FaceForensics++ C23 dataset (150 real + 150 fake `.mp4` videos via KaggleHub)
- **Face detection:** OpenCV DNN face detector (ResNet SSD, `res10_300x300_ssd`)
- **Output:** 32 aligned face-crop frames per video, saved as `.jpg` at 128×128
- **Structure produced:**
  ```
  istvt_data/
  ├── real/video_000/0000.jpg ... 0031.jpg
  └── fake/video_000/0000.jpg ... 0031.jpg
  ```
- All outputs saved to Google Drive to survive Colab session disconnects.

### Dataset (`dataset.ipynb`)

- **`FaceSequenceDataset`** samples T=6 consecutive frames from each video folder
- **Train/val split is by video** (not by frame) — prevents data leakage where frames from the same video appear in both sets
- 80/20 split → 240 train videos / 60 val videos
- Train augmentation: ColorJitter (brightness, contrast, saturation); val uses no augmentation
- Returns tensors of shape `(T, 3, H, W)` and a binary label (0=real, 1=fake)

---

## Training (`train.ipynb`)

| Hyperparameter | Value | Rationale |
|---|---|---|
| Sequence length T | 6 | Ablation shows T=6 is sweet spot; longer sequences add no signal (Table IV) |
| Transformer depth M | 6 | Paper uses 12 on full FF++; 6 used here for Colab memory constraints |
| Embedding dim | 256 | Balances capacity and speed |
| Attention heads | 8 | Standard for dim=256 (head_dim=32) |
| Optimizer | AdamW | Weight decay 0.01 |
| LR schedule | Cosine Annealing | 20 epochs |
| Effective batch size | 16 | batch_size=4 × grad_accum_steps=4 |
| Precision | Mixed (AMP) | CUDA `autocast` + `GradScaler` for T4 GPU efficiency |
| Loss | BCEWithLogitsLoss | Binary classification |
| Metric | AUROC + Accuracy | AUROC is standard for imbalanced deepfake benchmarks |

**Colab survival features:**
- Checkpoint saved every epoch to Google Drive (`last.pt`)
- Best model saved separately (`best.pt`) based on validation AUROC
- Auto-resume: training picks up from the last saved epoch on reconnect

---

## Interpretability Visualization (`visualize.ipynb`)

One of the paper's primary contributions is making the model's decisions *human-understandable*. This notebook implements **Attention Rollout** (Abnar & Zuidema, 2020) applied separately to the spatial and temporal attention streams.

For each input sequence, two sets of heatmaps are produced:

**Spatial Heatmaps** — highlight regions with suspicious per-frame artifacts:
- Blending edges (boundary between manipulated face and background)
- Abnormal facial feature borders
- Different across frames (each frame has independent spatial artifacts)

**Temporal Heatmaps** — highlight regions with inter-frame inconsistency:
- Moving areas: lips, jaw, eyes
- Consistent across frames (the same region is suspicious throughout)
- Ignores static areas even if they look spatially unusual

The paper demonstrates (Fig. 11, Fig. 12) that this method outperforms GradCAM and raw attention visualization. Erasing just **10% of pixels** guided by these heatmaps is enough to confuse the model — confirming the heatmaps genuinely reflect where the model is looking.

---

## Key Results from the Paper

### Intra-Dataset (Table I — FaceForensics++)

ISTVT achieves state-of-the-art on all four manipulation methods at both high (HQ) and low (LQ) quality:

| Method | DF HQ | F2F HQ | FS HQ | NT HQ | DF LQ | F2F LQ | FS LQ | NT LQ | Celeb-DF | DFDC |
|---|---|---|---|---|---|---|---|---|---|---|
| XN-avg (frame-based) | 98.9 | 98.9 | 99.6 | 95.0 | 96.8 | 91.1 | 94.6 | 87.1 | 99.4 | 84.6 |
| VTN (video transformer) | 99.6 | 99.3 | 99.6 | 95.4 | 97.9 | 92.1 | 95.7 | 90.4 | 99.3 | 91.7 |
| **ISTVT (ours)** | **99.6** | **99.6** | **100.0** | **96.8** | **98.9** | **96.1** | **97.5** | **92.1** | **99.8** | **92.1** |

### Cross-Dataset Generalization (Table II — Trained on FF++, Tested on Unseen Datasets)

Video-based methods generalize dramatically better to unseen manipulation methods:

| Method | Celeb-DF | DFDC | FaceShifter | DeeperForensics | Avg |
|---|---|---|---|---|---|
| Face X-ray (frame-based) | 79.5 | 65.5 | 92.8 | 86.8 | 81.2 |
| FTCN | 86.9 | 74.0 | 98.8 | 98.8 | 89.6 |
| **ISTVT (ours)** | **84.1** | **74.2** | **99.3** | **98.6** | **89.1** |

### Robustness to Perturbations

ISTVT significantly outperforms frame-based Xception under:
- **JPEG compression** — temporal signal survives when spatial details are blurred
- **Downscaling** — inter-frame inconsistency is preserved even at 0.1× scale
- **Random dropout** — Xception collapses beyond 36 dropout regions; ISTVT continues working because temporal artifacts are still present

---

## Key Takeaways for Interviewers

**1. Why decompose spatial and temporal attention?**  
Full spatio-temporal attention is O(T²H²W²) — infeasible for video. Decomposition reduces it to O(T² + H²W²) while matching or exceeding performance. It also enables *separate interpretability* for each dimension, which is the paper's novel contribution to explainability.

**2. Why does self-subtract work?**  
Deepfake synthesis is inherently frame-by-frame — no method explicitly preserves inter-frame consistency. Frame differences directly expose this inconsistency. By computing I[i] − I[i−1], the model is forced to look at *change* rather than *content*, filtering out static distractors (skin texture, lighting) that could cause false positives.

**3. Why do video-based methods generalize better?**  
Frame-based models overfit to the specific artifact texture of the training manipulation method. Temporal inconsistency is a more universal property of all Deepfake generation pipelines — no matter the synthesis method, frames are generated independently, so temporal artifacts always exist.

**4. Why use Attention Rollout over GradCAM for interpretability?**  
GradCAM was designed for CNNs. In transformers, the skip connections and layer normalizations make gradient-based methods unreliable. Attention Rollout tracks information flow through the full attention stack recursively, producing more faithful and concentrated heatmaps (demonstrated quantitatively in Fig. 12).

**5. What are the limitations of this approach?**  
- Requires short T=6 sequences (long sequences add noise, not signal)
- Performance on Celeb-DF is behind FTCN (32 frames) for long, stable videos where temporal inconsistency is subtle
- The self-subtract mechanism looks at *local* (adjacent-frame) inconsistency; long-range temporal patterns are not captured

---

## Environment & Dependencies

```
Python        3.10+
PyTorch       2.2.0
torchvision   0.17.0
facenet-pytorch   (MTCNN for optional face alignment)
opencv-python     (DNN face detector)
scikit-learn      (AUROC computation)
matplotlib        (visualization)
Pillow < 10.0.0   (compatibility)
kagglehub         (dataset download)
```

Training was conducted on **Google Colab (T4 GPU)** with Google Drive as persistent storage.

---

## How to Run

1. **Preprocess:** Run `preprocess_.ipynb` — downloads FF++ C23 via KaggleHub, extracts faces, saves crops to Drive
2. **Check dataset:** Run `dataset.ipynb` — verify batch shape `(4, 6, 3, 128, 128)` and train/val counts
3. **Train:** Run `train.ipynb` — trains for 20 epochs with auto-resume; checkpoints saved to Drive
4. **Visualize:** Run `visualize.ipynb` — load `best.pt`, generate spatial/temporal heatmaps for any test sequence

---

## References

- Zhao et al., "ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection," *IEEE TIFS*, 2023
- Abnar & Zuidema, "Quantifying Attention Flow in Transformers," 2020 (Rollout algorithm)
- Chefer et al., "Transformer Interpretability Beyond Attention Visualization," CVPR 2021 (LRP method, used in paper)
- Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions," CVPR 2017
- Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images," ICCV 2019
