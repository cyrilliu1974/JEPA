# AIM × V-JEPA 2: Interpretability via Emergent Symbol Systems

> **Can we read what a world model is thinking?**
> This project proposes a first answer.

---

## The Problem

[V-JEPA 2](https://github.com/facebookresearch/vjepa2) (Meta FAIR, 2025) is one of the most capable video understanding models to date. It learns by predicting missing regions of video in **latent space** — an abstract mathematical space that is deliberately designed to ignore pixel-level details and capture only the essential structure of the physical world.

This design makes V-JEPA 2 powerful. It also makes it opaque.

When V-JEPA 2 watches a robot grasp an object, its internal representation is a high-dimensional continuous vector. There is no way to directly read what the model "understood" from the video. Researchers can observe that it performs well on benchmarks, but cannot inspect the reasoning pathway that led to that performance. This is the **interpretability gap** that motivates this project.

---

## The Idea

The **AI Mother Tongue (AIM) framework** ([arXiv:2507.10566](https://arxiv.org/abs/2507.10566)) proposes that agents in multi-agent reinforcement learning can develop a self-emergent discrete symbol system — a compressed semantic language that arises from the agents' own learning dynamics, without any human-defined vocabulary.

The key observation that connects AIM to V-JEPA 2 is architectural:

> Both frameworks operate in latent space. V-JEPA 2 predicts in latent space. AIM compresses in latent space. They are naturally compatible.

This means AIM's Vector Quantized VAE (VQ-VAE) can be attached directly to V-JEPA 2's encoder output — not as a pixel decoder, not as a language model head, but as a **lightweight quantization probe** that converts continuous latent vectors into discrete symbol sequences.

If the resulting symbols systematically vary with physical conditions (object geometry, motion speed, grasp angle), then the symbols constitute a **structured internal narrative** — a first form of machine-readable explanation of what V-JEPA 2 perceived.

This is what we mean by interpretability: not pixel reconstruction, not natural language explanation, but a statistically auditable symbolic record of the model's latent state transitions.

### Why the encoder must be frozen in Stage 1

This is not merely a technical convenience — it is a scientific requirement.

Stage 1 asks a precise question: **can AIM's symbolization mechanism read knowledge that V-JEPA 2 has already learned?** To answer this cleanly, all confounding variables must be eliminated. Freezing the encoder achieves this:

```
If encoder is NOT frozen:
  Experiment fails → unclear whether AIM failed, or the encoder was disrupted
  Experiment succeeds → unclear whether AIM succeeded, or the encoder adapted to help

If encoder IS frozen:
  Experiment fails → AIM cannot read this latent space
  Experiment succeeds → AIM successfully read knowledge V-JEPA 2 already possessed
```

Freezing the encoder also has a practical consequence: because a frozen encoder always produces identical outputs for the same input, all z-vectors can be pre-computed once and reused throughout training. This is both faster and more sample-efficient — 50 videos × 16 frames = 800 training samples instead of 50.

Stage 3 will unfreeze the encoder, but only after Stage 1 and 2 have established that AIM and V-JEPA 2 are architecturally compatible. At that point, joint training becomes meaningful — without that foundation, it would be impossible to distinguish genuine co-learning from mutual accommodation.

---

## Project Structure

```
project_root/
├── aim_bridge/                    # Core integration modules
│   ├── stage1_diagnosis.py        # Main entry point for Stage 1
│   ├── aim_intervention_builder.py # Intervention experiments & dictionary
│   ├── quantizer.py               # AIM VQ quantizer for V-JEPA 2
│   └── video_dataset.py           # Video loading & preprocessing
│
├── aim/                           # AIM framework (cyrilliu1974/AI-Mother-Tongue)
│   ├── aim_dictionary_json.py     # Symbol dictionary persistence
│   ├── aim_adapter.py             # Data bridge for DRCB framework
│   ├── aim_collusion_framework.py # DRCB collusion detection
│   ├── analyze_aim.py             # Post-hoc dictionary analysis
│   └── enhanced_aim_dictionary.py # Extended dictionary management
│
├── test_vjepa2_latent.py          # ⚠️ Run this FIRST — checks z norm, N_tokens
├── test_projection_health.py      # Run if Loss=0 persists — checks projection layer
│
├── checkpoints/                   # Model weights (not tracked by git)
│   └── README.md                  # Download instructions (see below)
│
├── data/                          # Video datasets (not tracked by git)
│   └── kinetics_mini/
│       └── val/
│           ├── archery/
│           ├── bowling/
│           ├── flying_kite/
│           ├── high_jump/
│           └── marching/
│
└── stage1_results/                # Auto-generated outputs
    ├── stage1_report.json
    ├── aim_dictionary_stage1.json
    └── figures/
```

---

## Four-Stage Research Roadmap

This project follows a progressive integration strategy. Each stage has explicit pass/fail criteria before proceeding. The central design principle is **progressive unfreezing**: the encoder starts completely frozen and is only released once the symbolization mechanism has proven itself on stable features.

| Stage | Goal | V-JEPA 2 | Status |
|---|---|---|---|
| **Stage 1** | Verify AIM symbols capture semantic differences | **Frozen** — encoder untouched | ✅ In progress |
| Stage 2 | Quantizer warm-up with stable codebook | **Frozen** — encoder untouched | Planned |
| Stage 3 | Joint training with symmetric quantization | **Unfrozen** — encoder fine-tuned | Future work |
| Stage 4 | Action-conditioned symbolic world model | **Unfrozen** — encoder fine-tuned | Future work |

### Why this progression matters

**Stages 1–2 (encoder frozen):** The encoder's weights never change. Every input produces exactly the same output every time. This makes the experiment scientifically clean — any symbol patterns that emerge are entirely attributable to what V-JEPA 2 already knows, not to any adaptation that happened during our experiment. It also allows z-vectors to be pre-computed once and reused, making training both faster and more sample-efficient.

**Stages 3–4 (encoder unfrozen):** Joint training begins only after Stages 1 and 2 have confirmed architectural compatibility. Without that foundation, it would be impossible to tell whether joint training produced genuine co-learning or merely mutual accommodation — the two models adjusting to each other without either contributing meaningful knowledge.

The frozen-then-unfreeze strategy is the same logic used in transfer learning: first verify that the pre-trained representation is useful as-is, then fine-tune once you are confident the integration is sound.

**Stage 1 is the focus of this repository.** The encoder is completely frozen — no V-JEPA 2 weights are modified. Only the lightweight AIM quantizer is trained.

### Stage 1 Pass Criteria

| Metric | Threshold | Meaning |
|---|---|---|
| H1 Symbol Stability | > 95% | Same video → same symbol across 20 repeats |
| H2 Chi-square p-value | < 0.01 | Symbol distributions differ significantly across conditions |
| H2 MI Ratio | > 5× baseline | Experiment MI >> random noise MI |
| Codebook Active Ratio | > 30% | No codebook collapse |

---

## Model Download

### Option A: HuggingFace (Recommended)

The code loads models automatically from HuggingFace on first run. No manual download needed.

```python
# Handled automatically inside stage1_diagnosis.py
# ViT-L is used by default (--encoder_embed_dim 1024)
```

### Option B: Direct Download

Place checkpoint files in the `checkpoints/` directory.

**ViT-L** (recommended for Stage 1, ~1.2 GB):
```bash
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/vjepa2/vitl.pt -O checkpoints/vjepa2_vitl.pt
```

**ViT-g** (larger, ~10 GB):
```bash
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/vjepa2/vitg.pt -O checkpoints/vjepa2_vitg.pt
```

### Which Model to Use

| Model | embed_dim | File Size | Recommended For |
|---|---|---|---|
| ViT-L | 1024 | ~1.2 GB | Stage 1 (concept validation) |
| ViT-g | 1408 | ~10 GB | Full experiments |

For Stage 1, ViT-L is sufficient. Use `--encoder_embed_dim 1024` when running with ViT-L.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/aim-vjepa2.git
cd aim-vjepa2
pip install -r requirements.txt
```

For video decoding, install at least one of:
```bash
pip install torchcodec   # recommended
# OR
pip install decord
# OR
pip install opencv-python  # fallback
```

---

## Latent Space Diagnosis (Run This First)

Before running Stage 1, always run the latent space diagnostic tool on your own video data. This step takes less than 5 minutes and can save hours of failed training.

The code logic itself is fixed and stable. However, **different datasets will produce different latent space characteristics**, and some of these characteristics can cause training to fail silently. The diagnostic tool detects these issues before you commit to a full run.

### What it checks

`test_vjepa2_latent.py` verifies four things:

| Check | Why It Matters |
|---|---|
| z vector norm | High norm (>10) causes `Loss=0` and codebook collapse |
| N_tokens divisibility | Must be divisible by `num_frames` or `temporal_pool` will crash |
| Per-class z statistics | Low inter-class variance means AIM symbols cannot distinguish conditions |
| Per-video consistency | Detects random augmentation that would break H1 stability |

### Run the diagnostic

```bash
# Local
python test_vjepa2_latent.py \
    --video_root ./data/kinetics_mini/val \
    --device cuda

# Kaggle / Colab
!python test_vjepa2_latent.py \
    --video_root ./data/kinetics_mini/val \
    --device cuda
```

### Data-dependent issues and how to handle them

These issues are **not bugs in the code**. They depend on your dataset, model size, and video resolution. You may encounter them when switching to a new dataset.

---

**Issue 1: Loss drops to 0 immediately**

```
Step 0   | Loss=0.0013
Step 100 | Loss=0.0000   ← stop here
```

Cause: The z norm is too high. V-JEPA 2's ViT-L outputs vectors with norm ≈ 97. After two rounds of L2 normalization inside the quantizer, all vectors collapse to the same direction and the codebook cannot distinguish anything.

Check: Look at the `z norm` value in the diagnostic output.

| z norm | Action |
|---|---|
| 1–10 | No adjustment needed |
| 10–150 | Remove the first L2 normalize in `Stage1AIMQuantizer.encode()` |
| > 150 | Add stronger normalization before the projection layer |

Fix (remove first L2 normalize in `stage1_diagnosis.py`):

```python
# Remove this line:
z_norm = F.normalize(z_frame, dim=-1)

# Keep only:
z_proj = self.projection(z_frame)         # LayerNorm inside projection handles scale
z_proj_norm = F.normalize(z_proj, dim=-1) # normalize only after projection
```

---

**Issue 2: N_tokens is not divisible by num_frames**

```
AssertionError: N_tokens (1540) must be divisible by num_frames (16)
```

Cause: Different video resolutions produce different N_tokens. Kinetics-mini at 360×480 gives N_tokens=1568 (divisible by 16). Your videos at a different resolution may not.

Check: The diagnostic reports `N_tokens % num_frames == 0` directly.

Fix: Adjust `--num_frames` to a value that divides your N_tokens, or resize videos to 224×224 before processing.

---

**Issue 3: Codebook collapse (Active Ratio stays at 3%)**

```
Step 500 | Active=3%   ← stop if this does not improve
Step 200 | ↻ Reset 62 dead codes at step 200  ← reset happening but not helping
```

Cause: Your video classes are too similar in latent space. If all classes have nearly identical z vector directions (cosine similarity > 0.99), the quantizer cannot find meaningful cluster boundaries regardless of training duration.

Check: Compare `z mean` across classes in the diagnostic output. If all classes show nearly identical statistics, the dataset may not have sufficient latent-space diversity for AIM symbolization.

Fix options:
- Choose more semantically distinct video classes
- Increase `codebook_size` to a smaller number (e.g., 16 or 32) to reduce pressure on the codebook
- Increase `warmup_iterations` to give the codebook more time to spread out

---

**Issue 4: Random baseline MI is not close to 0**

```
Random baseline MI: 0.45   ← should be < 0.1
```

Cause: Your dataset has a systematic bias unrelated to the semantic variable you are testing (e.g., all videos of one class happen to be brighter, or filmed from a fixed angle). The codebook learned this bias instead of semantics.

Fix: Check whether your conditions differ in confounding factors (brightness, camera angle, background). Consider rebalancing the dataset or normalizing video statistics before processing.

---

**Issue 5: H1 stability below 95%**

```
H1 Result: 0.72   ← stop here
```

Cause: Something in your preprocessing pipeline is introducing randomness. This could be random frame sampling, random crop, color jitter, or any other stochastic augmentation applied during inference.

Fix: Ensure `encoder.eval()` is called and all augmentation is disabled before running H1. The code already handles this, but custom video loading pipelines may reintroduce randomness.

---

**Issue 5.5: Verify projection layer health (optional but recommended)**

If `test_vjepa2_latent.py` shows `z norm > 10` and you have modified the quantizer, run this additional check to confirm the projection layer produces distinguishable vectors:

```bash
python test_projection_health.py
```

No arguments needed. It generates synthetic vectors matching your encoder's norm characteristics and reports:

```
Cosine similarity between different inputs: 0.03
(Close to 1.0 = still collapsing, close to 0 = distinguishable)
```

A value below 0.1 means the projection layer is working correctly. If cosine similarity is above 0.9, increase `--projection_dim`. This test runs in under 10 seconds on CPU with no model download required.

---

**Issue 6: Perplexity drops during training (EMA collapse)**

```
Step   0 | Loss=0.0015 | Perplexity=2.69/4.16 (65%) | Active=14%
Step 100 | Loss=0.0000 | Perplexity=1.02/4.16 (25%) | Active=14%  ← stop here
```

Cause: The EMA decay rate is too conservative for your dataset's diversity. With `ema_decay=0.99`, the codebook updates only 1% per step. If your dataset has limited diversity (fewer videos, fewer classes, or semantically similar classes), the codebook vectors get pulled toward the data mean and collapse, even though individual vectors were distinguishable at initialization.

The code includes an early detection check at Step 50 that will warn you before Step 100:

```
⚠️  Perplexity dropping (2.69 → 1.02). EMA collapse likely.
    Stop and retry with lower ema_decay / higher commitment_cost.
    Suggested: --ema_decay 0.90 --commitment_cost 2.0
```

Fix: Stop training and rerun with more aggressive EMA settings:

```bash
python stage1_diagnosis.py --run_with_kinetics \
    --video_root ./data/kinetics_mini/val \
    --encoder_embed_dim 1024 \
    --device cuda \
    --ema_decay 0.90 \
    --commitment_cost 2.0 \
    --output_dir ./stage1_results
```

General guidance for tuning these parameters:

| Dataset diversity | Recommended ema_decay | Recommended commitment_cost |
|---|---|---|
| High (many classes, distinct) | 0.99 | 0.25 |
| Medium (5–10 classes) | 0.95 | 1.0 |
| Low (few classes, similar) | 0.90 | 2.0 |

If you are unsure, start with `ema_decay=0.95` and `commitment_cost=1.0` as a safe default for most datasets.

The diagnostic works with any video dataset. Point `--video_root` at any directory with class subdirectories containing `.mp4` files:

```
your_data/
├── class_A/
│   ├── video1.mp4
│   └── video2.mp4
└── class_B/
    ├── video3.mp4
    └── video4.mp4
```

```bash
python test_vjepa2_latent.py --video_root ./your_data --device cuda
```

Run this every time you switch to a new dataset or a different V-JEPA 2 model size. The latent space characteristics vary across ViT-L, ViT-H, and ViT-g.

---

## Quick Start

### Step 0: Run latent space diagnostic (required)

```bash
python test_vjepa2_latent.py \
    --video_root ./data/kinetics_mini/val \
    --device cuda
```

Check the output for `z norm` and adjust quantizer parameters if needed (see above).

### Step 1: Verify the pipeline (no model or video needed)

```bash
cd aim_bridge
python stage1_diagnosis.py --run_mock
```

Expected output: all 4 criteria pass (H1, H2-chi2, H2-MI, codebook).

### Step 2: Download videos

Two dataset options are available. Choose based on your needs:

**Option A: Kinetics-mini (default, ~50 videos/class)**

Quick to download, sufficient for concept validation:

```bash
# Standalone download
python video_dataset.py --download --dataset kinetics-mini

# Or let stage1_diagnosis.py download automatically
python stage1_diagnosis.py --run_with_kinetics \
    --auto_download --dataset kinetics-mini \
    --encoder_embed_dim 1024 --device cuda \
    --output_dir ./stage1_results
```

**Option B: Kinetics full (~400 videos/class)**

For larger-scale experiments. Specify which classes to download to save time and disk space:

```bash
# Standalone download (5 classes only)
python video_dataset.py --download --dataset kinetics \
    --classes archery bowling flying_kite high_jump marching

# Or let stage1_diagnosis.py download automatically
python stage1_diagnosis.py --run_with_kinetics \
    --auto_download --dataset kinetics \
    --download_classes archery bowling flying_kite high_jump marching \
    --max_per_class 100 \
    --encoder_embed_dim 1024 --device cuda \
    --output_dir ./stage1_results
```

Both options skip the download if the data already exists locally.

### Step 3: Run Stage 1 diagnosis

```bash
# Minimal (kinetics-mini, default parameters)
python stage1_diagnosis.py --run_with_kinetics \
    --video_root ./data/kinetics_mini/val \
    --encoder_embed_dim 1024 \
    --device cuda \
    --output_dir ./stage1_results

# If Step 50 shows EMA collapse warning, add these:
python stage1_diagnosis.py --run_with_kinetics \
    --video_root ./data/kinetics_mini/val \
    --encoder_embed_dim 1024 \
    --device cuda \
    --ema_decay 0.90 \
    --commitment_cost 2.0 \
    --batch_size 16 \
    --output_dir ./stage1_results

# Using ViT-g (larger model, requires --encoder_embed_dim 1408)
python stage1_diagnosis.py --run_with_kinetics \
    --video_root ./data/kinetics_mini/val \
    --encoder_embed_dim 1408 \
    --device cuda \
    --output_dir ./stage1_results
```

### Step 4: Analyze the dictionary

```bash
cd ../aim
python analyze_aim.py \
    --dict_path ../stage1_results/aim_dictionary_stage1.json \
    --K_val 64 \
    --top_N_aim 10
```

---

## Outputs

After Stage 1 completes, results are saved to `stage1_results/`:

```
stage1_results/
├── stage1_report.json              # Pass/fail for all 4 criteria
├── aim_dictionary_stage1.json      # Initial AIM symbol dictionary
└── figures/
    ├── stage_a_training.png        # Quantizer training curve
    ├── intervention_grasp_angle.png    # JSD heatmap + symbol distributions
    ├── intervention_object_geometry.png
    ├── intervention_motion_speed.png
    └── stage1_summary.png          # Overall pass/fail summary
```

---

## Related Work

- **V-JEPA 2**: [Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985) — Assran et al., Meta FAIR, 2025
- **AIM Framework**: [AI Mother Tongue: Self-Emergent Communication in MARL via Endogenous Symbol Systems](https://arxiv.org/abs/2507.10566) — Liu, 2025
- **DRCB**: Dynamic Representational Circuit Breaker for steganographic collusion detection in MARL (companion paper)

---

## Notes

- This project does **not** modify V-JEPA 2 weights in Stage 1. The encoder is completely frozen.
- **Always run `test_vjepa2_latent.py` before Stage 1**, especially when using a new dataset or a different model size. Latent space characteristics vary across ViT-L, ViT-H, and ViT-g, and can cause silent training failures if not checked first.
- The code logic is fixed and stable. Issues that arise when switching datasets are data-dependent, not code bugs. See the **Latent Space Diagnosis** section for a complete guide.
- The AIM quantizer trained in Stage 1 operates on frozen features and takes approximately 1–3 hours on CPU (ViT-L) or 30–60 minutes on GPU.
- Model checkpoint files (`.pt`, `.pth`) are excluded from version control via `.gitignore`.
- Stage 1 uses [Kinetics-mini](https://huggingface.co/datasets/nateraw/kinetics-mini) as a proof-of-concept dataset. Limitations of this dataset (confounding variables across action classes) are acknowledged; controlled physical experiments are planned as future work.

---

## License

MIT