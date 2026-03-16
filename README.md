\# AIM × V-JEPA 2: Interpretability via Emergent Symbol Systems



> \*\*Can we read what a world model is thinking?\*\*

> This project proposes a first answer.



\---



\## The Problem



\[V-JEPA 2](https://github.com/facebookresearch/vjepa2) (Meta FAIR, 2025) is one of the most capable video understanding models to date. It learns by predicting missing regions of video in \*\*latent space\*\* — an abstract mathematical space that is deliberately designed to ignore pixel-level details and capture only the essential structure of the physical world.



This design makes V-JEPA 2 powerful. It also makes it opaque.



When V-JEPA 2 watches a robot grasp an object, its internal representation is a high-dimensional continuous vector. There is no way to directly read what the model "understood" from the video. Researchers can observe that it performs well on benchmarks, but cannot inspect the reasoning pathway that led to that performance. This is the \*\*interpretability gap\*\* that motivates this project.



\---



\## The Idea



The \*\*AI Mother Tongue (AIM) framework\*\* (\[arXiv:2507.10566](https://arxiv.org/abs/2507.10566)) proposes that agents in multi-agent reinforcement learning can develop a self-emergent discrete symbol system — a compressed semantic language that arises from the agents' own learning dynamics, without any human-defined vocabulary.



The key observation that connects AIM to V-JEPA 2 is architectural:



> Both frameworks operate in latent space. V-JEPA 2 predicts in latent space. AIM compresses in latent space. They are naturally compatible.



This means AIM's Vector Quantized VAE (VQ-VAE) can be attached directly to V-JEPA 2's encoder output — not as a pixel decoder, not as a language model head, but as a \*\*lightweight quantization probe\*\* that converts continuous latent vectors into discrete symbol sequences.



If the resulting symbols systematically vary with physical conditions (object geometry, motion speed, grasp angle), then the symbols constitute a \*\*structured internal narrative\*\* — a first form of machine-readable explanation of what V-JEPA 2 perceived.



This is what we mean by interpretability: not pixel reconstruction, not natural language explanation, but a statistically auditable symbolic record of the model's latent state transitions.



\---



\## Project Structure



```

project\_root/

├                 # Core integration modules

│   ├── stage1\_diagnosis.py        # Main entry point for Stage 1

│   ├── aim\_intervention\_builder.py # Intervention experiments \& dictionary

│   ├── quantizer.py               # AIM VQ quantizer for V-JEPA 2

│   └── video\_dataset.py           # Video loading \& preprocessing

│

├── aim/                           # AIM framework (cyrilliu1974/AI-Mother-Tongue)

│   ├── aim\_dictionary\_json.py     # Symbol dictionary persistence

│   ├── aim\_adapter.py             # Data bridge for DRCB framework

│   ├── aim\_collusion\_framework.py # DRCB collusion detection

│   ├── analyze\_aim.py             # Post-hoc dictionary analysis

│   └── enhanced\_aim\_dictionary.py # Extended dictionary management

│

├── checkpoints/                   # Model weights (not tracked by git)


│

├── data/                          # Video datasets (not tracked by git)

│   └── kinetics\_mini/

│       └── val/

│           ├── archery/

│           ├── bowling/

│           ├── flying\_kite/

│           ├── high\_jump/

│           └── marching/

│

└── stage1\_results/                # Auto-generated outputs

&#x20;   ├── stage1\_report.json

&#x20;   ├── aim\_dictionary\_stage1.json

&#x20;   └── figures/

```



\---



\## Four-Stage Research Roadmap



This project follows a progressive integration strategy. Each stage has explicit pass/fail criteria before proceeding.



| Stage | Goal | V-JEPA 2 | Status |

|---|---|---|---|

| \*\*Stage 1\*\* | Verify AIM symbols capture semantic differences | Frozen (no training) | ✅ In progress |

| Stage 2 | Quantizer warm-up with stable codebook | Frozen | Planned |

| Stage 3 | Joint training with symmetric quantization | Fine-tuned | Future work |

| Stage 4 | Action-conditioned symbolic world model | Fine-tuned | Future work |



\*\*Stage 1 is the focus of this repository.\*\* The encoder is completely frozen — no V-JEPA 2 weights are modified. Only the lightweight AIM quantizer is trained.



\### Stage 1 Pass Criteria



| Metric | Threshold | Meaning |

|---|---|---|

| H1 Symbol Stability | > 95% | Same video → same symbol across 20 repeats |

| H2 Chi-square p-value | < 0.01 | Symbol distributions differ significantly across conditions |

| H2 MI Ratio | > 5× baseline | Experiment MI >> random noise MI |

| Codebook Active Ratio | > 30% | No codebook collapse |



\---



\## Model Download



\### Option A: HuggingFace (Recommended)



The code loads models automatically from HuggingFace on first run. No manual download needed.



```python

\# Handled automatically inside stage1\_diagnosis.py

\# ViT-L is used by default (--encoder\_embed\_dim 1024)

```



\### Option B: Direct Download



Place checkpoint files in the `checkpoints/` directory.



\*\*ViT-L\*\* (recommended for Stage 1, \~1.2 GB):

```bash

mkdir -p checkpoints

wget https://dl.fbaipublicfiles.com/vjepa2/vitl.pt -O checkpoints/vjepa2\_vitl.pt

```



\*\*ViT-g\*\* (larger, \~10 GB):

```bash

mkdir -p checkpoints

wget https://dl.fbaipublicfiles.com/vjepa2/vitg.pt -O checkpoints/vjepa2\_vitg.pt

```



\### Which Model to Use



| Model | embed\_dim | File Size | Recommended For |

|---|---|---|---|

| ViT-L | 1024 | \~1.2 GB | Stage 1 (concept validation) |

| ViT-g | 1408 | \~10 GB | Full experiments |



For Stage 1, ViT-L is sufficient. Use `--encoder\_embed\_dim 1024` when running with ViT-L.



\---



\## Installation



```bash

git clone https://github.com/YOUR\_USERNAME/aim-vjepa2.git

cd aim-vjepa2

pip install -r requirements.txt

```



For video decoding, install at least one of:

```bash

pip install torchcodec   # recommended

\# OR

pip install decord

\# OR

pip install opencv-python  # fallback

```



\---



\## Quick Start



\### Step 1: Verify the pipeline (no model or video needed)



```bash

cd aim\_bridge

python stage1\_diagnosis.py --run\_mock

```



Expected output: all 4 criteria pass (H1, H2-chi2, H2-MI, codebook).



\### Step 2: Download test videos



```python

from huggingface\_hub import snapshot\_download

snapshot\_download(

&#x20;   repo\_id="nateraw/kinetics-mini",

&#x20;   repo\_type="dataset",

&#x20;   local\_dir="./data/kinetics\_mini",

&#x20;   allow\_patterns="val/\*/\*.mp4"

)

```



\### Step 3: Run Stage 1 diagnosis



```bash

\# Using ViT-L (default, recommended)

python stage1\_diagnosis.py --run\_with\_kinetics \\

&#x20;   --video\_root ../data/kinetics\_mini/val \\

&#x20;   --encoder\_embed\_dim 1024 \\

&#x20;   --device cpu \\

&#x20;   --output\_dir ../stage1\_results



\# Using ViT-g (if downloaded)

python stage1\_diagnosis.py --run\_with\_kinetics \\

&#x20;   --video\_root ../data/kinetics\_mini/val \\

&#x20;   --encoder\_embed\_dim 1408 \\

&#x20;   --device cpu \\

&#x20;   --output\_dir ../stage1\_results

```



\### Step 4: Analyze the dictionary



```bash

cd ../aim

python analyze\_aim.py \\

&#x20;   --dict\_path ../stage1\_results/aim\_dictionary\_stage1.json \\

&#x20;   --K\_val 64 \\

&#x20;   --top\_N\_aim 10

```



\---



\## Outputs



After Stage 1 completes, results are saved to `stage1\_results/`:



```

stage1\_results/

├── stage1\_report.json              # Pass/fail for all 4 criteria

├── aim\_dictionary\_stage1.json      # Initial AIM symbol dictionary

└── figures/

&#x20;   ├── stage\_a\_training.png        # Quantizer training curve

&#x20;   ├── intervention\_grasp\_angle.png    # JSD heatmap + symbol distributions

&#x20;   ├── intervention\_object\_geometry.png

&#x20;   ├── intervention\_motion\_speed.png

&#x20;   └── stage1\_summary.png          # Overall pass/fail summary

```



\---



\## Related Work



\- \*\*V-JEPA 2\*\*: \[Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985) — Assran et al., Meta FAIR, 2025

\- \*\*AIM Framework\*\*: \[AI Mother Tongue: Self-Emergent Communication in MARL via Endogenous Symbol Systems](https://arxiv.org/abs/2507.10566) — Liu, 2025

\- \*\*DRCB\*\*: Dynamic Representational Circuit Breaker for steganographic collusion detection in MARL (companion paper)



\---



\## Notes



\- This project does \*\*not\*\* modify V-JEPA 2 weights in Stage 1. The encoder is completely frozen.

\- The AIM quantizer trained in Stage 1 operates on frozen features and takes approximately 1–3 hours on CPU (ViT-L).

\- Model checkpoint files (`.pt`, `.pth`) are excluded from version control via `.gitignore`.

\- Stage 1 uses \[Kinetics-mini](https://huggingface.co/datasets/nateraw/kinetics-mini) as a proof-of-concept dataset. Limitations of this dataset (confounding variables across action classes) are acknowledged; controlled physical experiments are planned as future work.



\---



\## License



MIT

