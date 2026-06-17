# PaliGemma2 · ScienceQA MLOps

Fine-tuning **PaliGemma2-3B** (LoRA) on the **ScienceQA** image subset to answer
multiple-choice science questions, with a full MLOps pipeline around it:
reproducible data (DVC), config-driven training (Hydra + Lightning), experiment
tracking and sweeps (W&B), cloud training and evaluation (Vertex AI), and
serving (FastAPI on Cloud Run).

## What the model does

Given a question, an image, answer choices, and optional hint/lecture, the model
generates the **answer letter** (A/B/C/…). Training and evaluation score
exact-match on that letter.

## Results at a glance

| Model | Test accuracy |
|---|---|
| Pre-fix baseline (truncated prompt) | 42.9% |
| Baseline (prompt fix) | 58.85% |
| Sweep winner `vague-sweep-3` (current production) | **64.1%** |

A retrain on the full ScienceQA train split (~6,218 examples, LoRA r=16) is
underway to push this higher. See `reports/RESULTS.md` for the full sweep
table, per-subject breakdown, and figures.

## Live demo

The serving API runs on Cloud Run (CPU, scale-to-zero):

```bash
curl https://paligemma-api-581237630637.europe-west4.run.app/
```

A Streamlit frontend (`scipali.serving.frontend`) provides a UI over `/predict`.

See [Architecture](architecture.md) for how the pieces fit together and
[Usage](usage.md) for how to run each stage.
