# scipali

PaliGemma2-3B fine-tuned (LoRA) on ScienceQA-IMG, wrapped in a full MLOps
pipeline — DVC data versioning, Hydra configs, W&B sweeps, Vertex AI training,
Cloud Run serving, drift monitoring, and CI/CD. Headline: **72.19%** exact-match
accuracy on the 2,017-sample test split. Full results in
[`reports/RESULTS.md`](reports/RESULTS.md); usage in
[`docs/source/usage.md`](docs/source/usage.md).

## Project description

### Overall goal of the project
The goal of the project is to develop techniques that improve reasoning accuracy
using the PaliGemma foundation model.

### What framework are you going to use (Kornia, Transformer, Pytorch-Geometrics)
The Hugging Face **Transformers** framework (pretrained PaliGemma2), with
**PEFT/LoRA** for parameter-efficient fine-tuning, **PyTorch Lightning** for the
training loop, and **Hydra** for configuration management.

### How to you intend to include the framework into your project
We utilise one of the strengths of the Transformers framework — thousands of
pretrained models — by starting from a pretrained PaliGemma2 checkpoint and
fine-tuning it on our data, then improving from there.

### What data are you going to run on (initially, may change)
We use **`derek-thomas/ScienceQA`** (the image subset, "ScienceQA-IMG"). We
initially planned `lmms-lab/ScienceQA`, but that mirror ships no train split, so
we switched. Splits: train 6,218 / val 2,097 / test 2,017. Each sample has an
image, a question, answer choices, the answer index, and optional hint / lecture
plus a subject label.

### What deep learning models do you expect to use
The **PaliGemma2-3B** vision-language model (`google/paligemma2-3b-pt-224`),
LoRA-adapted on the language-model attention projections with the vision encoder
frozen.

## Project structure

```txt
├── .github/workflows/      # CI: tests, linting, docs, data-change, model-registry
├── cloud/                  # Vertex AI + Cloud Build + ops scripts
├── configs/                # Hydra configs (data / model / trainer / sweep)
├── data/                   # DVC-tracked dataset (git-tracked pointers; data on GCS)
├── dockerfiles/            # api / train / predict images
├── docs/                   # MkDocs site
├── reports/                # figures, eval, profiling, monitoring, load + RESULTS.md
├── src/scipali/
│   ├── data/               # data.py, profile_data.py
│   ├── models/             # model.py, train.py, evaluate.py, optimize.py, visualize.py
│   ├── serving/            # api.py, predict.py, frontend.py, bento_service.py
│   └── monitoring/         # monitoring.py
├── tests/                  # pytest suite
├── pyproject.toml
└── tasks.py                # invoke tasks
```

## Serving

The FastAPI service (`src/scipali/serving/api.py`, image: `dockerfiles/api.dockerfile`)
serves single-sample ScienceQA predictions from the **production adapter**.

`CHECKPOINT_PATH` accepts a local adapter dir, a `.ckpt` file, or a `gs://` directory —
the stable production path is fetched at startup, so promoting a new adapter
(copy to GCS + W&B `production` alias) requires **no rebuild or redeploy**:

```bash
# local (model weights cached from HF; needs HF access for the gated base model)
CHECKPOINT_PATH=gs://mlops-paligemma-west4/models/production \
  uvicorn scipali.serving.api:app --host 0.0.0.0 --port 8000
```

**Deployment decision (2026-06-12):** demo-grade serving runs locally or as a
container on demand, NOT as an always-on cloud endpoint. Rationale: PaliGemma2-3B
needs a GPU for interactive latency; an always-on L4 endpoint (Vertex endpoint or
Cloud Run w/ GPU) costs more than this course project justifies, and Cloud Run CPU
inference (~minutes/request) times out for real use. The `gs://` startup fetch
keeps the container cloud-ready: `gcloud run deploy --image <api image>
--set-env-vars CHECKPOINT_PATH=gs://mlops-paligemma-west4/models/production`
is the documented path if an always-on endpoint is ever needed.

---

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template), a
[cookiecutter template](https://github.com/cookiecutter/cookiecutter) for MLOps.
