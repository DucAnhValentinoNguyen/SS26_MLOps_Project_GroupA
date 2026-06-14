# Results — PaliGemma2-3B fine-tuned on ScienceQA (image subset)

> **STATUS (2026-06-14).** The full-data r=16 retrain produced a **better model
> — `autumn-sweep-2`, test 0.713 (W&B artifact v11)** — but the GCP billing
> account closed mid-sweep, so it was **not yet promoted**. The **currently
> deployed** model is still `vague-sweep-3` (test 0.641, r=8, old data). Once
> billing is restored: promote `autumn-sweep-2`, re-run `evaluate.py` for the
> canonical per-subject `eval_results.json`, and regenerate the figures. Both
> adapters are backed up locally at `~/mlops-adapter-backup/` (W&B-sourced,
> independent of GCP billing).

Self-contained results summary for the exam report (paste into Q12/Q14/Q17).
All numbers are exact-match accuracy of the generated answer **letter** on the
held-out ScienceQA-IMG test split (2017 samples).

## Headline

| Model | Data | LoRA | Test accuracy | Status |
|---|---|---|---|---|
| Pre-fix baseline | old | r=8 | 42.9% | prompt truncated the choices |
| Sweep #1 winner | old | r=8 | 58.85% | prompt fix; `base_lr` 1e-4 |
| **`vague-sweep-3`** (sweep #2 winner) | old (1,677) | r=8 | **64.1%** (1293/2017) | **currently deployed** |
| **`autumn-sweep-2`** (full-data retrain) | **full (6,218)** | **r=16** | **71.3%** | **best; pending promotion** (W&B v11) |

Headline finding: **fixing the data pipeline (1,677 → 6,218 train examples) +
raising LoRA rank (8 → 16) lifted test accuracy 64.1% → 71.3% (+7.2 pts)** on the
same 2017-sample test set. 80%+ would need higher resolution (448) / unfreezing
the vision encoder / chain-of-thought — not attempted here.

The *deployed* adapter is still `vague-sweep-3` at
`gs://mlops-paligemma-west4/models/production/` (W&B `:production`, v3); the
better `autumn-sweep-2` (v11) awaits promotion once billing is restored.

## Winning hyperparameters (`vague-sweep-3`)

| Hyperparameter | Value | Swept? |
|---|---|---|
| `model.base_learning_rate` | 1.89e-4 | yes (log-uniform 7e-5–2e-4) |
| effective learning rate | 1.89e-4 | derived: `base × √(eff_batch/16)` |
| `data.batch_size` | 4 | fixed |
| `trainer.accumulate_grad_batches` | 4 | yes ({2,4,8}) |
| effective batch size | 16 | = batch_size × accum |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 | fixed |
| LoRA target modules | q,k,v,o\_proj | fixed |
| vision encoder | frozen | fixed |
| gradient checkpointing | on | fixed |
| `max_length` | 512 | fixed |
| epochs | ≤ 8, EarlyStopping(patience=3) on `val/accuracy` (max) | fixed |

The learning rate is decoupled from accumulation via a √ batch-size rule
(`resolve_learning_rate` in `train.py`): the sweep searches `base_learning_rate`
defined at a reference effective batch of 16, so trials at different
accumulation are compared at a comparable LR.

## Sweep #2 — all trials (W&B sweep `xptwdnis`, Bayesian, metric `val/accuracy` max)

| Run | val/accuracy | val/loss | base_lr | accum (eff. batch) |
|---|---|---|---|---|
| **vague-sweep-3** (winner) | **0.7024** | 0.5111 | 1.89e-4 | 4 (16) |
| devout-sweep-7 | 0.6738 | 0.6007 | 1.78e-4 | 4 (16) |
| vague-sweep-4 | 0.6690 | 0.5303 | 1.96e-4 | 8 (32) |
| comfy-sweep-1 | 0.6500 | 0.6480 | 1.88e-4 | 2 (8) |
| playful-sweep-2 | 0.6381 | 0.6487 | 1.11e-4 | 2 (8) |
| daily-sweep-6 | 0.6357 | 0.5477 | 1.82e-4 | 8 (32) |
| azure-sweep-8 | 0.6310 | 0.7022 | 8.31e-5 | 2 (8) |
| dutiful-sweep-5 | 0.6190 | **0.4643** | 8.22e-5 | 8 (32) |

## Per-subject accuracy (deployed model `vague-sweep-3`)

| Subject | Accuracy | n |
|---|---|---|
| social science | 76.2% | 764 |
| natural science | 57.2% | 1209 |
| language science | 45.5% | 44 |

(Per-subject for the better `autumn-sweep-2` awaits the canonical re-eval —
blocked on billing. Natural science is the weak split; it's the most
diagram-dependent, which is why 224-resolution caps it.)

## Methodology note — why we optimise `val/accuracy`, not `val/loss`

Sweep #1 optimised `val/loss` and promoted a trial that lost to the baseline on
test accuracy. Sweep #2 confirms why: the two metrics **disagree**.
`dutiful-sweep-5` has the *best* `val/loss` (0.464) but nearly the *worst*
`val/accuracy` (0.619); the winner has a *higher* loss (0.511) but the *best*
accuracy (0.702). Because the task is scored on exact-match of the answer
letter, we log a generation-based `val/accuracy` each epoch and select
checkpoints / early-stop on it (`mode=max`). See `reports/figures/sweep3_comparison.png`
(full-data r=16 sweep) and `sweep2_comparison.png` (earlier).

The LR pattern also held: trials at `base_lr ≈ 1.8–1.96e-4` reached
0.65–0.70 `val/accuracy`; the two low-LR trials (~8e-5) sat at the bottom —
which is why sweep #2 raised the LR floor above sweep #1's dead zone.

## Full-data retrain (r=16, W&B sweep `win9arpw`)

After switching the data source to `derek-thomas/ScienceQA` (the lmms-lab mirror
ships no train split, which had forced carving "train" out of validation), the
real splits are train 6,218 / val 2,097 / test 2,017, and LoRA rank was raised
8 → 16. A baseline + Bayesian sweep (metric `val/accuracy`) ran; the GCP billing
account closed mid-sweep (after trial 7), but the completed trials' adapters and
metrics are preserved in W&B.

| Run | test acc | val/acc | base_lr | accum (eff.) | state |
|---|---|---|---|---|---|
| **autumn-sweep-2** | **0.7129** | 0.699 | 1.33e-4 | 4 (16) | finished — best, **W&B v11** |
| rosy-sweep-1 | 0.6981 | 0.650 | 8.36e-5 | 4 (16) | finished |
| daily-sweep-5 | 0.6926 | 0.662 | 8.00e-5 | 2 (8) | failed\* |
| misunderstood-sweep-3 | 0.6564 | 0.663 | 1.33e-4 | 4 (16) | finished |
| sunny-sweep-6 | 0.6559 | 0.634 | 9.09e-5 | 4 (16) | failed\* |
| neat-sweep-4 | 0.6554 | 0.571 | 1.82e-4 | 2 (8) | finished |
| sandy-sweep-7 | — | 0.714 | 1.33e-4 | 4 (16) | killed by billing (top val, no test) |
| baseline | 0.6401 | 0.616 | 1.0e-4 | 4 (16) | finished (no sweep) |

\*W&B marks some trials `failed` (interrupted), yet they completed training + a
test eval before exiting; their artifacts/metrics are intact.

Winner `autumn-sweep-2`: r=16, alpha=32, base_lr 1.33e-4, accum 4 (eff. batch
16); same frozen-vision / `max_length` 512 / EarlyStopping-on-`val/accuracy`
setup. `sandy-sweep-7` had the highest *val* (0.714) but was cut off before its
test eval, so `autumn-sweep-2` is the best *completed* model.

## Distributed training & data loading (M29 / M30)

Both are "if applicable" and are **not applicable** at this scale:
- **M30 (distributed training):** training runs on a **single L4**. LoRA on
  PaliGemma2-3B (~6.4 M trainable params, ~3 B frozen) fits one GPU, so
  multi-GPU DDP would add complexity with no benefit. PyTorch-Lightning would
  enable it via `Trainer(devices=…, strategy="ddp")` if more GPUs were available.
- **M29 (distributed data loading):** we use a **multi-worker `DataLoader`**
  (`data.num_workers`) — the relevant loading optimisation here; sharded
  loading is unnecessary for a single-GPU job over a ~700 MB processed dataset.

## Artifact layout (`reports/`)

| Folder | Contents |
|---|---|
| `figures/` | `.png` visualizations (below) |
| `eval/` | eval data: `production_eval_results.json`, `sweep2_summary.json`, `sweep3_summary.json` |
| `monitoring/` | `drift_report.html` (Evidently) |
| `load/` | load-test summary + locust CSVs |

### Figures (`reports/figures/`)

| File | Shows |
|---|---|
| `accuracy_by_subject.png` | deployed model (`vague-sweep-3`) per-subject accuracy |
| `sweep3_comparison.png` | **full-data r=16 sweep (`win9arpw`)**: per-trial val/accuracy + the val/loss↔val/accuracy disagreement (winner `autumn-sweep-2` → test 71.3%) |
| `sweep2_comparison.png` | earlier sweep (`xptwdnis`, old data r=8) — same chart for the 64% era |
| `prediction_length_dist.png` | predicted answer length (sanity: single letters) |
| `error_samples.png` | qualitative grid of misclassified samples |

Reproduce with the committed source JSONs:

```bash
python -m project_name.visualize subject-accuracy reports/eval/production_eval_results.json
python -m project_name.visualize sweep-comparison  reports/eval/sweep3_summary.json   # full-data
python -m project_name.visualize sweep-comparison  reports/eval/sweep2_summary.json   # earlier
python -m project_name.visualize pred-lengths       reports/eval/production_eval_results.json
```

## Cloud workload inventory (what runs where, and why)

Every GPU-bound workload runs on **Vertex AI custom jobs** (single L4,
`europe-west4`, Flex Start to queue for capacity). Everything else is CPU and
stays local/CI by design.

| Workload | Where | Entry point |
|---|---|---|
| Training (baseline) | Vertex L4 | `cloud/run_baseline_and_sweep.sh` (`SKIP_BASELINE=0`) |
| Hyperparameter sweep | Vertex L4 | same script → `wandb agent` |
| Best-adapter eval (chained) | Vertex L4 | same script, step [3/3] |
| Standalone adapter eval | Vertex L4 | `cloud/run_eval.sh` (any `ADAPTER_GCS`) |
| Image build | Cloud Build | `cloud/cloudbuild.train.yaml` |
| Serving / inference | on-demand container (local or Cloud Run) | `dockerfiles/api.dockerfile` |
| Data preprocessing | local / CI | `project_name.data` (CPU: resize + tokenise) |
| Report figures | local | `project_name.visualize` (reads eval JSON) |

Notes:
- Secrets (W&B key, HF token) are fetched at container start from Secret
  Manager via google-auth ADC (`cloud/fetch_secrets.sh`); job specs carry only
  secret **names**. Jobs run as the compute service account, which holds
  `secretmanager.secretAccessor`.
- Job images are pinned by **digest** at submit time (Vertex resolves `:latest`
  at container start, which can drift across a long Flex Start queue).
- Serving is intentionally **not** an always-on GPU endpoint: a 3B model needs a
  GPU for interactive latency, and an always-on L4 endpoint costs more than this
  project warrants. The API reads its adapter from `CHECKPOINT_PATH`, which
  accepts a `gs://` path, so promoting a new adapter needs no redeploy.
