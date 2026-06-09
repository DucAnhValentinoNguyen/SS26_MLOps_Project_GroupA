#!/usr/bin/env bash
# Single-GPU "run everything" entrypoint for the Vertex container.
#
# Goal: acquire the scarce L4 ONCE, then run the baseline fine-tune immediately
# followed by an N-trial W&B sweep — so we never wait for GPU capacity more than
# once. The job stays RUNNING through the whole baseline + sweep.
#
# Env vars:
#   SWEEP_COUNT    number of sweep trials after the baseline (default 8)
#   WANDB_PROJECT  W&B project for the sweep (default scienceqa-paligemma2)
#   WANDB_API_KEY  (required) injected from Secret Manager by watch_job.sh
#   HF_TOKEN       (required) injected from Secret Manager by watch_job.sh
#
# Any extra args are forwarded to the BASELINE run as Hydra overrides, e.g.:
#   bash cloud/run_baseline_and_sweep.sh data.batch_size=4 data.num_workers=4
set -euo pipefail

SWEEP_COUNT="${SWEEP_COUNT:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-scienceqa-paligemma2}"

echo ">>> fetching DVC-tracked data"
dvc pull -v

echo ">>> verifying CUDA is visible"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible - image likely has CPU-only torch'; print('CUDA OK:', torch.version.cuda)"

echo ">>> [1/2] baseline fine-tune (run_name=baseline)"
python -m project_name.train \
  trainer.wandb.enabled=true \
  trainer.wandb.run_name=baseline \
  "$@"

echo ">>> [2/2] W&B sweep: ${SWEEP_COUNT} trials on project ${WANDB_PROJECT}"
# Register the sweep and capture the 'wandb agent ENTITY/PROJECT/ID' line it
# prints. tee keeps the full output in the job logs; '|| true' stops pipefail
# from killing us if grep finds nothing (handled by the emptiness check below).
SWEEP_AGENT_CMD="$(wandb sweep --project "${WANDB_PROJECT}" configs/sweep.yaml 2>&1 \
  | tee /dev/stderr | grep -oE 'wandb agent [^[:space:]]+' | tail -1 || true)"
if [ -z "${SWEEP_AGENT_CMD}" ]; then
  echo "!!! could not parse the sweep id from 'wandb sweep' output" >&2
  exit 1
fi

echo ">>> launching: ${SWEEP_AGENT_CMD} --count ${SWEEP_COUNT}"
exec ${SWEEP_AGENT_CMD} --count "${SWEEP_COUNT}"
