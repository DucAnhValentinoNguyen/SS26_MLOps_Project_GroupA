#!/usr/bin/env bash
# Vertex entrypoint for the optimization benchmark: download an adapter from
# GCS and benchmark bf16 vs int4 vs bf16+compile on the L4.
#
# Env:
#   ADAPTER_GCS  (required) gs:// dir with the adapter (e.g. models/production)
#   GCP_PROJECT / *_SECRET_NAME  see cloud/fetch_secrets.sh (HF token for the
#                base model; W&B unused here)
set -euo pipefail

# Unbuffer Python stdout/stderr: otherwise a hard crash (e.g. an OOM kill) loses
# all of the step's block-buffered logs, leaving no traceback in Cloud Logging.
export PYTHONUNBUFFERED=1

: "${ADAPTER_GCS:?set ADAPTER_GCS to the gs:// adapter directory}"
source "$(dirname "$0")/fetch_secrets.sh"

# Upload one result file to GCS now (if running on Vertex and the file exists),
# so a later step crashing can't discard results an earlier step already wrote.
upload_result() {
  local f="$1"
  [ -n "${AIP_MODEL_DIR:-}" ] || return 0
  [ -f "${f}" ] || { echo ">>> ${f} not produced — skipping upload"; return 0; }
  RESULT_FILE="${f}" python - <<'PY'
import os
from pathlib import Path

from scipali.models.train import upload_to_gcs

print("uploaded", upload_to_gcs(Path(os.environ["RESULT_FILE"]), os.environ["AIP_MODEL_DIR"]))
PY
}

echo ">>> fetching DVC-tracked data"
dvc pull -v data/processed/ScienceQA-IMG.dvc

echo ">>> installing bitsandbytes (4-bit; CUDA-only, not in the base image)"
uv pip install --no-cache-dir bitsandbytes

ADAPTER_DIR="checkpoints/opt-adapter"
echo ">>> downloading adapter from ${ADAPTER_GCS}"
ADAPTER_GCS="${ADAPTER_GCS}" ADAPTER_DIR="${ADAPTER_DIR}" python - <<'PY'
import os
from pathlib import Path
from urllib.parse import urlparse

from google.cloud import storage

uri = os.environ["ADAPTER_GCS"]
dest = Path(os.environ["ADAPTER_DIR"])
parsed = urlparse(uri)
prefix = parsed.path.lstrip("/").rstrip("/") + "/"
client = storage.Client()
n = 0
for blob in client.list_blobs(parsed.netloc, prefix=prefix):
    rel = blob.name[len(prefix):]
    if not rel:
        continue
    (dest / rel).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest / rel))
    n += 1
print(f"downloaded {n} files")
PY

echo ">>> benchmarking (bf16 / int4 / bf16+compile)"
python -m scipali.models.optimize benchmark "${ADAPTER_DIR}" --output-path optimize_results.json
upload_result optimize_results.json   # bank these before prune-sweep can crash

echo ">>> pruning ablation (sparsity vs accuracy)"
prune_rc=0
python -m scipali.models.optimize prune-sweep "${ADAPTER_DIR}" \
  --sparsities "${PRUNE_SPARSITIES:-0.0,0.3,0.5,0.7}" \
  --n-batches "${PRUNE_N_BATCHES:-0}" \
  --output-path prune_results.json || prune_rc=$?

upload_result prune_results.json      # upload whatever it produced before the run script exits

exit "${prune_rc}"                    # still surface a prune-sweep failure to Vertex
