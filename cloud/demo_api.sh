#!/usr/bin/env bash
# Shell demo of the deployed ScienceQA API: health -> predict -> drift.
#
# Usage:
#   ./cloud/demo_api.sh                          # auto-uses test sample 0
#   ./cloud/demo_api.sh IMG.png "Question?" "a,b,c"   # bring your own sample
#
# Override the backend with API_URL=... (defaults to the live Cloud Run service).
# The first /predict on a cold instance can take ~160s while the model loads.
set -euo pipefail

API_URL="${API_URL:-https://paligemma-api-581237630637.europe-west4.run.app}"
echo "Backend: ${API_URL}"

echo
echo "[1/3] GET /  (health — model_loaded is False until the first prediction):"
curl -s --max-time 30 "${API_URL}/"; echo

echo
echo "[2/3] POST /predict  (cold start can take ~160s):"
if [ "$#" -ge 3 ]; then
  # Bring-your-own sample: stdlib only, no project env needed.
  PAYLOAD="$(python3 - "$1" "$2" "$3" <<'PY'
import base64, json, sys
img, question, csv = sys.argv[1], sys.argv[2], sys.argv[3]
choices = [c.strip() for c in csv.split(",") if c.strip()]
sys.stderr.write(f"  question: {question}\n  choices:  {choices}\n")
print(json.dumps({
    "question": question,
    "choices": choices,
    "image_b64": base64.b64encode(open(img, "rb").read()).decode(),
}))
PY
)"
else
  # No args: pull test sample 0 from the processed dataset (needs the local venv).
  echo "  (no args — extracting test sample 0 from the processed dataset)"
  PAYLOAD="$(UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/mlops}" \
    uv run --quiet python - <<'PY'
import base64, io, json, sys
from datasets import load_from_disk
from scipali.data.data import DATASET_SUBSET, PROCESSED_DATA_DIR
s = load_from_disk(PROCESSED_DATA_DIR / DATASET_SUBSET)["test"][0]
buf = io.BytesIO(); s["image"].convert("RGB").save(buf, format="PNG")
sys.stderr.write(f"  question: {s['question']}\n  choices:  {list(s['choices'])}\n")
print(json.dumps({
    "question": s["question"],
    "choices": list(s["choices"]),
    "image_b64": base64.b64encode(buf.getvalue()).decode(),
}))
PY
)"
fi

RESP="$(curl -s --max-time 600 -X POST "${API_URL}/predict" \
  -H 'Content-Type: application/json' -d "${PAYLOAD}")"
echo "  response: ${RESP}"
echo "  -> predicted letter: $(printf '%s' "${RESP}" | \
  python3 -c 'import json,sys; print(json.load(sys.stdin).get("prediction","?"))')"

echo
echo "[3/3] GET /monitor/drift  (real production table once traffic is collected):"
curl -s --max-time 180 "${API_URL}/monitor/drift"; echo
