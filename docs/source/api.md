# API design

The serving API is a small FastAPI app (`project_name.api`) with three
endpoints: a health check, single-sample inference, and a data-drift check.
The full request/response models are validated by Pydantic and published as
OpenAPI at `/openapi.json`, with interactive docs at `/docs` (Swagger) and
`/redoc`.

## Endpoints

| Method & path        | Purpose                          | Body / params                         | Returns |
|----------------------|----------------------------------|---------------------------------------|---------|
| `GET /`              | Health + model-loaded flag       | —                                     | `{status, model_loaded}` |
| `POST /predict`      | Inference on one ScienceQA item  | `PredictRequest` (JSON)               | `{prediction}` — the answer letter |
| `GET /monitor/drift` | Evidently input data-drift check | `current_gcs` (optional gs:// override) | `DriftResponse` |

### `POST /predict`

```jsonc
// request — PredictRequest
{
  "question": "Which property do these objects have in common?",  // required, ≥1 char
  "choices": ["soft", "salty", "sticky"],                          // required, 2–10 items
  "hint": "",                                                      // optional, appended to prompt
  "lecture": "",                                                   // optional, appended to prompt
  "image_b64": "<base64 JPEG/PNG>",                                // required
  "max_new_tokens": 10                                             // optional, 1–128
}
// response — PredictResponse
{ "prediction": "C" }   // a single normalised answer letter (A/B/C/…)
```

### `GET /monitor/drift`

```json
{
  "dataset_drift": true,
  "n_drifted_columns": 7,
  "n_columns": 7,
  "reference_rows": 6218,
  "current_rows": 6,
  "current_source": "gs://mlops-paligemma-west4/monitoring/current_production.csv"
}
```

`current_source` reports which "current" table was actually compared, so the
verdict is never silently a self-comparison (see the drift loop below).

## Request flow & drift loop

```mermaid
flowchart LR
    client["Client"] -->|"POST /predict (image_b64 + question)"| api["FastAPI /predict<br/>(Cloud Run, CPU)"]
    api -->|"lazy-load adapter"| model["PaliGemma2 + LoRA<br/>from gs://…/models/production"]
    model -->|answer letter| client
    api -->|"one JSON line/request"| logs["Cloud Logging<br/>(jsonPayload)"]
    logs -->|"monitoring collect"| prod[("GCS current_production.csv")]
    ref[("GCS reference.csv<br/>seed-reference")] --> drift["GET /monitor/drift<br/>(Evidently)"]
    prod --> drift
    drift -->|"dataset_drift + current_source"| client
```

## Design choices

- **On-demand, scale-to-zero serving (no always-on GPU).** Cloud Run runs the
  3B model on CPU (`--memory 32Gi --cpu 8`), `min-instances 0`, with **lazy
  loading** (`LAZY_LOAD=1`): the container passes the startup probe immediately
  and loads the model on the first `/predict`. First call per instance is
  ~160 s; warm calls ~10–27 s. Cheaper than a standing GPU endpoint for a
  course-scale workload.
- **The adapter is read from `gs://…/models/production` at startup**, not baked
  into the image, so promoting a new model needs no rebuild/redeploy — re-copy
  the adapter and move the W&B `production` alias.
- **One inference per instance** (`concurrency 1`) avoids OOM on the 3B model;
  `max-instances 3` lets overflow spin a new instance instead of returning 429.
- **Drift tables live in GCS, not the image** (`reference.csv`,
  `current_production.csv`, `current_sample.csv`), so the input distribution can
  be refreshed without a redeploy. `reference.csv` is regenerable from committed
  code via `monitoring seed-reference`; `current_production.csv` is materialised
  from real traffic by `monitoring collect`.
- **`/predict` logs one structured single-line JSON per request** (severity
  `INFO`) straight to stdout, *not* through the Rich logger. Rich renders at a
  fixed 80-column width and Cloud Run stores each wrapped line as a separate
  entry, which would fragment the JSON and break `collect`. A lone JSON object
  is parsed by Cloud Run into `jsonPayload` and read back intact, closing the
  drift loop end-to-end.
- **The drift comparison uses only the columns both tables share.** Production
  inputs can't carry train-only columns (e.g. `subject`, unknown at inference
  time), so the endpoint intersects columns rather than erroring or registering
  spurious drift.
