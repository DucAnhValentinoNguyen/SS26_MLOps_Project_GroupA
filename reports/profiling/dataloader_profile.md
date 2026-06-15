# Data-loading profile

**What / why.** "Use profiling to optimize your code". This profiles the
**training** `DataLoader` — the exact pipeline used during fine-tuning — to find
where data-loading time goes and whether it is worth optimizing.

**How (reproducible, CPU-only, no GPU, no network):**

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  uv run python -m project_name.profile_data --n-batches 200 --batch-size 4 --workers 0,2,4,8
```

Tool: Python `cProfile` + wall-clock timing (`src/project_name/profile_data.py`).
Reads the DVC-pulled `data/processed/ScienceQA-IMG` (train = 6218) and the
locally-cached `google/paligemma2-3b-pt-224` processor. Batch size 4 (= training).
Run on the local laptop CPU (macOS, `spawn` start method).

Raw artifacts in this folder: `dataloader.pstats` (binary, open with `pstats`/
`snakeviz`), `dataloader_profile.txt` (top functions), `dataloader_summary.json`
(throughput numbers).

## Result 1 — where the time goes (cProfile, single-process, 200 batches = 2.285 s)

| Stage | Self / cum time | Share | Source |
|---|---|---|---|
| **Image decode** (PIL decompress on dataset access) | 1.017 s self | **~45%** | `datasets/features/image.py:decode_example` → `ImagingDecoder.decode` |
| **Image preprocess** (resize→224 + rescale/normalize) | 0.630 s cum | **~28%** | `transformers …/image_processing_utils.py:preprocess` (resize alone 0.366 s ≈ 16%) |
| **Tokenisation + tensor assembly** | ~0.49 s | **~21%** | `processing_paligemma.py:__call__` (text side) + `_collate` |

The two halves of the loop are almost exactly balanced: the dataset fetch
(`__getitems__`, 1.125 s) is dominated by **raw image decoding**, and the collate
(`_collate`, 1.137 s) is dominated by the **processor's image resize/normalize**;
text tokenisation is a minor tail. **≈77% of data-loading time is image handling,
≈21% is text.**

## Result 2 — throughput vs `num_workers` (wall-clock, 200 batches)

| `num_workers` | samples/s | ms/batch |
|---|---|---|
| 0 | 353.9 | 11.3 |
| 2 | 531.0 | 7.5 |
| **4 (training default)** | 448.4 | 8.9 |
| 8 | 805.3 | 5.0 |

Multi-worker loading helps (best observed ~2.3× at 8 workers) but is **noisy and
non-monotonic** on this CPU box (2 > 4): macOS `spawn` makes each worker re-import
torch/transformers, and at small batch sizes that fixed cost competes with the
parallelism gain. The absolute ranking is machine- and noise-dependent; the
robust takeaway is the **per-batch cost (~11 ms single-process)** and the hotspot
ranking above, not the exact best worker count.

## Findings / optimization conclusions

1. **The loader is image-bound, not text-bound.** ~45% is PIL decoding the stored
   image bytes every epoch and ~28% is resizing them to 224. The concrete
   code-level lever is therefore in `data.py:preprocess()`: **store images
   already resized to 224×224** (and/or as raw arrays) so train-time decode is
   cheaper and the processor's resize becomes a near no-op. That would cut the
   bulk of the ~73% image cost.

2. **…but the loader is not the training bottleneck, so this optimization was not
   needed.** Even single-process loading delivers ~354 samples/s (~11 ms/batch).
   A single LoRA training *step* for a 3-B model (batch 4, seq 512, gradient
   checkpointing) on the L4 is far slower than 11 ms, so data preparation fully
   overlaps with GPU compute — even `num_workers=0` would not starve the GPU.
   The training default `num_workers=4` is a safe choice with headroom; raising
   it would not speed up training.

3. **This empirically supports the M29 decision** (no distributed data loading):
   loading is cheap relative to compute and is not the limiting factor, so
   sharded/distributed loading would add complexity with no throughput benefit.
