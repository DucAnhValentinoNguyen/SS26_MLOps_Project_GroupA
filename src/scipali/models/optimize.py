"""Inference optimization benchmark (M31): quantization + compilation.

Loads the fine-tuned adapter in a few configurations and measures load time,
peak GPU memory, and per-generate latency on a fixed set of test samples:

- ``bf16``            : the serving default.
- ``int4``           : 4-bit weights via bitsandbytes (QLoRA-style) — CUDA only.
- ``bf16+compile``   : ``torch.compile`` of the bf16 model.

Pruning is intentionally skipped: unstructured pruning a LoRA-adapted model
gives no inference speedup without sparse kernels, so it isn't worth the report
space.

Runs on a CUDA GPU (4-bit needs bitsandbytes/CUDA). On Vertex via
``cloud/run_optimize.sh``. Results saved as JSON.
"""

import json
import logging
import time
from pathlib import Path

import torch
import typer
from peft import PeftModel
from rich.logging import RichHandler
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from scipali.data.data import DATASET_SUBSET, PROCESSED_DATA_DIR, DataModule
from scipali.models.model import MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
log = logging.getLogger(__name__)
app = typer.Typer(help="Benchmark quantization/compilation of the fine-tuned model.")


def _sample_batch(adapter_dir: Path, n: int):
    """Build one eval batch of ``n`` samples plus the matching processor."""
    processor = AutoProcessor.from_pretrained(str(adapter_dir))
    data = DataModule(
        processed_dir=PROCESSED_DATA_DIR,
        subset=DATASET_SUBSET,
        processor=processor,
        batch_size=n,
        num_workers=0,
    )
    data.setup()
    batch = next(iter(data.test_dataloader()))
    return batch


def _load(adapter_dir: Path, mode: str):
    """Load base + adapter in the requested mode; return (model, load_seconds)."""
    kwargs: dict = {"torch_dtype": torch.bfloat16}
    if mode == "int4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
        kwargs["device_map"] = "cuda"
    t = time.time()
    base = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_NAME, **kwargs)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    if mode != "int4":
        model = model.to("cuda")
    if mode == "bf16+compile":
        model = torch.compile(model)  # type: ignore[assignment]
    model.eval()
    return model, time.time() - t


@app.command()
def benchmark(
    adapter_dir: Path = typer.Argument(..., help="LoRA adapter directory."),
    n_samples: int = typer.Option(8, help="Samples per generate call."),
    iters: int = typer.Option(5, help="Timed generate iterations (after warmup)."),
    output_path: Path = typer.Option(Path("optimize_results.json")),
) -> None:
    """Benchmark bf16 vs int4 vs bf16+compile and save a results table."""
    if not torch.cuda.is_available():
        typer.echo("CUDA required (4-bit + meaningful latency need a GPU).", err=True)
        raise typer.Exit(code=1)

    batch = _sample_batch(adapter_dir, n_samples)
    gen_kwargs = dict(
        input_ids=batch["input_ids"].to("cuda"),
        attention_mask=batch["attention_mask"].to("cuda"),
        pixel_values=batch["pixel_values"].to("cuda", torch.bfloat16),
        max_new_tokens=10,
        do_sample=False,
    )

    results = []
    for mode in ("bf16", "int4", "bf16+compile"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model, load_s = _load(adapter_dir, mode)
        with torch.inference_mode():
            model.generate(**gen_kwargs)  # warmup (also triggers compile)
            t = time.time()
            for _ in range(iters):
                model.generate(**gen_kwargs)
            latency = (time.time() - t) / iters
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        row = {
            "mode": mode,
            "load_s": round(load_s, 1),
            "latency_s_per_batch": round(latency, 3),
            "latency_s_per_sample": round(latency / n_samples, 3),
            "peak_gpu_gb": round(peak_gb, 2),
        }
        results.append(row)
        log.info("%s", row)
        del model

    output_path.write_text(
        json.dumps({"n_samples": n_samples, "results": results}, indent=2)
    )
    log.info("Saved benchmark to %s", output_path)


if __name__ == "__main__":
    app()
