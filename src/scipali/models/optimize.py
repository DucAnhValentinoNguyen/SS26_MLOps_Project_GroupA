"""Inference optimization benchmarks for the fine-tuned model.

Two Typer commands, both run on a CUDA GPU (4-bit needs bitsandbytes/CUDA), on
Vertex via ``cloud/run_optimize.sh``. Results saved as JSON.

``benchmark`` loads the adapter in a few configurations and measures load time,
peak GPU memory, and per-generate latency on a fixed set of test samples:

- ``bf16``            : the serving default.
- ``int4``           : 4-bit weights via bitsandbytes (QLoRA-style) — CUDA only.
- ``bf16+compile``   : ``torch.compile`` of the bf16 model.

``prune-sweep`` merges the LoRA adapter into the base, then global
magnitude-prunes the Linear weights to several sparsity levels and measures test
accuracy at each. Latency is reported too, but it stays flat by design:
unstructured pruning only zeros weights, so the dense kernels still do the full
matmul — there is no speedup without sparse kernels. The deliverable is the
accuracy-vs-sparsity curve, not a latency win.
"""

import gc
import json
import logging
import time
from pathlib import Path

import torch
import typer
from peft import PeftModel
from rich.logging import RichHandler
from torch.nn.utils import prune
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from scipali.data.data import DATASET_SUBSET, PROCESSED_DATA_DIR, DataModule
from scipali.models.model import MODEL_NAME, extract_answer_letter

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


def prune_linear_layers(model: torch.nn.Module, amount: float) -> float:
    """Global L1-unstructured prune of every ``nn.Linear`` weight to ``amount``.

    Uses a single global magnitude threshold across all Linear weights (so the
    sparsity budget is allocated adaptively, like ``prune.global_unstructured``),
    but finds the cutoff with ``torch.kthvalue`` on a pooled CPU copy of the
    ``|weights|``. kthvalue returns just the scalar threshold, avoiding the
    ~10GB int64 top-k *index* tensor that OOMs the 24GB L4 on a 3B model (the
    boolean masks below are ~1 byte/elem, which is cheap). Each layer is masked
    with ``|w| > threshold`` and baked in with ``prune.remove``.

    Returns the ACHIEVED sparsity (fraction with ``|w| <= threshold``). With a
    smooth magnitude distribution this matches ``amount`` to well within 1%; the
    only deviation is weights tied exactly at the threshold, which can nudge it
    up by at most the size of that one magnitude bin. The sweep records this
    achieved value, so the curve is plotted against real sparsity. ``amount == 0``
    is a no-op.
    """
    linears = [(m, "weight") for m in model.modules() if isinstance(m, torch.nn.Linear)]
    total = sum(m.weight.numel() for m, _ in linears)
    if amount > 0:
        # Pool |weights| into one preallocated CPU float32 buffer (so we never
        # hold two ~10GB copies at once) and take the global magnitude cutoff.
        pooled = torch.empty(total, dtype=torch.float32)
        offset = 0
        for module, _ in linears:
            n = module.weight.numel()
            pooled[offset : offset + n] = (
                module.weight.detach().abs().float().flatten().cpu()
            )
            offset += n
        threshold = pooled.kthvalue(int(amount * total)).values.item()
        del pooled
        for module, name in linears:
            mask = (module.weight.detach().abs() > threshold).to(module.weight.dtype)
            prune.custom_from_mask(module, name, mask)  # 1 = keep, 0 = prune
            prune.remove(module, name)  # bake the zeros into the dense weight
    zeros = sum(int((m.weight == 0).sum()) for m, _ in linears)
    return zeros / total if total else 0.0


def _load_merged(adapter_dir: Path):
    """Load base+adapter in bf16, merge LoRA into the base, return a CUDA model.

    ``low_cpu_mem_usage`` keeps the CPU load peak to ~1x the model size (it
    streams weights in instead of allocating a second full copy for the state
    dict) -- the sweep reloads this once per sparsity level, so without it the
    host RAM climbs and the job hits "Replicas low on memory".
    """
    base = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    return model.to("cuda").eval()


def _score_accuracy(model, processor, loader, n_batches: int) -> tuple[int, int]:
    """Accuracy over up to ``n_batches`` (0 = all); returns (correct, total).

    Mirrors evaluate.py: generate, decode the continuation, and compare the
    extracted choice letter against the dataset's ``answer_texts``.
    """
    correct = total = 0
    for batch_idx, batch in enumerate(loader):
        if n_batches and batch_idx >= n_batches:
            break
        input_ids = batch["input_ids"].to("cuda")
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"].to("cuda"),
            pixel_values=batch["pixel_values"].to("cuda", torch.bfloat16),
            max_new_tokens=10,
            do_sample=False,
        )
        preds = processor.batch_decode(
            generated_ids[:, input_ids.shape[1] :], skip_special_tokens=True
        )
        for pred, target in zip(preds, batch["answer_texts"]):
            correct += int(extract_answer_letter(pred) == extract_answer_letter(target))
            total += 1
    return correct, total


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


@app.command()
def prune_sweep(
    adapter_dir: Path = typer.Argument(..., help="LoRA adapter directory."),
    sparsities: str = typer.Option(
        "0.0,0.3,0.5,0.7", help="Comma-separated target sparsity levels."
    ),
    n_batches: int = typer.Option(
        0, help="Test batches scored per level (0 = whole test split)."
    ),
    batch_size: int = typer.Option(8, help="Eval/latency batch size."),
    iters: int = typer.Option(5, help="Timed generate iterations for latency."),
    output_path: Path = typer.Option(Path("prune_results.json")),
) -> None:
    """Prune the merged model to each sparsity and measure accuracy + latency.

    Accuracy is the headline (pruning degrades the adapter, which was trained on
    the un-pruned base); latency is reported only to confirm it does not drop.
    """
    if not torch.cuda.is_available():
        typer.echo("CUDA required (bf16 generate + meaningful latency).", err=True)
        raise typer.Exit(code=1)

    levels = [float(s) for s in sparsities.split(",") if s.strip()]
    processor = AutoProcessor.from_pretrained(str(adapter_dir))
    data = DataModule(
        processed_dir=PROCESSED_DATA_DIR,
        subset=DATASET_SUBSET,
        processor=processor,
        batch_size=batch_size,
        num_workers=2,
    )
    data.setup()

    results = []
    for amount in levels:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # Reload + re-merge each level: prune.remove is destructive, so every
        # sparsity must start from the clean baseline weights.
        model = _load_merged(adapter_dir)
        achieved = prune_linear_layers(model, amount)

        with torch.inference_mode():
            correct, total = _score_accuracy(
                model, processor, data.test_dataloader(), n_batches
            )
            batch = next(iter(data.test_dataloader()))
            gen_kwargs = dict(
                input_ids=batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                pixel_values=batch["pixel_values"].to("cuda", torch.bfloat16),
                max_new_tokens=10,
                do_sample=False,
            )
            model.generate(**gen_kwargs)  # warmup
            t = time.time()
            for _ in range(iters):
                model.generate(**gen_kwargs)
            latency = (time.time() - t) / iters

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        row = {
            "sparsity_requested": amount,
            "sparsity_achieved": round(achieved, 4),
            "accuracy": round(correct / total, 4) if total else 0.0,
            "correct": correct,
            "total": total,
            "latency_s_per_batch": round(latency, 3),
            "peak_gpu_gb": round(peak_gb, 2),
        }
        results.append(row)
        log.info("%s", row)
        del model
        gc.collect()  # free the CPU copy before the next level reloads the base

    output_path.write_text(
        json.dumps({"batch_size": batch_size, "results": results}, indent=2)
    )
    log.info("Saved pruning sweep to %s", output_path)


if __name__ == "__main__":
    app()
