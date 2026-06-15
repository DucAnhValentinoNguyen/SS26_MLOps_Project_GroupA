r"""Data-loading profiling: where does training-time data loading spend time?

Profiles the TRAIN ``DataLoader`` — the exact pipeline used during fine-tuning —
on CPU, with no GPU and no network:

- wall-clock throughput at several ``num_workers`` (training uses 4), to show
  how much the multi-worker loader buys over single-process loading;
- a ``cProfile`` breakdown of the single-process collate path (image
  preprocessing + tokenisation), saved for inspection.

Reads the DVC-pulled processed dataset from disk and the locally-cached
PaliGemma2 processor. Run (offline, cache only):

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
        uv run python -m project_name.profile_data dataloader

Outputs land in ``reports/profiling/``.
"""

import cProfile
import io
import json
import logging
import pstats
import time
from pathlib import Path

import typer
from rich.logging import RichHandler
from transformers import AutoProcessor

from project_name.data import DATASET_SUBSET, PROCESSED_DATA_DIR, DataModule
from project_name.model import MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
log = logging.getLogger(__name__)
app = typer.Typer(help="Profile the data-loading pipeline.")

OUT_DIR = Path("reports/profiling")


def _make_train_loader(processor, batch_size: int, num_workers: int):
    """Build the same train DataLoader fine-tuning uses."""
    dm = DataModule(
        processed_dir=PROCESSED_DATA_DIR,
        subset=DATASET_SUBSET,
        processor=processor,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    dm.setup()
    return dm.train_dataloader()


def _time_loader(loader, n_batches: int, warmup: int = 2):
    """Consume ``n_batches`` after a warmup; return (seconds, batches_done)."""
    it = iter(loader)
    for _ in range(warmup):
        next(it)
    t0 = time.perf_counter()
    done = 0
    for _ in range(n_batches):
        try:
            next(it)
        except StopIteration:
            break
        done += 1
    return time.perf_counter() - t0, done


@app.command()
def dataloader(
    n_batches: int = typer.Option(50, help="Timed batches (after warmup)."),
    batch_size: int = typer.Option(4, help="Batch size (matches training)."),
    workers: str = typer.Option("0,2,4", help="Comma list of num_workers to compare."),
) -> None:
    """Profile the train DataLoader: throughput vs num_workers + a collate cProfile."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Loading processor %s (offline cache) ...", MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # 1) wall-clock throughput across num_workers
    rows = []
    for nw in [int(x) for x in workers.split(",")]:
        loader = _make_train_loader(processor, batch_size, nw)
        secs, nb = _time_loader(loader, n_batches)
        sps = nb * batch_size / secs if secs else 0.0
        mpb = secs / nb * 1000 if nb else 0.0
        rows.append(
            {
                "num_workers": nw,
                "batches": nb,
                "seconds": round(secs, 3),
                "samples_per_s": round(sps, 1),
                "ms_per_batch": round(mpb, 1),
            }
        )
        log.info(
            "num_workers=%d: %d batches in %.2fs -> %.1f samples/s (%.1f ms/batch)",
            nw,
            nb,
            secs,
            sps,
            mpb,
        )

    # 2) cProfile the single-process collate path (num_workers=0 so the collate
    #    runs in-process and cProfile can actually see it).
    loader0 = _make_train_loader(processor, batch_size, 0)
    it = iter(loader0)
    next(it)  # warmup
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n_batches):
        try:
            next(it)
        except StopIteration:
            break
    pr.disable()
    pr.dump_stats(str(OUT_DIR / "dataloader.pstats"))

    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf)
    buf.write("=== top 25 by cumulative time ===\n")
    st.sort_stats("cumulative").print_stats(25)
    buf.write("\n=== top 20 by total (self) time ===\n")
    st.sort_stats("tottime").print_stats(20)
    (OUT_DIR / "dataloader_profile.txt").write_text(buf.getvalue())

    summary = {"n_batches": n_batches, "batch_size": batch_size, "throughput": rows}
    (OUT_DIR / "dataloader_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(
        "Wrote %s, %s, %s",
        OUT_DIR / "dataloader.pstats",
        OUT_DIR / "dataloader_profile.txt",
        OUT_DIR / "dataloader_summary.json",
    )
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
