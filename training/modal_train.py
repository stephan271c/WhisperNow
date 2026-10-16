"""Fine-tunes the cleanup model on a Modal GPU with `whispernow_training.model.train`.

The container mirrors `training/` with only what training reads: the package, `serving/` (the
chat template), `configs/`, `prompts/` and the build's train and val splits. Hugging Face
downloads and run outputs live on Volumes. Modal's GPUs can be preempted, so a config with
`checkpoint_minutes` saves a resumable checkpoint to the runs Volume, and the restarted function
resumes from it.

    modal run -e main modal_train.py::prefetch --config configs/lfm25-2.6b.toml
    modal run -e main modal_train.py --config configs/lfm25-2.6b.toml --gpu H100 --probe-steps 50
    modal run --detach -e main modal_train.py --config configs/lfm25-2.6b.toml --name NAME --gpu H100
    modal volume get -e main whispernow-runs NAME runs/
"""

from __future__ import annotations

import statistics
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
REMOTE = Path("/root/training")
HF_CACHE = Path("/hf")
RUNS = REMOTE / "runs"
ROWS_PER_STEP = 32

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(HERE), extras=["train"])
    # Without expandable segments, a 32-row micro-batch fragments the H100's memory (64 GB allocated,
    # 84 GB reserved) until the eval briefly runs out.
    .env({"HF_HOME": str(HF_CACHE), "HF_XET_HIGH_PERFORMANCE": "1", "PYTHONPATH": str(REMOTE / "src"),
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(HERE / "src", str(REMOTE / "src"), ignore=["**/__pycache__"])
    .add_local_dir(HERE / "serving", str(REMOTE / "serving"))
    .add_local_dir(HERE / "configs", str(REMOTE / "configs"))
    .add_local_dir(HERE / "prompts", str(REMOTE / "prompts"))
    .add_local_file(HERE / "data/build/train.jsonl", str(REMOTE / "data/build/train.jsonl"))
    .add_local_file(HERE / "data/build/val.jsonl", str(REMOTE / "data/build/val.jsonl"))
)
hf_cache = modal.Volume.from_name("whispernow-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("whispernow-runs", create_if_missing=True)

app = modal.App("whispernow-train", image=image)
CONTAINER = dict(volumes={str(HF_CACHE): hf_cache, str(RUNS): runs}, cpu=4.0, memory=32 * 1024)


def _callbacks(timings: list | None = None, save_at: int = 0) -> list:
    from transformers import TrainerCallback

    class CommitOnSave(TrainerCallback):
        """Commits the runs Volume after each checkpoint, so a restarted container sees it.

        A 31 GB checkpoint took 266 s to commit on an H100 (and 98 s to write), so the commit runs
        in the background while training goes on.
        """

        def __init__(self) -> None:
            self.thread: threading.Thread | None = None
            self.step_end = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            self.step_end = time.monotonic()

        def on_save(self, args, state, control, **kwargs):
            self.wait()
            written = time.monotonic() - self.step_end
            self.thread = threading.Thread(target=self.commit, args=(state.global_step, written))
            self.thread.start()

        def on_train_end(self, args, state, control, **kwargs):
            self.wait()

        def commit(self, step: int, written: float) -> None:
            started = time.monotonic()
            runs.commit()
            print(f"step {step}: checkpoint written in {written:.0f} s, committed in {time.monotonic() - started:.0f} s",
                  flush=True)

        def wait(self) -> None:
            if self.thread is not None:
                self.thread.join()
                self.thread = None

    class StepTimes(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            timings.append((state.global_step, time.monotonic()))

    class SaveAt(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == save_at:
                control.should_save = True

    return [CommitOnSave(), *([StepTimes()] if timings is not None else []), *([SaveAt()] if save_at else [])]


@app.function(timeout=1800, volumes={str(HF_CACHE): hf_cache}, cpu=2.0, memory=8 * 1024)
def prefetch(config: str) -> str:
    """Checks the image on a CPU and downloads the base model into the cache Volume, so no GPU time goes to either."""
    import torch
    from huggingface_hub import snapshot_download

    from whispernow_training.model import config as train_config
    from whispernow_training.model import train  # noqa: F401  (imports every training dependency)

    settings = train_config.load(REMOTE / config)
    path = snapshot_download(settings.base_model)
    hf_cache.commit()
    shipped = sorted(str(file.relative_to(REMOTE)) for file in REMOTE.rglob("*") if file.is_file() and "src/" not in str(file))
    return f"torch {torch.__version__} (CUDA {torch.version.cuda}); {settings.base_model} in {path}; shipped besides src/: {shipped}"


@app.function(gpu="H100", timeout=8 * 3600, retries=modal.Retries(max_retries=2, initial_delay=0.0),
              single_use_containers=True, **CONTAINER)
def train_run(config: str, name: str, overrides: list[str]) -> dict:
    from whispernow_training.model import config as train_config
    from whispernow_training.model import train

    settings = train_config.load(REMOTE / config, overrides)
    summary = train.train(settings, RUNS / name, callbacks=_callbacks())
    runs.commit()
    return {key: value for key, value in summary.items() if key != "log_history"}


@app.function(gpu="H100", timeout=3600, **CONTAINER)
def probe(config: str, batch_sizes: list[int], steps: int, save: bool) -> list[dict]:
    """Times `steps` optimizer steps per micro-batch size, smallest first, until one runs out of memory.

    The length-grouped sampler puts the longest row in the first batch, so a short run meets the
    peak memory. With `save`, the first size also writes one checkpoint to the runs Volume, to time it.
    """
    import gc
    import shutil
    import subprocess

    import torch

    from whispernow_training.model import config as train_config
    from whispernow_training.model import train

    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                         capture_output=True, text=True).stdout, flush=True)
    results = []
    for index, batch_size in enumerate(batch_sizes):
        overrides = [f"batch_size={batch_size}", f"grad_accum={ROWS_PER_STEP // batch_size}", f"max_steps={steps}",
                     "checkpoint_minutes=0"]
        settings = train_config.load(REMOTE / config, overrides)
        saving = save and index == 0
        out = (RUNS if saving else Path("/tmp")) / f"probe-bs{batch_size}"
        timings: list = []
        torch.cuda.reset_peak_memory_stats()
        try:
            summary = train.train(settings, out, callbacks=_callbacks(timings, steps if saving else 0))
        except torch.OutOfMemoryError as error:
            results.append({"batch_size": batch_size, "oom": str(error).splitlines()[0]})
            print(results[-1], flush=True)
            break
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        at = dict(timings)
        gaps = [at[step] - at[step - 1] for step in range(6, steps + 1)]
        logs = {entry["step"]: entry for entry in summary["log_history"] if "num_tokens" in entry}
        rate = None
        if 25 in logs and steps in logs:
            rate = (logs[steps]["num_tokens"] - logs[25]["num_tokens"]) / (at[steps] - at[25])
        results.append({
            "batch_size": batch_size,
            "seconds": summary["seconds"],
            "step_seconds_median": round(statistics.median(gaps), 3),
            "step_seconds_mean": round(statistics.mean(gaps), 3),
            "tokens_per_second_steps_26_on": rate and round(rate),
            "peak_allocated_gb": summary["peak_gpu_gb"],
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
            "final_eval_loss": summary["final_eval_loss"],
            "log": [{key: entry.get(key) for key in ("step", "loss", "num_tokens", "grad_norm")} for entry in logs.values()],
        })
        print(results[-1], flush=True)
        shutil.rmtree(out, ignore_errors=True)
        if saving:
            runs.commit()
    return results


@app.local_entrypoint()
def main(config: str, name: str = "", gpu: str = "H100", overrides: str = "", probe_steps: int = 0,
         batch_sizes: str = "8,16,32", save: bool = True):
    """Trains `config` as runs/NAME, or with `--probe-steps N` times N steps per micro-batch size.

    `--overrides` is a comma-separated list of `key=value`, as `cleanup train --set` takes them.
    """
    if probe_steps:
        sizes = [int(size) for size in batch_sizes.split(",")]
        for result in probe.with_options(gpu=gpu).remote(config, sizes, probe_steps, save):
            print(result)
        return
    if not name:
        raise SystemExit("--name is required for a training run")
    changes = [item for item in overrides.split(",") if item]
    call = train_run.with_options(gpu=gpu).spawn(config, name, changes)
    print(f"spawned {call.object_id}; follow it with `modal app logs -e main whispernow-train`")
