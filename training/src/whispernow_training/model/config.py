"""Training configs: TOML files in `configs/`, with `--set key=value` overrides."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    base_model: str
    learning_rate: float = 3e-5
    epochs: float = 2.0
    # Examples per optimizer step is batch_size × grad_accum.
    batch_size: int = 16
    grad_accum: int = 2
    # A float below 1 is a fraction of the total steps.
    warmup: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"
    max_length: int = 1280
    seed: int = 109
    # Stop after this many optimizer steps; 0 trains the full schedule.
    max_steps: int = 0
    # Train on a fixed random sample of this many rows; 0 uses them all.
    train_rows: int = 0
    # The dataset directory, under `training/`, whose train and val splits the run reads.
    build: str = "data/build"
    # Recompute activations in the backward pass; TRL's default.
    gradient_checkpointing: bool = True
    # TRL's chunked cross-entropy (`loss_type="chunked_nll"`): skips the output layer on prompt and
    # padding tokens and never builds the whole vocabulary's logits.
    chunked_loss: bool = False
    # Save a resumable checkpoint (weights, optimizer, scheduler, data position) this often, keeping
    # only the latest, and resume from it when the run starts again; 0 saves none. For preemptible
    # cloud GPUs (modal_train.py).
    checkpoint_minutes: float = 0
    # Save a resumable checkpoint after this many optimizer steps of the full schedule and stop; the same
    # config without it resumes and finishes the run, e.g. in another Modal workspace. 0 never pauses.
    pause_at_step: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _coerce(value: str):
    """Parses an override value as a TOML value, falling back to a bare string."""
    try:
        return tomllib.loads(f"v = {value}")["v"]
    except tomllib.TOMLDecodeError:
        return value


def load(path: Path, overrides: list[str] = ()) -> TrainConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    for override in overrides:
        key, _, value = override.partition("=")
        raw[key] = _coerce(value)
    known = {f.name for f in fields(TrainConfig)}
    unknown = set(raw) - known
    if unknown:
        raise SystemExit(f"unknown config keys {sorted(unknown)} in {path}")
    return TrainConfig(**raw)
