"""Training configs: TOML files in `configs/`, with `--set key=value` overrides for sweeps."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal

Method = Literal["full", "lora"]


@dataclass(frozen=True)
class LoraSettings:
    r: int = 64
    alpha: int = 64
    dropout: float = 0.0
    target_modules: str | list[str] = "all-linear"


@dataclass(frozen=True)
class TrainConfig:
    base_model: str
    method: Method = "full"
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
    # Recompute activations in the backward pass; TRL's default.
    gradient_checkpointing: bool = True
    # Memory savers for models that do not fit a 12 GB GPU otherwise: a transformers optimizer
    # name (e.g. "paged_adamw_8bit" from bitsandbytes), keeping the input embeddings frozen in
    # bf16, and Liger's fused linear cross-entropy, which never materializes the logits.
    optim: str = "adamw_torch_fused"
    freeze_embeddings: bool = False
    liger: bool = False
    # TRL's chunked cross-entropy (`loss_type="chunked_nll"`): skips the output layer on prompt and
    # padding tokens and never builds the whole vocabulary's logits. For architectures Liger has no
    # kernels for, such as LFM2, where `liger` silently does nothing.
    chunked_loss: bool = False
    # Save a resumable checkpoint (weights, optimizer, scheduler, data position) this often, keeping
    # only the latest, and resume from it when the run starts again; 0 saves none. For preemptible
    # cloud GPUs (modal_train.py).
    checkpoint_minutes: float = 0
    # Save a resumable checkpoint after this many optimizer steps of the full schedule and stop; the same
    # config without it resumes and finishes the run, e.g. in another Modal workspace. 0 never pauses.
    pause_at_step: int = 0
    # Fresh speech noise per epoch: train one pass over `epochs` noisy variants of the
    # train split (`datagen epochs`), so each row is seen once per epoch, each time with new noise.
    noise_per_epoch: bool = False
    lora: LoraSettings = field(default_factory=LoraSettings)

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
        section, _, name = key.partition(".")
        if name:
            raw.setdefault(section, {})[name] = _coerce(value)
        else:
            raw[key] = _coerce(value)
    known = {f.name for f in fields(TrainConfig)}
    unknown = set(raw) - known
    if unknown:
        raise SystemExit(f"unknown config keys {sorted(unknown)} in {path}")
    lora = LoraSettings(**raw.pop("lora", {}))
    config = TrainConfig(**raw, lora=lora)
    if config.method not in ("full", "lora"):
        raise SystemExit(f"method must be full or lora, not {config.method!r}")
    if config.noise_per_epoch and config.epochs != int(config.epochs):
        raise SystemExit("noise_per_epoch needs a whole number of epochs over the build's train split")
    return config
