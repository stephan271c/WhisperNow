"""Training configs: TOML files in `configs/`, with `--set key=value` overrides for sweeps."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal

Method = Literal["full", "lora"]
Objective = Literal["sft", "dpo"]


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
    # Recompute activations in the backward pass; TRL's default.
    gradient_checkpointing: bool = True
    # Memory savers for models that do not fit a 12 GB GPU otherwise: a transformers optimizer
    # name (e.g. "paged_adamw_8bit" from bitsandbytes), keeping the input embeddings frozen in
    # bf16, and Liger's fused linear cross-entropy, which never materializes the logits.
    optim: str = "adamw_torch_fused"
    freeze_embeddings: bool = False
    liger: bool = False
    # Fresh speech noise per epoch: train one pass over `epochs` noisy variants of the
    # train split (`datagen epochs`), so each row is seen once per epoch, each time with new noise.
    noise_per_epoch: bool = False
    # Train on this JSONL file (relative to `training/`) instead of the build's train split: the
    # judged samples of rejection-sampling SFT, or the preference pairs of DPO.
    train_file: str = ""
    # `sft` trains on references; `dpo` on preference pairs (`chosen`, `rejected`) with TRL's
    # DPOTrainer, sigmoid loss plus `sft_weight` × the NLL of the chosen reply (RPO), against
    # reference log-probabilities computed once before training.
    objective: Objective = "sft"
    # Fact-weighted SFT: the loss on reply tokens inside fact spans (`model/facts.py`)
    # weighs this much against 1 for the rest, rescaled to average 1 per micro-batch. 0 is plain SFT.
    fact_weight: float = 0.0
    beta: float = 0.1
    sft_weight: float = 1.0
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
    if config.objective not in ("sft", "dpo"):
        raise SystemExit(f"objective must be sft or dpo, not {config.objective!r}")
    if config.noise_per_epoch and (config.epochs != int(config.epochs) or config.train_file):
        raise SystemExit("noise_per_epoch needs a whole number of epochs over the build's train split")
    if config.objective == "dpo" and not config.train_file:
        raise SystemExit("objective dpo needs a train_file of preference pairs")
    return config
