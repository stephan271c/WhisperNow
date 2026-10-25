"""Full fine-tuning with TRL's SFT, loss on the reply only."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Gemma4ForCausalLM, TrainerCallback
from trl import SFTConfig, SFTTrainer

from ..datagen.paths import BUILD, TRAINING_ROOT
from . import prompt
from .config import TrainConfig

# Validation loss is measured on a fixed slice of the val split to keep evaluation cheap,
# twice per epoch, so every epoch boundary has its own value.
VAL_ROWS = 600
EVALS_PER_EPOCH = 2


def load_rows(files: list[Path], limit: int | None = None) -> Dataset:
    # Only the two columns: `datasets` takes a JSON file's schema from its first rows, and later rounds'
    # rows carry provenance fields round 1's rows lack.
    pairs = []
    for path in files:
        with path.open(encoding="utf-8") as source:
            pairs += [{"input": row["input"], "output": row["output"]} for row in map(json.loads, filter(str.strip, source))]
    rows = Dataset.from_list(pairs)
    if limit is not None:
        rows = rows.shuffle(seed=109).select(range(min(limit, len(rows))))
    return rows.map(
        lambda row: prompt.prompt_completion(row["input"], row["output"]),
        remove_columns=rows.column_names,
    )


def load_split(split: str, limit: int | None = None, build: Path = BUILD) -> Dataset:
    return load_rows([build / f"{split}.jsonl"], limit)


def resolve(path_or_id: str) -> str:
    """A model path under `training/`, else a Hub id as given."""
    local = TRAINING_ROOT / path_or_id
    return str(local) if local.exists() else path_or_id


def load_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    chat = prompt.chat_format(AutoConfig.from_pretrained(base_model).model_type)
    tokenizer.chat_template = chat.template()
    # Qwen3.5 has no bos token. llama.cpp then takes token 11 (",") as the bos, and the template would
    # render it when serving, so prompts start with the base model's end of document instead.
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    # Replies end at the end of the turn, whatever the base model's own eos is.
    tokenizer.eos_token = chat.end_of_turn
    return tokenizer


def load_model(path: str, dtype: torch.dtype):
    # A Gemma 4 checkpoint holds audio and vision towers besides the text model, which is under
    # `model.language_model.`; load only the text model. transformers saves it back under that prefix,
    # so its own saves need the mapping too. Without it no text weight matches, and transformers
    # initializes them all at random with only a load report.
    if AutoConfig.from_pretrained(path).model_type in ("gemma4", "gemma4_text"):
        return Gemma4ForCausalLM.from_pretrained(
            path, dtype=dtype, attn_implementation="sdpa", key_mapping={r"^model\.language_model\.": "model."}
        )
    return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa")


def checkpoint_dir(out: Path, epoch: int) -> Path:
    return out / f"epoch-{epoch}"


def _bf16_state_dict(model) -> dict[str, torch.Tensor]:
    """The parameters as bf16 copies on the CPU; the model is untouched.

    Buffers keep their dtype: LFM2-MoE's routing bias is fp32 in the base checkpoint, and rounding it
    to bf16 changes which experts some tokens go to.
    """
    buffers = {name for name, _ in model.named_buffers()}
    return {name: tensor.detach().to("cpu", tensor.dtype if name in buffers else torch.bfloat16)
            for name, tensor in model.state_dict().items()}


class EpochCheckpoints(TrainerCallback):
    """Saves a loadable bf16 model after every epoch but the last, whose model is `model/`.

    A checkpoint mid-schedule was taken before the learning rate decayed, so it scores somewhat
    below a run that ends at that epoch; it shows the trend, not the best a shorter run can do.
    """

    def __init__(self, out: Path, epochs: float, tokenizer) -> None:
        self.out, self.epochs, self.tokenizer = out, epochs, tokenizer
        self.at: dict[int, int] = {}

    def on_train_begin(self, args, state, control, **kwargs):
        self.at = {round(state.max_steps * epoch / self.epochs): epoch for epoch in range(1, math.ceil(self.epochs))}

    def on_step_end(self, args, state, control, model=None, **kwargs):
        epoch = self.at.get(state.global_step)
        if epoch is None or model is None:
            return
        target = checkpoint_dir(self.out, epoch)
        model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        model.save_pretrained(target, state_dict=_bf16_state_dict(model))
        self.tokenizer.save_pretrained(target)


class StopOnNonFiniteGradients(TrainerCallback):
    """Stops at the first logged non-finite gradient norm: clipping turns it into NaN weights for the rest of the run.

    fla 0.5.2's gated delta rule backward gave NaN gradients on a B200 (Triton 3.8.0) for some batches under 128
    tokens, and the weights never recovered.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "grad_norm" in logs and not math.isfinite(logs["grad_norm"]):
            control.should_training_stop = True


class Pause(TrainerCallback):
    """Asks for a resumable checkpoint at `step`, then stops."""

    def __init__(self, step: int) -> None:
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            control.should_save = True
            control.should_training_stop = True


class TimedCheckpoints(TrainerCallback):
    """Asks for a resumable checkpoint every `minutes` of training, but not at the last step."""

    def __init__(self, minutes: float) -> None:
        self.seconds = minutes * 60
        self.last = time.monotonic()

    def on_train_begin(self, args, state, control, **kwargs):
        self.last = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() - self.last >= self.seconds and state.global_step < state.max_steps:
            control.should_save = True

    def on_save(self, args, state, control, **kwargs):
        self.last = time.monotonic()


def latest_checkpoint(out: Path) -> str | None:
    """The newest complete checkpoint in `out/checkpoints`.

    The trainer writes `trainer_state.json` after the weights, optimizer, scheduler and RNG state, and
    deletes the previous checkpoint only after that, so a save cut short leaves an older complete one.
    """
    complete = [path for path in (out / "checkpoints").glob("checkpoint-*") if (path / "trainer_state.json").exists()]
    return str(max(complete, key=lambda path: int(path.name.split("-")[1]))) if complete else None


def train(config: TrainConfig, out: Path, callbacks: list[TrainerCallback] = ()) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    base_model = resolve(config.base_model)
    tokenizer = load_tokenizer(base_model)
    resumed_from = latest_checkpoint(out)
    # fp32 master weights under bf16 autocast. A resumed run loads its weights here: the trainer
    # reloads them with a non-strict `load_state_dict`, which matches no key when transformers saves
    # another layout (Qwen3.5's `model.language_model.*`) and silently keeps the base weights.
    model = load_model(resumed_from or base_model, torch.float32)
    if config.freeze_per_layer_embeddings:
        # Frozen weights need no fp32 master copy; the layers still run under bf16 autocast.
        embeddings = model.get_per_layer_input_embeddings()
        embeddings.requires_grad_(False)
        embeddings.to(torch.bfloat16)

    train_rows = load_rows([TRAINING_ROOT / config.build / "train.jsonl"], config.train_rows or None)
    eval_rows = load_split("val", VAL_ROWS, TRAINING_ROOT / config.build)
    common = dict(
        output_dir=str(out / "checkpoints"),
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps or -1,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.grad_accum,
        warmup_steps=config.warmup,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler,
        max_length=config.max_length,
        bf16=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=25,
        eval_strategy="steps",
        # A short probe evaluates once, at its end; a ratio below 1 is a share of the total steps.
        eval_steps=config.max_steps or 1 / (EVALS_PER_EPOCH * math.ceil(config.epochs)),
        save_strategy="no",
        save_total_limit=1,
        report_to="none",
        seed=config.seed,
        dataloader_num_workers=2,
    )
    callbacks = [*callbacks, StopOnNonFiniteGradients(),
                 *([] if config.max_steps else [EpochCheckpoints(out, config.epochs, tokenizer)])]
    if config.checkpoint_minutes:
        callbacks.append(TimedCheckpoints(config.checkpoint_minutes))
    if config.pause_at_step:
        callbacks.append(Pause(config.pause_at_step))
    args = SFTConfig(
        **common,
        completion_only_loss=True,
        train_sampling_strategy="group_by_length",
        loss_type="chunked_nll" if config.chunked_loss else "nll",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_rows,
        eval_dataset=eval_rows,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer.train(resume_from_checkpoint=resumed_from)
    seconds = time.monotonic() - started
    if config.pause_at_step and trainer.state.global_step == config.pause_at_step < trainer.state.max_steps:
        return {"paused_at": trainer.state.global_step, "resumed_from": resumed_from, "seconds": round(seconds),
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    final_eval = trainer.evaluate()
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    trained = trainer.model
    # Buffers keep their dtype, as in `_bf16_state_dict`.
    for parameter in trained.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    trained.generation_config.eos_token_id = tokenizer.eos_token_id
    trained.generation_config.pad_token_id = tokenizer.pad_token_id
    trained.save_pretrained(out / "model")
    tokenizer.save_pretrained(out / "model")
    summary = {
        "config": config.to_dict(),
        "train_files": [str(TRAINING_ROOT / config.build / "train.jsonl")],
        "train_rows": len(train_rows),
        "steps": trainer.state.global_step,
        # A resumed run's seconds and peak memory cover only the part since `resumed_from`.
        "resumed_from": resumed_from,
        "seconds": round(seconds),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "trainable_parameters": trainable,
        "final_eval_loss": final_eval["eval_loss"],
        "log_history": trainer.state.log_history,
    }
    (out / "train.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
