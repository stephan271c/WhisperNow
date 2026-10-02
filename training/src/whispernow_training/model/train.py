"""Supervised fine-tuning with TRL: full fine-tune or LoRA, loss on the reply only."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from ..datagen.paths import BUILD
from . import prompt
from .config import TrainConfig

# Validation loss is measured on a fixed slice of the val split to keep evaluation cheap.
VAL_ROWS = 600
EVALS_PER_RUN = 8


def load_split(split: str, limit: int | None = None) -> Dataset:
    rows = load_dataset("json", data_files=str(BUILD / f"{split}.jsonl"), split="train")
    if limit is not None:
        rows = rows.shuffle(seed=109).select(range(min(limit, len(rows))))
    return rows.map(
        lambda row: prompt.prompt_completion(row["input"], row["output"]),
        remove_columns=rows.column_names,
    )


def load_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.chat_template = prompt.chat_template()
    # Replies end at the end of the turn; MiniCPM's own eos is `</s>`.
    tokenizer.eos_token = prompt.END_OF_TURN
    return tokenizer


def checkpoint_dir(out: Path, epoch: int) -> Path:
    return out / f"epoch-{epoch}"


def _bf16_state_dict(model) -> dict[str, torch.Tensor]:
    """The weights as bf16 copies on the CPU, with any LoRA adapter added in; the model is untouched."""
    if not hasattr(model, "get_base_model"):
        return {name: tensor.detach().to("cpu", torch.bfloat16) for name, tensor in model.state_dict().items()}
    base = model.get_base_model()
    merged = {}
    for name, tensor in base.state_dict().items():
        if "lora_" in name:
            continue
        merged[name.replace(".base_layer.", ".")] = tensor.detach().to("cpu", torch.float32)
    for name, module in base.named_modules():
        if isinstance(module, LoraLayer):
            for adapter in module.active_adapters:
                merged[f"{name}.weight"] += module.get_delta_weight(adapter).detach().to("cpu", torch.float32)
    return {name: tensor.to(torch.bfloat16) for name, tensor in merged.items()}


class EpochCheckpoints(TrainerCallback):
    """Saves a loadable bf16 model after every epoch but the last, whose model is `model/`.

    A checkpoint mid-schedule was taken before the learning rate decayed, so it scores somewhat
    below a run that ends at that epoch; it shows the trend, not the best a shorter run can do.
    """

    def __init__(self, out: Path, epochs: float, tokenizer) -> None:
        self.out, self.epochs, self.tokenizer = out, epochs, tokenizer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = round(state.epoch)
        if epoch >= self.epochs or model is None:
            return
        target = checkpoint_dir(self.out, epoch)
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        base.generation_config.eos_token_id = self.tokenizer.eos_token_id
        base.generation_config.pad_token_id = self.tokenizer.pad_token_id
        base.save_pretrained(target, state_dict=_bf16_state_dict(model))
        self.tokenizer.save_pretrained(target)


def train(config: TrainConfig, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    tokenizer = load_tokenizer(config.base_model)
    # Full fine-tunes keep fp32 master weights under bf16 autocast; LoRA trains fp32 adapters on bf16 weights.
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        dtype=torch.float32 if config.method == "full" else torch.bfloat16,
        attn_implementation="sdpa",
    )
    if config.freeze_embeddings:
        # Frozen weights need no fp32 master copy; the layers still run under bf16 autocast.
        embeddings = model.get_input_embeddings()
        embeddings.requires_grad_(False)
        embeddings.to(torch.bfloat16)
    peft_config = None
    if config.method == "lora":
        peft_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            task_type="CAUSAL_LM",
        )

    train_rows = load_split("train")
    steps_per_epoch = len(train_rows) // (config.batch_size * config.grad_accum)
    total_steps = config.max_steps or int(steps_per_epoch * config.epochs)
    args = SFTConfig(
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
        completion_only_loss=True,
        train_sampling_strategy="group_by_length",
        bf16=True,
        optim=config.optim,
        gradient_checkpointing=config.gradient_checkpointing,
        use_liger_kernel=config.liger,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=max(1, total_steps // EVALS_PER_RUN),
        save_strategy="no",
        report_to="none",
        seed=config.seed,
        dataloader_num_workers=2,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_rows,
        eval_dataset=load_split("val", VAL_ROWS),
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[EpochCheckpoints(out, config.epochs, tokenizer)],
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer.train()
    seconds = time.monotonic() - started
    final_eval = trainer.evaluate()
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    trained = trainer.model.merge_and_unload() if config.method == "lora" else trainer.model
    trained = trained.to(torch.bfloat16)
    trained.generation_config.eos_token_id = tokenizer.eos_token_id
    trained.generation_config.pad_token_id = tokenizer.pad_token_id
    trained.save_pretrained(out / "model")
    tokenizer.save_pretrained(out / "model")
    summary = {
        "config": config.to_dict(),
        "train_rows": len(train_rows),
        "steps": trainer.state.global_step,
        "seconds": round(seconds),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "trainable_parameters": trainable,
        "final_eval_loss": final_eval["eval_loss"],
        "log_history": trainer.state.log_history,
    }
    (out / "train.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
