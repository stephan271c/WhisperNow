"""Fact-weighted SFT (todo 114): extra loss weight on the reply tokens that carry facts.

The plain SFT loss averages over every reply token, and most tokens of a long reply are easy
copying, so leaving out one fact costs little. Here the tokens inside `facts.fact_spans` weigh
`fact_weight` and the rest 1. The weights are rescaled to average 1 over each micro-batch's reply
tokens, so the loss keeps the plain loss's scale and the learning rate stays comparable.

Training rows are tokenized here, as TRL tokenizes a conversational prompt-completion row, with a
`loss_weights` column beside `labels`. The collator carries the weights through and the trainer
pops them before the forward pass. Evaluation rows have no weights, so the val loss is the plain
one, comparable with SFT runs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.checkpoint import checkpoint
from trl import SFTTrainer

from . import facts

# Positions per chunk of the output head and cross-entropy. The full logits of a long batch, and
# their gradient, are never made: at 8 × 2,816 tokens and a 65k vocabulary they take about 6 GB
# each in fp32. Each chunk is checkpointed, so only the hidden states are kept for the backward pass.
_CHUNK = 512


def weights_for(tokenizer, prompt: list[dict], completion: list[dict], fact_weight: float, max_length: int) -> dict:
    """`input_ids`, `labels` (-100 on the prompt) and `loss_weights` for one row."""
    prompt_text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    full_text = tokenizer.apply_chat_template(prompt + completion, tokenize=False)
    if not full_text.startswith(prompt_text):
        raise ValueError("the rendered prompt is not a prefix of the rendered row")
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("the prompt tokenizes differently inside the row")
    reply = completion[0]["content"]
    reply_start = full_text.index(reply, len(prompt_text))
    mask = facts.char_mask(reply)
    labels, weights = [], []
    for index, (token, (start, end)) in enumerate(zip(ids, offsets, strict=True)):
        if index < len(prompt_ids):
            labels.append(-100)
            weights.append(0.0)
            continue
        labels.append(token)
        inside = range(max(start, reply_start) - reply_start, min(end, reply_start + len(reply)) - reply_start)
        weights.append(fact_weight if any(mask[i] for i in inside) else 1.0)
    return {"input_ids": ids[:max_length], "labels": labels[:max_length], "loss_weights": weights[:max_length]}


def tokenize(rows: Dataset, tokenizer, fact_weight: float, max_length: int) -> Dataset:
    """Prompt-completion rows (`load_rows`) as token ids, labels and loss weights."""
    return rows.map(
        lambda row: weights_for(tokenizer, row["prompt"], row["completion"], fact_weight, max_length),
        remove_columns=rows.column_names,
        desc="Tokenizing with fact weights",
    )


class WeightedCollator:
    """TRL's collator, with each row's `loss_weights` padded with zeros to the batch's labels."""

    def __init__(self, inner) -> None:
        self.inner = inner

    def __call__(self, examples: list[dict]) -> dict:
        weights = [example.pop("loss_weights") for example in examples if "loss_weights" in example]
        batch = self.inner(examples)
        if weights:
            if len(weights) != len(examples):
                raise ValueError("some rows of the batch have no loss weights")
            padded = torch.zeros(batch["labels"].shape, dtype=torch.float32)
            for row, row_weights in enumerate(weights):
                padded[row, : len(row_weights)] = torch.tensor(row_weights, dtype=torch.float32)
            batch["loss_weights"] = padded
        return batch


def weighted_loss(
    hidden: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, head=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of weight × next-token cross-entropy over labelled positions, and the count of those positions.

    `hidden` is the last hidden states and `head` the output layer, applied one chunk at a time; without
    a head, `hidden` is the logits. The weights are rescaled to average 1 over the labelled positions, so
    the sum divided by the count is the plain mean when every weight is 1.
    """
    labels = labels[:, 1:]
    weights = weights[:, 1:].to(torch.float32)
    valid = labels != -100
    count = valid.sum()
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    scale = count / weights.sum().clamp_min(1e-8)
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, labels.shape[1], _CHUNK):
        chunk = slice(start, start + _CHUNK)
        total = total + checkpoint(
            _chunk_loss, hidden[:, :-1][:, chunk], labels[:, chunk], weights[:, chunk], head, use_reentrant=False
        )
    return total * scale, count


def _chunk_loss(hidden: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, head) -> torch.Tensor:
    logits = (hidden if head is None else head(hidden)).float()
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="none")
    return (losses * weights.reshape(-1)).sum()


class WeightedSFTTrainer(SFTTrainer):
    """SFT whose training loss weights each reply token by its row's `loss_weights`."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weights = inputs.pop("loss_weights", None)
        if weights is None or not model.training:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        labels = inputs.pop("labels")
        # The decoder alone: its last hidden states go through the output head chunk by chunk.
        outputs = getattr(model, model.base_model_prefix)(**inputs, use_cache=False)
        total, count = weighted_loss(outputs.last_hidden_state, labels, weights, model.get_output_embeddings())
        # As the Trainer expects: summed over the accumulated batch's tokens when the model takes
        # loss kwargs (it then skips dividing by the accumulation steps), else this micro-batch's mean.
        if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            loss = total / num_items_in_batch
        else:
            loss = total / count.clamp_min(1)
        return (loss, outputs) if return_outputs else loss
