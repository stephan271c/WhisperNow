import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")

import torch.nn.functional as F  # noqa: E402

from whispernow_training.model import weighted  # noqa: E402


def test_unit_weights_give_the_plain_mean():
    torch.manual_seed(0)
    logits = torch.randn(2, 700, 30)
    labels = torch.randint(0, 30, (2, 700))
    labels[:, :50] = -100
    labels[1, 600:] = -100
    total, count = weighted.weighted_loss(logits, labels, torch.ones(2, 700))
    plain = F.cross_entropy(logits[:, :-1].reshape(-1, 30), labels[:, 1:].reshape(-1), ignore_index=-100)
    assert torch.allclose(total / count, plain)
    assert count == (labels[:, 1:] != -100).sum()


def test_fact_tokens_weigh_more_and_the_weights_average_one():
    torch.manual_seed(1)
    logits = torch.randn(1, 20, 10)
    labels = torch.randint(0, 10, (1, 20))
    labels[:, :5] = -100
    weights = torch.ones(1, 20)
    weights[0, 10] = 4.0
    losses = F.cross_entropy(logits[0, :-1], labels[0, 1:], ignore_index=-100, reduction="none")
    total, count = weighted.weighted_loss(logits, labels, weights)
    shifted = weights[0, 1:].clone()
    shifted[labels[0, 1:] == -100] = 0
    expected = (losses * shifted).sum() * count / shifted.sum()
    assert torch.allclose(total, expected)
    # Padding and prompt positions carry no weight, whatever the collator put there.
    padded = weights.clone()
    padded[0, :5] = 9.0
    assert torch.allclose(weighted.weighted_loss(logits, labels, padded)[0], total)


class _Inner:
    def __call__(self, examples):
        width = max(len(e["input_ids"]) for e in examples)
        labels = torch.full((len(examples), width), -100)
        for row, example in enumerate(examples):
            labels[row, : len(example["labels"])] = torch.tensor(example["labels"])
        return {"labels": labels}


def test_collator_pads_weights_to_the_labels():
    collator = weighted.WeightedCollator(_Inner())
    batch = collator([
        {"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "loss_weights": [0.0, 1.0, 3.0]},
        {"input_ids": [4, 5], "labels": [-100, 5], "loss_weights": [0.0, 1.0]},
    ])
    assert batch["loss_weights"].tolist() == [[0.0, 1.0, 3.0], [0.0, 1.0, 0.0]]
    # Evaluation rows have no weights and pass through untouched.
    assert "loss_weights" not in collator([{"input_ids": [1], "labels": [1]}])


class _Tokenizer:
    """A character tokenizer with a two-message chat template, enough for `weights_for`."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        text = "".join(f"<{m['role']}>{m['content']}</>" for m in messages)
        return text + "<assistant>" if add_generation_prompt else text

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        out = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


def test_weights_mark_fact_characters_of_the_reply_only():
    prompt = [{"role": "user", "content": "say it"}]
    completion = [{"role": "assistant", "content": "Pay Dana $40."}]
    row = weighted.weights_for(_Tokenizer(), prompt, completion, 3.0, 1000)
    text = "".join(chr(i) for i in row["input_ids"])
    reply_at = text.index("Pay Dana")
    assert all(label == -100 for label in row["labels"][:reply_at])
    weights = dict(zip(text[reply_at:reply_at + 13], row["loss_weights"][reply_at:reply_at + 13], strict=True))
    assert weights["D"] == 3.0 and weights["$"] == 3.0 and weights["4"] == 3.0
    assert weights["y"] == 3.0  # "Pay" opens the sentence
    assert row["loss_weights"][text.index("</>", reply_at)] == 1.0
