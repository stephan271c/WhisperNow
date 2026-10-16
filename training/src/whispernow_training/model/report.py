"""Markdown tables from the files runs write: `train.json` and `eval*/<split>.scores.json`."""

from __future__ import annotations

import json

from ..datagen.paths import TRAINING_ROOT

RUNS = TRAINING_ROOT / "runs"

# Column → rate name in scores.json. Judge columns are blank until the run is judged.
QUALITY_COLUMNS = {
    "chrF": "chrf",
    "exact %": "exact",
    "layout ok %": "layout_match",
    "fmt added %": "formatting_added",
    "fmt missed %": "formatting_missed",
    "protected kept %": "protected_kept",
    "filler left %": "leftover_filler",
    "deleted >30% %": "deleted_over_30pct",
    "loops %": "loop",
    "answered %": "answered",
    "dropped %": "facts_dropped",
    "invented %": "facts_invented",
    "out/ref words": "length_ratio",
}


# Column → (fault kind, measure) in scores.json's `faults`, shown when the judge counted faults.
FAULT_COLUMNS = {
    "lost /row": ("facts_lost", "per_row"),
    "lost /1k words": ("facts_lost", "per_1000_words"),
    "dropped /row": ("dropped", "per_row"),
    "changed /row": ("changed", "per_row"),
    "wrong side /row": ("wrong_side", "per_row"),
    "unresolved %": ("unresolved", "rows"),
    "invented /row": ("invented", "per_row"),
}


def fault_cells(summary: dict) -> list[str]:
    faults = summary.get("faults", {})
    cells = []
    for kind, measure in FAULT_COLUMNS.values():
        value = faults.get(kind, {}).get(measure)
        cells.append("" if value is None else f"{value:.1f}" if measure == "rows" else f"{value:.2f}")
    return cells


def _rate(summary: dict, name: str) -> str:
    value = summary.get(name)
    if value is None:
        return ""
    if isinstance(value, dict):
        return f"{value['rate']:.1f}"
    return f"{value:.2f}" if name == "length_ratio" else f"{value:.1f}"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def quality_cells(summary: dict) -> list[str]:
    return [_rate(summary, name) for name in QUALITY_COLUMNS.values()]


def eval_loss_at(train: dict, epoch: float) -> float | None:
    """Val loss nearest the end of `epoch`, counted in steps: a run over per-epoch noise variants is
    one pass, so its trainer's own epoch count ends at 1."""
    epochs = train["config"]["epochs"]
    losses = [entry for entry in train["log_history"] if "eval_loss" in entry]
    if not losses:
        return None
    nearest = min(losses, key=lambda entry: abs(epochs * entry["step"] / train["steps"] - epoch))
    return nearest["eval_loss"]


def sweep_table(names: list[str], stem: str) -> str:
    header = ["run", "method", "start", "lr", "epochs", "min", "GPU GB", "val loss", *QUALITY_COLUMNS]
    lines = [_row(header), _row(["---"] * len(header))]
    for name in names:
        train_file, scores_file = RUNS / name / "train.json", RUNS / name / "eval" / f"{stem}.scores.json"
        if not train_file.exists():
            continue
        train = json.loads(train_file.read_text(encoding="utf-8"))
        config = train["config"]
        method = config["method"] if config["method"] == "full" else f"lora r{config['lora']['r']}"
        start = "base" if config["base_model"].endswith("-Base") else "instruct"
        overall = json.loads(scores_file.read_text(encoding="utf-8"))["overall"] if scores_file.exists() else {}
        lines.append(
            _row(
                [
                    name,
                    method,
                    start,
                    f"{config['learning_rate']:g}",
                    f"{config['epochs']:g}",
                    f"{train['seconds'] / 60:.0f}",
                    f"{train['peak_gpu_gb']:.1f}",
                    f"{train['final_eval_loss']:.4f}",
                    *quality_cells(overall),
                ]
            )
        )
        # Epoch checkpoints, taken before the learning rate decayed; their loss is the eval nearest that epoch.
        for checkpoint in sorted((RUNS / name).glob("epoch-*")):
            epoch = int(checkpoint.name.removeprefix("epoch-"))
            scores_file = RUNS / name / f"eval-{checkpoint.name}" / f"{stem}.scores.json"
            overall = json.loads(scores_file.read_text(encoding="utf-8"))["overall"] if scores_file.exists() else {}
            nearest = eval_loss_at(train, epoch)
            lines.append(
                _row(
                    [
                        f"{name} @ epoch {epoch}",
                        method,
                        start,
                        f"{config['learning_rate']:g}",
                        f"{epoch} of {config['epochs']:g}",
                        "",
                        "",
                        f"{nearest:.4f}" if nearest is not None else "",
                        *quality_cells(overall),
                    ]
                )
            )
    return "\n".join(lines)


def comparison_table(loaded: dict[str, dict], group: str | None = None) -> str:
    """Candidates' scores (label → a scores file's contents) side by side, overall or per member of one group."""
    with_faults = any("faults" in scores["overall"] for scores in loaded.values())
    header = ["model", *([group] if group else []), "rows", *QUALITY_COLUMNS, *(FAULT_COLUMNS if with_faults else [])]
    lines = [_row(header), _row(["---"] * len(header))]
    for label, scores in loaded.items():
        section = scores["overall" if group is None else group]
        # `overall` and `commands` are one summary; the other groups map members to summaries.
        sections = {group or "": section} if "rows" in section else section
        for member, summary in sections.items():
            faults = fault_cells(summary) if with_faults else []
            lines.append(_row([label, *([member] if group else []), str(summary["rows"]), *quality_cells(summary), *faults]))
    return "\n".join(lines)
