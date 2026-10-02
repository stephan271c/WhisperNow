"""Markdown tables from the files runs write: `train.json` and `eval*/<split>.scores.json`."""

from __future__ import annotations

import json
from pathlib import Path

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
}


def _rate(summary: dict, name: str) -> str:
    value = summary.get(name)
    if value is None:
        return ""
    return f"{value['rate']:.1f}" if isinstance(value, dict) else f"{value:.1f}"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def quality_cells(summary: dict) -> list[str]:
    return [_rate(summary, name) for name in QUALITY_COLUMNS.values()]


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
            losses = [entry for entry in train["log_history"] if "eval_loss" in entry]
            nearest = min(losses, key=lambda entry: abs(entry["epoch"] - epoch)) if losses else None
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
                        f"{nearest['eval_loss']:.4f}" if nearest else "",
                        *quality_cells(overall),
                    ]
                )
            )
    return "\n".join(lines)


def comparison_table(candidates: dict[str, Path], group: str | None = None) -> str:
    """Candidates (label → scores.json) side by side, overall or per member of one group."""
    loaded = {label: json.loads(path.read_text(encoding="utf-8")) for label, path in candidates.items()}
    header = ["model", *([group] if group else []), "rows", *QUALITY_COLUMNS]
    lines = [_row(header), _row(["---"] * len(header))]
    for label, scores in loaded.items():
        section = scores["overall" if group is None else group]
        # `overall` and `commands` are one summary; the other groups map members to summaries.
        sections = {group or "": section} if "rows" in section else section
        for member, summary in sections.items():
            lines.append(_row([label, *([member] if group else []), str(summary["rows"]), *quality_cells(summary)]))
    return "\n".join(lines)
