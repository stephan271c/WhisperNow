"""Markdown tables from the files runs write: `train.json` and `eval*/<split>.scores.json`."""

from __future__ import annotations

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
