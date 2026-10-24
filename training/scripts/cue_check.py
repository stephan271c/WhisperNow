"""Correction words in eval outputs: on test and bench-noise rows whose transcript says one, how often each eval
dir keeps it where the reference keeps it (the word was meant) and drops it where the reference drops it (the
speaker was correcting themselves). A word-level check, no judge.

    .venv/bin/python scripts/cue_check.py runs/RUN/eval-gpu-q8 [runs/OTHER/eval-gpu-q8 ...]
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from whispernow_training.datagen.schema import read_examples  # noqa: E402
from whispernow_training.model.evaluate import BENCHMARKS, load_rows, read_outputs  # noqa: E402

CUES = {
    "scratch": r"\bscratch", "never mind": r"\bnever ?mind\b", "forget": r"\bforget", "hold on": r"\bhold on\b",
    "wait": r"\bwait\b", "that's wrong": r"\bthat'?s (?:wrong|not right)\b", "sorry": r"\bsorry\b",
    "make that": r"\bmake that\b", "change that": r"\bchange that\b", "correction": r"\bcorrection\b", "I mean": r"\bi mean\b",
}
SPLITS = ("test", "bench-noise")


def cases() -> list[tuple[str, str, str, bool, bool]]:
    """(split, row id, cue, whether the reference keeps it, whether the transcript has at most 25 words)."""
    out = []
    for split in SPLITS:
        rows = load_rows(split) if split == "test" else read_examples(BENCHMARKS[split])
        for row in rows:
            for cue, pattern in CUES.items():
                if re.search(pattern, row.input, re.I):
                    out.append((split, row.id, cue, bool(re.search(pattern, row.output, re.I)), len(row.input.split()) <= 25))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    found = cases()
    outputs = {d: {split: read_outputs(d / f"{split}.outputs.jsonl") for split in SPLITS} for d in args.eval_dirs}

    def rate(d: Path, keep: bool, short: bool | None, cue: str | None = None) -> str:
        picked = [c for c in found if c[3] == keep and (short is None or c[4] == short) and (cue is None or c[2] == cue)]
        hits = sum(bool(re.search(CUES[c[2]], outputs[d][c[0]][c[1]]["output"], re.I)) == keep for c in picked)
        return f"{hits / len(picked):.0%} of {len(picked)}" if picked else "-"

    print("| eval | meant, kept (<=25 words) | meant, kept (all) | correction, dropped (<=25 words) | correction, dropped (all) |")
    print("|---|---|---|---|---|")
    for d in args.eval_dirs:
        print(f"| {d.parent.name} | {rate(d, True, True)} | {rate(d, True, None)} | {rate(d, False, True)} | {rate(d, False, None)} |")
    print("\nmeant and kept, by word (all lengths)\n")
    print("| word | " + " | ".join(d.parent.name for d in args.eval_dirs) + " |")
    print("|---|" + "---|" * len(args.eval_dirs))
    for cue in CUES:
        if any(c[2] == cue and c[3] for c in found):
            print(f"| {cue} | " + " | ".join(rate(d, True, None, cue) for d in args.eval_dirs) + " |")


if __name__ == "__main__":
    main()
