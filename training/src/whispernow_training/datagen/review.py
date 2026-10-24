"""A second look at teacher references, by the same model, with evidence the teacher did not have.

- `disfl_qa`: a Disfl-QA reference against the written question its transcript was made from
  (`prompts/review/disfl_qa.md`), for references that keep words the speaker abandoned.
- `literal_case`: a reference line that starts with a capitalized command or file name
  (`prompts/review/literal_case.md`), which the contract keeps as dictated.

A proposed fix is then shown blind, old and new reference in random order, to the judge role
(`prompts/review/verify.md`), and counts only when the new reference wins.

`review_build` runs both on data/build-v2-no-heard's train and test rows and writes `FIXES`. The v3
build applies the verified train fixes; the verified test fixes were written into the v1 test file,
whose old lines `data/fixes/test-before.jsonl` keeps.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Sequence
from functools import cache

from . import paths, sources
from .llm import LlmClient, LlmError, parse_json_reply
from .teacher import EDIT_REQUEST_FILE, SYSTEM_FILE

FIXES = paths.DATA / "fixes" / "refs.jsonl"
KINDS = ("disfl_qa", "literal_case")
PER_REQUEST = {"disfl_qa": 20, "literal_case": 8}
VERIFY_PER_REQUEST = 20
ATTEMPTS = 3

SUB = {
    "git": "push|pull|stash|checkout|commit|status|log|diff|add|clone|fetch|merge|rebase|reset|branch|switch|tag|init|remote|cherry-pick|restore|bisect|revert",
    "npm": "install|i|run|ci|init|test|start|publish|update|uninstall|audit|link",
    "npx": r"[a-z@][\w@/.-]*",
    "yarn": "add|install|run|dev|build|test|remove",
    "pnpm": "add|install|run|dev|build|test|remove",
    "pip": "install|uninstall|freeze|list|show",
    "pip3": "install|uninstall|freeze|list",
    "uv": "add|sync|run|pip|venv|init|lock|remove|tool",
    "cargo": "build|run|test|publish|add|new|check|clippy|fmt|install|update|bench",
    "docker": "compose|run|build|ps|pull|push|exec|images|logs|stop|rm|start|login|tag|system|volume|network",
    "kubectl": "get|apply|describe|logs|delete|exec|rollout|scale|port-forward|config|create|edit",
    "helm": "install|upgrade|repo|uninstall|list|template|rollback|lint",
    "terraform": "init|plan|apply|destroy|fmt|validate|import|state|workspace",
    "brew": "install|update|upgrade|uninstall|services|tap",
    "apt": "install|update|upgrade|remove",
    "systemctl": "start|stop|restart|status|enable|disable|reload",
    "gh": "pr|repo|issue|run|auth|release",
    "conda": "install|create|activate|env",
    "poetry": "add|install|run|lock|build",
    "go": "build|run|test|mod|get|install|fmt|vet",
}
ARG = ("cd ls mkdir rm cp mv chmod chown kill sudo ssh scp curl wget tar grep cat touch make python python3 node "
       "source export psql ffmpeg").split()
_START = r"^(\s*(?:[-*•]|\d+[.)])?\s*)"
_COMMAND_SUB = re.compile(_START + r"(" + "|".join(sorted(SUB)) + r")\s+(\S+)", re.I | re.M)
_COMMAND_ARG = re.compile(_START + r"(" + "|".join(sorted(ARG)) + r")\s+(-{1,2}[\w-]+|[~./][\w./~-]*|[\w-]+/[\w./-]*|\$\w+)", re.I | re.M)
_FILE = re.compile(_START + r"([A-Z][\w-]*\.(?:toml|json|ya?ml|lock|py|rs|ts|js|tsx|jsx|sh|env|cfg|ini|txt))\b", re.M)


def words(text: str) -> list[str]:
    return re.sub(r"[^\w ]", " ", text.lower()).split()


def capitalized_literal(output: str) -> str | None:
    """The first line that starts with a capitalized command invocation (a tool and its subcommand or an
    argument) or file name, such as "Docker compose up -d."; prose about a tool matches too."""
    for match in _COMMAND_SUB.finditer(output):
        tool, following = match.group(2), match.group(3)
        if tool[0].isupper() and re.fullmatch(r"(?:" + SUB[tool.lower()] + r")[.,:;)]*|-{1,2}[\w-]+", following):
            return output[match.start(2):].split("\n")[0]
    for pattern in (_COMMAND_ARG, _FILE):
        for match in pattern.finditer(output):
            if match.group(2)[0].isupper():
                return output[match.start(2):].split("\n")[0]
    return None


@cache
def _contract() -> str:
    return (
        "The model's editing contract (its system message, then the edit request it gets with every transcript):\n\n"
        + SYSTEM_FILE.read_text().strip() + "\n\n" + EDIT_REQUEST_FILE.read_text().strip()
    )


def _system(name: str) -> str:
    return (paths.PROMPTS / "review" / f"{name}.md").read_text(encoding="utf-8").replace("{contract}", _contract())


@cache
def disfl_originals() -> dict[str, str]:
    """Each Disfl-QA row's original written question, by row id."""
    return {e.id: e.output for e in sources.load_disfl_qa()}


def _review_request(kind: str, rows: Sequence[dict]) -> list[dict]:
    parts = []
    for i, row in enumerate(rows):
        if kind == "disfl_qa":
            parts.append(f"[{i}]\nTRANSCRIPT: {row['input']}\nORIGINAL: {row['original']}\nREFERENCE: {row['output']}")
        else:
            parts.append(f"[{i}]\nTRANSCRIPT:\n{row['input']}\n\nREFERENCE:\n{row['output']}\n\nFLAGGED: {row['flagged']}")
    return [{"role": "system", "content": _system(kind)}, {"role": "user", "content": "\n\n".join(parts)}]


def _verify_request(rows: Sequence[dict], flips: Sequence[bool]) -> list[dict]:
    parts = []
    for i, (row, flip) in enumerate(zip(rows, flips, strict=True)):
        a, b = (row["new_output"], row["output"]) if flip else (row["output"], row["new_output"])
        original = f"ORIGINAL: {row['original']}\n" if row["kind"] == "disfl_qa" else ""
        parts.append(f"[{i}]\nTRANSCRIPT: {row['input']}\n{original}A: {a}\nB: {b}")
    return [{"role": "system", "content": _system("verify")}, {"role": "user", "content": "\n\n".join(parts)}]


async def _results(llm: LlmClient, messages_for_attempt, count: int, role: str, what: str) -> dict[int, dict] | None:
    for attempt in range(ATTEMPTS):
        try:
            reply = parse_json_reply(await llm.complete(messages_for_attempt(), role=role, attempt=attempt))
            results = {r["i"]: r for r in reply["results"]}
            if set(results) == set(range(count)):
                return results
        except (LlmError, KeyError, TypeError, ValueError) as error:
            print(f"{what}: attempt {attempt} failed: {str(error)[:200]}", flush=True)
    return None


async def review(llm: LlmClient, kind: str, rows: Sequence[dict]) -> list[dict]:
    """`rows` (id, input, output, and `original` or `flagged`) with the teacher's verdict, `ok` or `fix`
    with `new_output`, or `error`."""
    results = await _results(llm, lambda: _review_request(kind, rows), len(rows), "teacher", f"{kind} review of {rows[0]['id']}")
    if results is None:
        return [{**row, "kind": kind, "verdict": "error", "new_output": None, "note": ""} for row in rows]
    reviewed = []
    for i, row in enumerate(rows):
        result = results[i]
        new = str(result.get("output", "")).strip()
        fixed = result.get("verdict") == "fix" and new not in ("", row["output"].strip())
        reviewed.append({**row, "kind": kind, "verdict": "fix" if fixed else "ok", "new_output": new if fixed else None,
                         "note": result.get("note", "")})
    return reviewed


async def verify(llm: LlmClient, rows: Sequence[dict]) -> list[dict]:
    """Fixes from `review` with `verified`: whether the judge, shown both references blind, picked the new one."""
    flips = [int(hashlib.sha256(row["id"].encode()).hexdigest(), 16) % 2 == 1 for row in rows]
    results = await _results(llm, lambda: _verify_request(rows, flips), len(rows), "judge", f"verify of {rows[0]['id']}")
    if results is None:
        return [{**row, "verified": None} for row in rows]
    out = []
    for i, (row, flip) in enumerate(zip(rows, flips, strict=True)):
        better = results[i].get("better", "same")
        new_won = better == ("A" if flip else "B")
        out.append({**row, "verified": new_won, "verify_choice": "new" if new_won else ("same" if better == "same" else "old"),
                    "verify_note": results[i].get("note", "")})
    return out


async def check(llm: LlmClient, kind: str, rows: Sequence[dict]) -> list[dict]:
    """`review` then `verify` for every row of one kind; a row's reference to keep is `final_output`."""
    size = PER_REQUEST[kind]
    batches = await asyncio.gather(*(review(llm, kind, rows[i:i + size]) for i in range(0, len(rows), size)))
    reviewed = [row for batch in batches for row in batch]
    fixes = [row for row in reviewed if row["verdict"] == "fix"]
    checked = await asyncio.gather(*(verify(llm, fixes[i:i + VERIFY_PER_REQUEST]) for i in range(0, len(fixes), VERIFY_PER_REQUEST)))
    verified = {row["id"]: row for batch in checked for row in batch}
    out = []
    for row in reviewed:
        row = verified.get(row["id"], row)
        out.append({**row, "final_output": row["new_output"] if row.get("verified") else row["output"]})
    return out


def disfl_row(row_id: str, transcript: str, reference: str, **extra) -> dict | None:
    """A review row for a Disfl-QA reference that differs from its original question, else None."""
    original = disfl_originals().get(row_id)
    if original is None or words(reference) == words(original):
        return None
    return {"id": row_id, "input": transcript, "output": reference, "original": original, **extra}


def case_row(row_id: str, transcript: str, reference: str, **extra) -> dict | None:
    flagged = capitalized_literal(reference)
    return {"id": row_id, "input": transcript, "output": reference, "flagged": flagged, **extra} if flagged else None


def build_candidates(limit: int = 0) -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {kind: [] for kind in KINDS}
    for split in ("train", "test"):
        for line in (paths.BUILD_V2_NO_HEARD / f"{split}.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            if row["source"].endswith("disfl_qa"):
                if candidate := disfl_row(row["id"], row["input"], row["output"], split=split):
                    found["disfl_qa"].append(candidate)
            elif candidate := case_row(row["id"], row["input"], row["output"], split=split):
                found["literal_case"].append(candidate)
    return {kind: rows[:limit] if limit else rows for kind, rows in found.items()}


async def review_build(llm: LlmClient, limit: int = 0) -> dict:
    """Reviews and verifies data/build-v2-no-heard's suspect train and test references into `FIXES`."""
    found = build_candidates(limit)
    done = [row for kind, rows in found.items() for row in await check(llm, kind, rows)]
    FIXES.parent.mkdir(parents=True, exist_ok=True)
    with FIXES.open("w", encoding="utf-8") as out:
        for row in done:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {}
    for kind in KINDS:
        for split in ("train", "test"):
            mine = [r for r in done if r["kind"] == kind and r["split"] == split]
            report[f"{kind} {split}"] = {"rows": len(mine), "fix proposed": sum(r["verdict"] == "fix" for r in mine),
                                         "verified": sum(bool(r.get("verified")) for r in mine),
                                         "error": sum(r["verdict"] == "error" for r in mine)}
    return report


def verified_fixes(split: str) -> dict[str, str]:
    """The verified fixed reference of each `split` row in `FIXES`, by id."""
    if not FIXES.exists():
        return {}
    rows = (json.loads(line) for line in FIXES.open(encoding="utf-8"))
    return {row["id"]: row["new_output"] for row in rows if row["split"] == split and row.get("verified")}
