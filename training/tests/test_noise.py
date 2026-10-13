import asyncio
import json
import re

import pytest

from conftest import judge_items

from whispernow_training.datagen import gates, noise
from whispernow_training.datagen.schema import read_jsonl


def test_detect_sees_each_code_visible_pattern():
    assert noise.detect("we could uh um maybe push it") == ["filler_cluster"]
    assert "determiner_filler" in noise.detect("grab the uh the blue folder")
    assert noise.detect("I don't I don't know") == ["repeats"]
    assert noise.detect("she said that that plan won't work") == []
    assert noise.detect("Hello? Can you hear me? Okay so the plan") == ["thinking_aloud"]
    assert noise.detect("it's in the. second drawer") == ["pause_punct"]
    assert noise.detect("Buy garlic? Onions, spinach.") == ["pause_punct"]
    assert noise.detect("the brackets are in the. Back of the van") == ["pause_punct"]
    assert noise.detect("It is done. We can go.") == []


def test_shows_uses_evidence_for_restarts_and_corrections():
    text = "call them about I need to call them about the invoice, on Friday, sorry, Saturday"
    assert noise.shows("restart", text, ["call them about I need to call them about"])
    assert not noise.shows("restart", text, ["the invoice"])
    assert noise.shows("short_cue", text, ["Friday, sorry, Saturday"])
    # Evidence must be in the transcript.
    assert not noise.shows("short_cue", text, ["Monday, no, Tuesday"])
    decided = "the pasta or hmm actually no, let's do the risotto"
    assert noise.shows("redecision", decided, [decided])
    reasoned = "the blue one, except it's out of stock, so let's get the grey one instead"
    assert noise.shows("reasoned", reasoned, [reasoned])
    assert not noise.shows("reasoned", decided, [decided])
    assert noise.shows("restart", "Buy, um we need to buy two blocks", ["Buy, um we need to buy"])
    assert noise.shows("redecision", "a pilsner, or no, make it the stout", ["or no, make it the stout"])
    split = "the hall is booked all week, so let's use the studio instead"
    assert noise.shows("reasoned", split, ["let's use the studio instead"])


def test_control_rows_have_no_fillers_and_references_keep_the_words():
    text = "I'd like the red one, and I mean it."
    assert noise.shows("control", text, ["I'd like the red one", "I mean it"])
    assert not noise.shows("control", "um I'd like the red one", ["I'd like the red one"])
    assert noise.keeps_control_words("I'd like the red one, and I mean it.", ["I'd like the red one", "I mean it"])
    assert not noise.keeps_control_words("I want the red one, seriously.", ["I'd like the red one", "I mean it"])
    # A repeated "that" and a discourse "you know" may go; the content stays.
    assert noise.keeps_control_words("Tell Bjorn that plan won't work.", ["that that plan won't work"])
    assert not noise.keeps_control_words("We're sticking with them, and they were cheaper.", ["it went well"])


def reply(messages, role):
    if role == "judge":
        return json.dumps({"results": [{"i": i["i"], **{c: True for c in gates.COMPOSE_CRITERIA}} for i in judge_items(messages)]})
    user = messages[-1]["content"]
    if role == "teacher":
        return re.search(r"<transcript>\n(.*)\n</transcript>", user, re.S).group(1).replace("uh um ", "")
    assert role == "generator"
    batch = int(re.search(r"Batch (\d+)\.", user).group(1))
    items = [
        {"input": f"so for batch {batch} row {i} we could uh um maybe push it", "evidence": ["uh um"], "layout": "prose"}
        for i in range(9)
    ]
    items.append({"input": f"batch {batch} has no pattern at all", "evidence": ["pattern"], "layout": "prose"})
    return json.dumps({"items": items})


def test_generate_kind_keeps_rows_with_the_pattern_and_tags_them(fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(noise, "OUT_DIR", tmp_path)
    kept = asyncio.run(noise.generate_kind(fake_llm(reply), "filler_cluster", 12, set(), progress=lambda _: None))
    assert kept == 12
    rows = noise.load()
    assert all("pattern:filler_cluster" in r.tags and "noise_bench" in r.tags for r in rows)
    rejected = [json.loads(line) for line in (tmp_path / "filler_cluster.rejected.jsonl").open()]
    assert rejected and all(r["failed"] == ["pattern"] for r in rejected)


def test_client_type_failure_preserves_batches_and_resumes_after_repair(fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(noise, "OUT_DIR", tmp_path)
    generator_requests = []

    def malformed_then_valid(messages, role):
        if role == "generator":
            generator_requests.append(messages)
            if len(generator_requests) == 1:
                return "no json here"
        return reply(messages, role)

    llm = fake_llm(malformed_then_valid)
    assert asyncio.run(noise.generate_kind(llm, "filler_cluster", 1, set(), progress=lambda _: None)) == 1
    log_path = tmp_path / "filler_cluster.batches.jsonl"
    log = list(read_jsonl(log_path))
    assert len(log) == 2 and log[0]["error"].startswith("unreadable generator reply:")
    assert log[0]["returned"] == log[0]["passed"] == 0
    prior_files = {path: path.read_bytes() for path in tmp_path.glob("*.jsonl")}
    prior_rows = noise.load()
    known = {row.input for row in prior_rows}
    failure = TypeError("invalid client configuration")
    failed_requests = []

    def unavailable(messages, role):
        failed_requests.append(messages)
        raise failure

    llm.reply = unavailable
    with pytest.raises(TypeError) as error:
        asyncio.run(noise.generate_kind(llm, "filler_cluster", 2, known, progress=lambda _: None))
    assert error.value is failure and len(failed_requests) == 1
    assert {path: path.read_bytes() for path in tmp_path.glob("*.jsonl")} == prior_files

    llm.reply = malformed_then_valid
    assert asyncio.run(noise.generate_kind(llm, "filler_cluster", 2, known, progress=lambda _: None)) == 2
    assert generator_requests[-1] == failed_requests[0]
    assert noise.load()[:len(prior_rows)] == prior_rows
    resumed_log = list(read_jsonl(log_path))
    assert resumed_log[:-1] == log and not resumed_log[-1]["error"]


def test_assemble_swaps_in_kept_probed_rows(fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(noise, "OUT_DIR", tmp_path)
    monkeypatch.setattr(noise, "BENCH_FILE", tmp_path / "noise.jsonl")
    asyncio.run(noise.generate_kind(fake_llm(reply), "filler_cluster", 3, set(), progress=lambda _: None))
    first = noise.load()[0]
    heard = "So for batch zero, row zero, we could uh um maybe push it."
    (tmp_path / "probed.gated.jsonl").write_text(
        json.dumps({"id": first.id, "heard": heard, "output": "So we could maybe push it.", "attempt": 0, "kept": True, "reason": ""}) + "\n"
    )
    summary = noise.assemble()
    assert summary["rows"] == 3 and summary["probed"] == 1
    probed = [json.loads(line) for line in (tmp_path / "noise.jsonl").open() if "asr_probe" in line]
    assert probed[0]["input"] == heard and f"probed_from:{first.id}" in probed[0]["tags"]
    assert "pattern:filler_cluster" in probed[0]["tags"]
