import asyncio
import dataclasses
import json
import re

import pytest

from conftest import judge_items

from whispernow_training.datagen import compose, gates
from whispernow_training.datagen.schema import V2_TAG, Example, read_jsonl, write_examples


def _words(n, tag):
    return " ".join(f"{tag}{i}" for i in range(n))


def reply(messages, role):
    if role == "judge":
        results = [{"i": item["i"], **{c: True for c in gates.COMPOSE_CRITERIA}, "note": ""} for item in judge_items(messages)]
        return json.dumps({"results": results})
    user = messages[-1]["content"]
    if role == "teacher":
        transcript = re.search(r"<transcript>\n(.*)\n</transcript>", user, re.S).group(1)
        return transcript.capitalize() + "."
    assert role == "generator"
    batch = int(re.search(r"Batch (\d+)\.", user).group(1))
    count = int(re.search(r"Write (\d+) transcripts", user).group(1))
    low, high = map(int, re.search(r"each (\d+) to (\d+) words", user).groups())
    size = (low + high) // 2
    items = [{"input": _words(size, f"b{batch}i{i}w"), "layout": "prose", "tags": ["late_correction"]} for i in range(count - 1)]
    items.insert(0, {"input": _words(40, f"short{batch}"), "layout": "prose"})  # under 150 words
    items.insert(1, {"input": _words(200, f"off{batch}"), "layout": "prose"})  # in another bucket
    return json.dumps({"items": items})


def test_targets_split_by_bucket_shares():
    goals = compose.targets(800)
    per_bucket = {b: sum(n for (_, bucket), n in goals.items() if bucket == b) for b in compose.BUCKETS}
    assert per_bucket == {"150-300": 320, "300-600": 280, "600-1000": 200}
    bench = compose.targets(360, compose.BENCH_SHARES)
    assert {sum(n for (_, bucket), n in bench.items() if bucket == b) for b in compose.BUCKETS} == {120}


def test_targeted_targets_follow_the_category_weights_and_bucket_shares():
    goals = compose.targets(1000, compose.TARGETED_SHARES, compose.CATEGORIES, compose.TARGETED_WEIGHTS)
    per_category = {c: sum(n for (category, _), n in goals.items() if category == c) for c in compose.CATEGORIES}
    per_bucket = {b: sum(n for (_, bucket), n in goals.items() if bucket == b) for b in compose.BUCKETS}
    assert per_category["revised_choice"] == 320 and per_category["recipe_quantities"] == 220
    assert per_bucket == {"150-300": 200, "300-600": 400, "600-1000": 400}
    one = compose.targets(100, compose.TARGETED_SHARES, ["revised_choice"], compose.TARGETED_WEIGHTS)
    assert sum(one.values()) == 100


def test_file_bucket_follows_the_length_and_stays_in_150_to_1000():
    assert compose.file_bucket(150) is None and compose.file_bucket(151) == "150-300"
    assert compose.file_bucket(320) == "300-600" and compose.file_bucket(1000) == "600-1000"
    assert compose.file_bucket(1001) is None


def test_bench_and_train_requests_share_no_prompt_names_or_settings():
    train = compose.generator_messages(compose.TRAIN, "revised_choice", "150-300", "recognizer", 6, 0, [])
    bench = compose.generator_messages(compose.BENCH, "revised_choice", "150-300", "recognizer", 6, 0, [])
    assert train[0]["content"] != bench[0]["content"]
    assert not set(compose.TRAIN.names) & set(compose.BENCH.names)
    assert not set(compose.TRAIN.settings) & set(compose.BENCH.settings)
    assert compose.style_for(compose.BENCH, "revised_choice", "150-300", 3) == compose.style_for(compose.BENCH, "revised_choice", "150-300", 3)


def test_targeted_pools_share_nothing_with_the_training_or_benchmark_pools():
    for other in (compose.TRAIN, compose.BENCH):
        assert not set(compose.TARGETED.names) & set(other.names)
        assert not set(compose.TARGETED.settings) & set(other.settings)


@pytest.fixture(autouse=True)
def one_lane(monkeypatch):
    """One batch at a time per cell, so a test's batches run in order."""
    monkeypatch.setattr(compose, "LANES", dict.fromkeys(compose.BUCKETS, 1))


def test_a_cell_runs_its_lanes_at_once_with_distinct_batches_and_resumes_after_the_last(fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(compose, "LANES", {**compose.LANES, "300-600": 3})
    run = dataclasses.replace(compose.BENCH, out_dir=tmp_path)
    llm = fake_llm(reply)
    counts = compose.counts_by_cell(run)
    original_complete = llm.complete

    async def observe_overlap():
        entered = []
        all_entered = asyncio.Event()
        release = asyncio.Event()

        async def complete(messages, *, role="teacher", **kwargs):
            if role == "generator" and not release.is_set():
                batch = int(re.search(r"Batch (\d+)\.", messages[-1]["content"]).group(1))
                entered.append(batch)
                if len(entered) == 3:
                    all_entered.set()
                await release.wait()
            return await original_complete(messages, role=role, **kwargs)

        monkeypatch.setattr(llm, "complete", complete)
        task = asyncio.create_task(compose.generate_cell(
            llm, run, "revised_choice", "300-600", 4, set(), counts, progress=lambda _: None
        ))
        try:
            try:
                await asyncio.wait_for(all_entered.wait(), timeout=5)
            except TimeoutError:
                pytest.fail(f"Expected three overlapping generator requests; observed {entered}")
            assert len(entered) == 3 and set(entered) == {0, 1, 2}
            assert not task.done()
            release.set()
            return await asyncio.wait_for(task, timeout=5)
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            monkeypatch.setattr(llm, "complete", original_complete)

    # Each lane checks the count before starting, so the cell stops within its lanes' batches.
    assert 4 <= asyncio.run(observe_overlap()) < 4 + 3
    batches = [json.loads(line)["batch"] for line in (tmp_path / "revised_choice.300-600.batches.jsonl").open()]
    assert len(set(batches)) == len(batches)
    # A batch lost to a stop leaves a gap; the run continues after the highest logged batch.
    log = tmp_path / "revised_choice.300-600.batches.jsonl"
    log.write_text("".join(line for line in log.open() if json.loads(line)["batch"] != min(batches)))
    asyncio.run(compose.generate_cell(llm, run, "revised_choice", "300-600", 100, set(), counts, progress=lambda _: None))
    later = [json.loads(line)["batch"] for line in log.open()]
    assert min(batches) not in later and len(set(later)) == len(later)
    assert min(set(later) - set(batches)) == max(batches) + 1


def test_generate_cell_files_rows_by_length_rejects_short_ones_and_resumes(fake_llm, tmp_path):
    run = dataclasses.replace(compose.BENCH, out_dir=tmp_path)
    llm = fake_llm(reply)
    counts = compose.counts_by_cell(run)
    assert asyncio.run(compose.generate_cell(llm, run, "revised_choice", "300-600", 4, set(), counts, progress=lambda _: None)) == 4
    rows = compose.load(run)
    in_bucket = [r for r in rows if len(r.input.split()) == 450]
    elsewhere = [r for r in rows if len(r.input.split()) == 200]
    assert len(in_bucket) == 4 and len(elsewhere) == len(rows) - 4 >= 1
    assert all({"composed", "length:300-600"} <= set(r.tags) and r.category == "revised_choice" for r in in_bucket)
    assert all("length:150-300" in r.tags for r in elsewhere)
    assert (tmp_path / "revised_choice.150-300.jsonl").exists()
    rejected = [json.loads(line) for line in (tmp_path / "revised_choice.300-600.rejected.jsonl").open()]
    assert rejected and all(r["failed"] == ["length"] for r in rejected)
    log = [json.loads(line) for line in (tmp_path / "revised_choice.300-600.batches.jsonl").open()]
    assert log[0]["notes"] and "length" in log[0]["notes"][0]
    calls = llm.calls
    assert asyncio.run(compose.generate_cell(llm, run, "revised_choice", "300-600", 4, set(), counts, progress=lambda _: None)) == 4
    assert llm.calls == calls


def long_rows(messages, role):
    if role == "generator":
        batch = int(re.search(r"Batch (\d+)\.", messages[-1]["content"]).group(1))
        return json.dumps({"items": [{"input": _words(800, f"long{batch}w"), "layout": "prose"}]})
    return reply(messages, role)


def test_a_cell_short_of_one_row_runs_one_batch_whatever_its_lanes(fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(compose, "LANES", dict.fromkeys(compose.BUCKETS, 12))
    run = dataclasses.replace(compose.BENCH, out_dir=tmp_path)
    counts = compose.counts_by_cell(run)
    asyncio.run(compose.generate_cell(fake_llm(long_rows), run, "revised_choice", "600-1000", 1, set(), counts, progress=lambda _: None))
    assert len(list(read_jsonl(tmp_path / "revised_choice.600-1000.batches.jsonl"))) == 1


def test_targeted_rows_carry_the_run_tag_and_the_teacher_reasoning(fake_llm, tmp_path):
    run = dataclasses.replace(compose.TARGETED, out_dir=tmp_path)
    counts = compose.counts_by_cell(run)
    asyncio.run(compose.generate_cell(fake_llm(long_rows), run, "revised_choice", "600-1000", 1, set(), counts, progress=lambda _: None))
    rows = compose.load(run)
    assert len(rows) == 1 and {compose.TARGETED_TAG, V2_TAG} <= set(rows[0].tags) and rows[0].teacher_reasoning == "xhigh"


def test_rows_rejects_and_batch_records_name_the_generator_teacher_and_judge(fake_llm, tmp_path):
    run = dataclasses.replace(compose.TARGETED, out_dir=tmp_path)
    counts = compose.counts_by_cell(run)
    asyncio.run(compose.generate_cell(fake_llm(reply), run, "revised_choice", "300-600", 1, set(), counts, progress=lambda _: None))
    row = compose.load(run)[0]
    assert (row.generator_provider, row.generator_model, row.generator_reasoning) == ("openai", "teacher-model", "low")
    assert (row.teacher_model, row.judge_model) == ("teacher-model", "judge-model")
    # Rows up to 600 words are judged at low whatever the judge's setting.
    assert row.judge_reasoning == "low"
    rejected = list(read_jsonl(tmp_path / "revised_choice.300-600.rejected.jsonl"))
    assert rejected and {r["generator"] for r in rejected} == {"openai:teacher-model#low"}
    record, = read_jsonl(tmp_path / "revised_choice.300-600.batches.jsonl")
    assert record["generator"] == "openai:teacher-model#low"


def test_load_gives_rows_written_before_a_run_tag_existed_that_tag(tmp_path):
    run = dataclasses.replace(compose.TARGETED, out_dir=tmp_path)
    old = Example(id="old", input="buy eggs", output="Buy eggs.", source="teacher", category="revised_choice", license="MIT", tags=["targeted"])
    write_examples(compose.output_file(run, "revised_choice", "150-300"), [old])
    assert compose.load(run)[0].tags == ["targeted", V2_TAG]


def test_client_request_failure_preserves_completed_cells_and_resumes_after_repair(fake_llm, tmp_path):
    run = dataclasses.replace(compose.BENCH, out_dir=tmp_path)
    generator_requests = []

    def malformed_then_valid(messages, role):
        if role == "generator":
            generator_requests.append(messages)
            if len(generator_requests) == 1:
                return '{"wrong_key": []}'
        return reply(messages, role)

    llm = fake_llm(malformed_then_valid)
    counts = compose.counts_by_cell(run)
    assert asyncio.run(compose.generate_cell(llm, run, "revised_choice", "300-600", 2, set(), counts, progress=lambda _: None)) == 2
    log_path = tmp_path / "revised_choice.300-600.batches.jsonl"
    log = list(read_jsonl(log_path))
    assert len(log) == 3 and log[0]["error"].startswith("unreadable generator reply:")
    assert log[0]["returned"] == log[0]["passed"] == 0
    prior_files = {path: path.read_bytes() for path in tmp_path.glob("*.jsonl")}
    prior_rows = compose.load(run)
    known = {row.input for row in prior_rows}
    failure = ValueError("invalid client request")
    failed_requests = []

    def unavailable(messages, role):
        failed_requests.append(messages)
        raise failure

    llm.reply = unavailable
    with pytest.raises(ValueError) as error:
        asyncio.run(compose.generate_cell(llm, run, "revised_choice", "300-600", 3, known, counts, progress=lambda _: None))
    assert error.value is failure and len(failed_requests) == 1
    assert counts == compose.counts_by_cell(run)
    assert {path: path.read_bytes() for path in tmp_path.glob("*.jsonl")} == prior_files

    llm.reply = malformed_then_valid
    counts = compose.counts_by_cell(run)
    assert asyncio.run(compose.generate_cell(llm, run, "revised_choice", "300-600", 3, known, counts, progress=lambda _: None)) == 3
    assert generator_requests[-1] == failed_requests[0]
    rows = {row.id: row for row in compose.load(run)}
    assert all(rows[row.id] == row for row in prior_rows)
    resumed_log = list(read_jsonl(log_path))
    assert resumed_log[:-1] == log and not resumed_log[-1]["error"]


def test_rows_over_600_words_get_their_own_judge_and_teacher_reasoning(fake_llm, tmp_path):
    def long_and_short(messages, role):
        if role == "generator":
            items = [{"input": _words(800, "long"), "layout": "prose"}, {"input": _words(200, "off"), "layout": "prose"}]
            return json.dumps({"items": items})
        return reply(messages, role)

    run = dataclasses.replace(compose.BENCH, out_dir=tmp_path)
    llm = fake_llm(long_and_short)
    counts = compose.counts_by_cell(run)
    asyncio.run(compose.generate_cell(llm, run, "revised_choice", "600-1000", 1, set(), counts, progress=lambda _: None))
    # The 800-word row keeps the judge's setting (None); the 200-word row is checked at low.
    assert sorted(map(str, (r for role, r in llm.reasonings if role == "judge"))) == ["None", "low"]
    # The teacher writes the 800-word row's reference at xhigh and the 200-word row's at its setting.
    assert sorted(map(str, (r for role, r in llm.reasonings if role == "teacher"))) == ["None", "xhigh"]


def test_a_reference_against_the_intended_layout_is_dropped_before_the_judge_sees_it(fake_llm, tmp_path):
    def list_meant(messages, role):
        if role == "generator":
            return json.dumps({"items": [{"input": _words(450, "steps"), "layout": "list"}]})
        return reply(messages, role)

    run = dataclasses.replace(compose.BENCH, out_dir=tmp_path)
    llm = fake_llm(list_meant)
    counts = compose.counts_by_cell(run)
    asyncio.run(compose.generate_cell(llm, run, "technical_steps", "300-600", 1, set(), counts, progress=lambda _: None))
    rejected = [json.loads(line) for line in (tmp_path / "technical_steps.300-600.rejected.jsonl").open()]
    assert rejected and all(r["failed"] == ["layout_intent"] for r in rejected)
    assert not [role for role, _ in llm.reasonings if role == "judge"]
