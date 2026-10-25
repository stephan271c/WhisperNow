import json
import re
from collections import Counter
from dataclasses import replace

from whispernow_training import text_rules
from whispernow_training.datagen import paths
from whispernow_training.datagen.dataset import build, dedup, filters, round1_base, round2_targeted, round3_weak_spots, shape, stats
from whispernow_training.datagen.references import relabel, review
from whispernow_training.datagen.schema import EMPTY, ROUND2_TAG, ROUND3_TAG, Example, read_examples, write_examples
from whispernow_training.datagen.sources import aawaaz, aawaaz_audit, public, public_audit
from whispernow_training.datagen.speech import repair
from whispernow_training.datagen.synthetic import compose, generate, restyle


def row(id_, input_, output, source="owner/src", split="", seed_group="", tags=None):
    return Example(
        id=id_, input=input_, output=output, source=source, category="c", license="MIT",
        split=split, seed_group=seed_group, tags=tags or [],
    )


def test_markdown_is_caught_but_content_asterisks_are_not():
    assert text_rules.has_markdown("This is **bold** text.")
    assert text_rules.has_markdown("Run `ls` now.")
    assert text_rules.has_markdown("## Heading\nText")
    assert not text_rules.has_markdown("Use the 6-31G** basis set and x**2.")
    assert not text_rules.has_markdown("Edit __init__.py and #general.")
    assert not text_rules.has_markdown("Use __del__ because __del__ releases the object.")


def test_malformed_lists():
    assert text_rules.malformed_list("We need:\n* Milk") == "bullet other than '- '"
    assert text_rules.malformed_list("Steps:\n1. Build.\n3. Ship.") == "numbering skips"
    assert text_rules.malformed_list("We need:\n- \n- Eggs") == "empty list item"
    assert text_rules.malformed_list("Steps:\n1. Build.\n2. Ship.\n\nThen:\n1. Rest.") is None
    assert text_rules.malformed_list("Plain text.") is None
    assert text_rules.malformed_list("1. Milk\n1. Eggs") == "numbering skips"
    assert text_rules.malformed_list("1. Milk\n\n1. Eggs") is None
    assert text_rules.malformed_list("1. Milk\n\n2. Eggs") is None
    assert text_rules.malformed_list("1. Cook.\nLet it cool.\n2. Serve.") is None


def test_answered_outputs():
    assert filters.answered(row("a", "what's the capital of France", "Sure! The capital of France is Paris."))
    assert not filters.answered(row("b", "sure I can make it at 3", "Sure, I can make it at 3."))
    assert not filters.answered(row("c", "um here's a thought about news", "Here's a thought about news."))
    assert not filters.answered(row("d", "yeah so I can't connect to the vpn", "I can't connect to the VPN."))


def test_changed_protected_tokens():
    assert filters.changed_protected_token(row("a", "open readme.md and main_test.py", "Open README.md and main_test.py.")) == "readme.md"
    assert filters.changed_protected_token(row("b", "the x25 protocol", "The X.25 protocol.")) == "x25"
    assert filters.changed_protected_token(row("c", "send to a@b.com no wait c@d.com", "Send to c@d.com.")) is None
    assert filters.changed_protected_token(row("d", "run npm install --save-dev", "Run npm install --save-dev.")) is None
    assert filters.changed_protected_token(row("e", "co2 levels in q2 on k8s", "CO2 levels in Q2 on K8s.")) is None
    assert filters.changed_protected_token(row("f", "the PD-L1— marker", "The PD-L1 marker.")) is None
    for original, changed in (
        (r"C:\Users\Dana\Notes.txt", r"C:\Users\Dana\notes.txt"),
        (r"C:\Users\Dana", r"C:\users\Dana"),
        ("ComputeBudget::requestUnits", "ComputeBudget.requestUnits"),
        ("--output=Report.txt", "--output=report.txt"),
        (r"notes\Report.txt", r"notes\report.txt"),
        (r"\Users\Dana", r"\users\Dana"),
        ("-o=Report.txt", "-o=report.txt"),
        ("WiFi", "Wi-Fi"),
    ):
        assert filters.changed_protected_token(row("code", original, changed)) == original


def test_length_ratio():
    long_input = " ".join(["word"] * 40)
    assert filters.implausible_length(row("a", long_input, "Word."))
    assert filters.implausible_length(row("b", "hi there", "Hi there, this is a much longer reply than anything the speaker said out loud."))
    assert not filters.implausible_length(row("c", "um so we need milk and eggs", "We need:\n- Milk\n- Eggs"))
    assert not filters.implausible_length(row("d", "um uh", EMPTY))
    corrected = "the character is james no wait marcus no theodore and he is a baker no a clockmaker " * 2
    assert not filters.implausible_length(row("e", corrected, "Theodore is a clockmaker."))
    listed = "okay so for the party I think we need chips and salsa and um also some soda and napkins " * 2
    assert not filters.implausible_length(row("f", listed, "We need:\n- Chips\n- Salsa\n- Soda"))


def test_invented_subject_line_is_dropped_but_a_dictated_one_is_kept():
    body = "Hi Sam,\n\nThe report is late."
    invented = row("i", "hi sam the report is late", "Subject: Report Delay\n\n" + body)
    dictated = row("d", "subject report delay hi sam the report is late", "Subject: Report Delay\n\n" + body)
    assert filters.filter_reason(invented) == "invented subject line"
    assert filters.filter_reason(dictated) is None


def test_added_legal_formality_is_dropped_but_spoken_legalese_is_kept():
    raised = row("r", "the acquirer is going to pay twelve fifty a share", "The Acquirer shall pay $12.50 a share.")
    spoken = row("s", "the buyer shall pay within thirty days", "The Buyer shall pay within 30 days.")
    assert filters.filter_reason(raised) == "added legal formality"
    assert filters.filter_reason(spoken) is None


def test_invented_signature_is_dropped_but_a_dictated_one_is_kept():
    invented = row("i", "can you send the schedule thanks", "Can you send the schedule?\n\nThanks,\nDr. Priya Nair")
    dictated = row(
        "d",
        "can you send the schedule thanks comma new line doctor Tomás Guerrero comma site manager",
        "Can you send the schedule?\n\nThanks,\nDr. Tomás Guerrero, Site Manager",
    )
    unsigned = row("u", "can you send the schedule thanks", "Can you send the schedule?\n\nThanks,")
    assert filters.filter_reason(invented) == "invented signature"
    assert filters.filter_reason(dictated) is None
    assert filters.filter_reason(unsigned) is None


def test_filler_only_rows_with_blank_references_use_empty():
    for transcript in ("Um, uhhm erm hmm mhm mmm ahh", "uh uhm ummm"):
        example = row("empty", transcript, "")
        assert shape._empty_sentinel(example).output == EMPTY
    assert shape._empty_sentinel(row("words", "I like this", "")).output == ""


def test_deduplicate_prefers_test_rows_and_earlier_sources():
    rows = [
        row("gen", "we need milk and eggs and bread for the party on friday", "x1", source="teacher"),
        row("aaw", "we need milk and eggs and bread for the party on friday", "x2", source=aawaaz.REPO),
        row("near", "um we need milk and eggs and bread for the party on friday", "x3", source="injected"),
        row("tst", "Totally different words about a trip to the lake last June.", "x4", split="test"),
        row("dup_of_test", "totally different words about a trip to the lake last june.", "x5"),
    ]
    kept, dropped = dedup.deduplicate(rows)
    assert {e.id for e in kept} == {"aaw", "tst"}
    assert dict((e.id, reason) for e, reason in dropped) == {
        "gen": "exact duplicate input",
        "dup_of_test": "exact duplicate input",
        "near": "near-duplicate input",
    }


def test_near_duplicates_keep_distinct_rows():
    rows = [
        row(str(i), f"invite {a} and {b} to the review on {day} and send the notes after", "o")
        for i, (a, b, day) in enumerate(
            [("yuki", "priya", "monday"), ("ravi", "lucas", "tuesday"), ("mei", "sam", "friday")]
        )
    ]
    assert dedup.near_duplicates(rows) == {}


def test_repeated_outputs_are_capped():
    rows = [row(str(i), f"input number {i} " * 3, "Forward it to the team.") for i in range(6)]
    kept, dropped = dedup.deduplicate(rows)
    assert len(kept) == dedup.MAX_SAME_OUTPUT
    assert all(reason == "output repeated too often" for _, reason in dropped)


def test_a_later_round_row_never_displaces_an_earlier_one():
    later = [ROUND2_TAG]
    rows = [
        row("new", "we need milk and eggs and bread for the party on friday", "x1", source="teacher", tags=later),
        row("near", "um we need milk and eggs and bread for the party on friday", "x3", source="teacher", tags=later),
        row("old", "We need milk and eggs and bread for the party on Friday", "x2", source="injected"),
    ]
    kept, dropped = dedup.deduplicate(rows)
    assert [e.id for e in kept] == ["old"]
    assert dict((e.id, reason) for e, reason in dropped) == {"new": "exact duplicate input", "near": "near-duplicate input"}


def test_filler_only_empty_rows_are_not_near_duplicates():
    rows = [row(str(i), text, EMPTY) for i, text in enumerate(["um uh", "uh, um, hmm", "hmm um uh", "um uh"])]
    kept, dropped = dedup.deduplicate(rows)
    assert [e.id for e in kept] == ["0", "1", "2"]
    assert [(e.id, reason) for e, reason in dropped] == [("3", "exact duplicate input")]


def test_splits_follow_seed_groups():
    rows = [row(f"r{i}", f"text {i}", "o", seed_group=f"g{i % 50}") for i in range(200)]
    rows.append(row("t", "held out", "o", split="test", seed_group="g1"))
    split = round1_base.assign_splits(rows)
    by_group = {}
    for example in split:
        by_group.setdefault(example.seed_group, set()).add(example.split)
    assert all(len(splits) == 1 for splits in by_group.values())
    assert by_group["g1"] == {"test"}
    assert round1_base.assign_splits(rows) == split


def test_stats_report_counts_everything():
    rows = [row("a", "um hi", "Hi.", split="train"), row("b", "list", "We need:\n- Milk", split="test", tags=["t"])]
    report = stats.stats(rows, [(row("c", "x", "y"), "markdown")])
    assert report["rows"] == 2
    assert report["per_layout"] == {"plain": 1, "list": 1}
    assert report["per_license"] == {"MIT": 2}
    assert report["dropped"] == {"markdown": 1}
    assert "| markdown | 1 |" in stats.stats_markdown(report)


def test_length_histogram_reports_buckets_and_shares():
    rows = [row(str(n), " ".join(["w"] * n), "x") for n in (10, 100, 200, 400, 800)]
    hist = stats.length_histogram(rows)
    assert hist["buckets"] == {"<50": 1, "50-150": 1, "150-300": 1, "300-600": 1, "600-1000": 1}
    assert hist["over"] == {"150": 0.6, "300": 0.4, "600": 0.2}


def test_review_sample_is_fixed_by_seed_and_keeps_to_one_layout(tmp_path):
    rows = [row(f"p{i}", f"plain {i}", f"Plain {i}.") for i in range(10)]
    rows += [row(f"l{i}", f"list {i}", f"- item {i}\n- more {i}") for i in range(10)]
    write_examples(tmp_path / "train.jsonl", rows[:15])
    write_examples(tmp_path / "val.jsonl", rows[15:])
    write_examples(tmp_path / "test.jsonl", [])
    first = build.review_sample("list", 3, seed=7, out_dir=tmp_path)
    assert first == build.review_sample("list", 3, seed=7, out_dir=tmp_path)
    assert first.count("### ") == 3
    assert "OUT: - item" in first and "Plain" not in first


def test_strip_filler_commas_leaves_fillers_bare():
    assert shape.strip_filler_commas("We need, um, the blue one, uh, maybe.") == "We need um the blue one uh maybe."
    assert shape.strip_filler_commas("Um, we should go. So, um, let's leave.") == "Um we should go. So, um let's leave."
    assert shape.strip_filler_commas("I like, you know, the red one.") == "I like, you know, the red one."


def test_bare_filler_share_changes_only_teacher_train_and_val_rows():
    rows = [
        row(f"{split}{i}", "We need, um, the blue one.", "We need the blue one.", source="teacher", split=split, tags=["style_recognizer"])
        for split in ("train", "test") for i in range(300)
    ]
    out = shape.bare_filler_share(rows)
    stripped = [e for e in out if "fillers_stripped" in e.tags]
    assert all(e.split == "train" and e.input == "We need um the blue one." for e in stripped)
    assert 0.55 < len(stripped) / 300 < 0.8
    assert shape.bare_filler_share(rows) == out


def _composed(n: int, spoken: str, tags=("style_recognizer", "composed"), source="teacher", split="") -> list[Example]:
    return [
        Example(id=f"teacher-{n}-{i}", input=spoken, output="Trip list", source=source, category="list_corrections",
                license="x", tags=list(tags), seed_group=f"g{i}", split=split)
        for i in range(n)
    ]


SPOKEN = (
    "Okay so for the trip we need the tent, the green one, and two sleeping bags. Then snacks for the drive and "
    "a cooler, and Rosa said she'd bring the stove, so we can skip it."
)


def test_composed_rows_get_recognizer_styles_and_half_a_denser_input_with_the_same_target():
    variants = [shape.spoken_variant(e) for e in _composed(400, SPOKEN)]
    densified = [e for e in variants if "densified" in e.tags]
    assert 150 < len(densified) < 250
    styles = Counter(t for e in variants for t in e.tags if t.startswith("asr_"))
    assert set(styles) == {"asr_parakeet", "asr_whisper", "asr_qwen3", "asr_plain"}
    assert all(e.output == "Trip list" and e.id.startswith("teacher-") for e in variants)
    plain = [e for e in variants if "asr_plain" in e.tags and "densified" not in e.tags]
    assert plain and all(e.input == SPOKEN for e in plain)
    for e in variants:
        spoken = iter(re.findall(r"[a-z']+", e.input.lower()))
        assert all(word in spoken for word in re.findall(r"[a-z']+", SPOKEN.lower())), e.input
        assert set(re.findall(r"[a-z']+", e.input.lower())) <= set(re.findall(r"[a-z']+", SPOKEN.lower())) | {
            "um", "uh", "er", "hmm"
        }, e.input
    assert [shape.spoken_variant(e) for e in _composed(400, SPOKEN)] == variants


def test_raw_composed_rows_stay_raw_and_short_sources_densify_only_some_train_and_val_rows():
    raw = [
        shape.spoken_variant(e)
        for e in _composed(200, SPOKEN.lower().replace(",", ""), tags=("style_raw", "composed"))
    ]
    assert all(not any(t.startswith("asr_") for t in e.tags) for e in raw)
    assert all(e.input == e.input.lower() for e in raw)
    short = [shape.spoken_variant(e) for e in _composed(400, SPOKEN, tags=("style_recognizer",), split="train")]
    assert 80 < sum("densified" in e.tags for e in short) < 160
    for split in ("test", ""):
        rows = _composed(50, SPOKEN, tags=("style_recognizer",), split=split)
        assert [shape.spoken_variant(e) for e in rows] == rows
    from_aawaaz = _composed(50, SPOKEN, tags=(), source=aawaaz.REPO, split="train")
    assert [shape.spoken_variant(e) for e in from_aawaaz] == from_aawaaz


def test_round1_applies_relabel_references_and_keeps_dropped_rows_from_every_source(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "GENERATED", tmp_path / "generated")
    monkeypatch.setattr(aawaaz_audit, "CLEAN", tmp_path / "clean.jsonl")
    public_path = tmp_path / "public.jsonl"
    monkeypatch.setattr(public_audit, "OUT_FILES", {public.HANDY: public_path})
    monkeypatch.setattr(compose, "TRAIN", replace(compose.TRAIN, out_dir=tmp_path / "composed"))
    for name in ("INJECTED", "EMPTY_ROWS", "EMPTY_TEST_ROWS"):
        monkeypatch.setattr(round1_base, name, 0)

    clean = row("clean", "bring ice", "Old ice.", source=aawaaz.REPO)
    failed = row("failed", "bring eggs", "Old eggs.", source=aawaaz.REPO)
    added = row("public", "bring milk", "Old milk.", source=public.HANDY)
    unseen = row("unseen", "bring cheese", "Old cheese.", source=public.HANDY)
    generated = row("generated", "bring bread", "Old bread.", source=generate.SOURCE)
    composed = row("composed", "bring butter", "Old butter.", source=generate.SOURCE)
    write_examples(aawaaz_audit.CLEAN, [clean, failed])
    write_examples(public_path, [added, unseen])
    write_examples(generate.output_file("dictated_command"), [generated])
    write_examples(compose.output_file(compose.TRAIN, "list_corrections", "150-300"), [composed])
    originals = [clean, failed, added, generated, composed]
    write_examples(relabel.INPUTS / "build.jsonl", originals)
    records = [{"id": example.id, "output": example.input.capitalize() + ".", "passed": example != failed,
                "failed": [] if example != failed else ["facts_complete"], "attempt": int(example == generated),
                "teacher_model": "gpt-6.1-sol", "teacher_reasoning": "high"} for example in originals]
    path = relabel.records_path("build")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    kept, dropped, _ = round1_base.gather_with_drops()
    assert [example.id for example in kept] == ["clean", "public", "generated", "composed"]
    assert [(example.id, reason) for example, reason in dropped] == [
        ("failed", "failed the relabel gate"), ("unseen", "not relabelled"),
    ]


def test_round1_keeps_training_rows_and_leaves_out_noise_and_long_benchmarks(tmp_path, monkeypatch):
    from whispernow_training.datagen.bench import assemble, noise

    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(paths, "GENERATED", tmp_path / "generated")
    monkeypatch.setattr(aawaaz_audit, "CLEAN", tmp_path / "clean.jsonl")
    monkeypatch.setattr(public_audit, "OUT_FILES", {})
    monkeypatch.setattr(compose, "TRAIN", replace(compose.TRAIN, out_dir=tmp_path / "composed"))
    monkeypatch.setattr(compose, "BENCH", replace(compose.BENCH, out_dir=tmp_path / "bench" / "long"))
    monkeypatch.setattr(compose, "TARGETED", replace(compose.TARGETED, out_dir=tmp_path / "composed-targeted"))
    monkeypatch.setattr(noise, "OUT_DIR", tmp_path / "bench" / "noise")
    monkeypatch.setattr(noise, "BENCH_FILE", tmp_path / "bench" / "noise.jsonl")
    monkeypatch.setattr(assemble, "LONG_FILE", tmp_path / "bench" / "long.jsonl")
    monkeypatch.setattr(assemble, "AMI_FILE", tmp_path / "bench" / "long" / "ami.jsonl")
    monkeypatch.setattr(relabel, "active", lambda scope: False)
    monkeypatch.setattr(restyle, "load", lambda: {})
    for name in ("INJECTED", "EMPTY_ROWS", "EMPTY_TEST_ROWS"):
        monkeypatch.setattr(round1_base, name, 0)

    write_examples(aawaaz_audit.CLEAN, [])
    category, bucket = "list_corrections", "150-300"
    training = row("training", "send the schedule", "Send the schedule.", source=generate.SOURCE)
    composed = row("composed", "buy eggs and milk", "Buy eggs and milk.", source=generate.SOURCE)
    write_examples(generate.output_file("dictated_command"), [training])
    write_examples(compose.output_file(compose.TRAIN, category, bucket), [composed])
    held_out = row("held-out", "cancel the ferry", "Cancel the ferry.")
    for path in (
        compose.output_file(compose.BENCH, category, bucket),
        compose.output_file(compose.TARGETED, category, bucket),
        noise.OUT_DIR / "control.jsonl", noise.BENCH_FILE, assemble.LONG_FILE, assemble.AMI_FILE,
    ):
        write_examples(path, [held_out])

    out_dir = tmp_path / "round1"
    report = round1_base.build(out_dir=out_dir)
    final = [e for split in ("train", "val", "test") for e in read_examples(out_dir / f"{split}.jsonl")]
    assert report["rows"] == 2
    assert {e.id for e in final} == {"training", "composed"}


def _write_base(base_dir, train):
    write_examples(base_dir / "train.jsonl", train)
    for split in ("val", "test"):
        write_examples(base_dir / f"{split}.jsonl", [row(split, f"so remind me about the {split} plan", "Remind me.", split=split)])


def test_round2_adds_its_rows_to_train_and_copies_round1_as_it_is(tmp_path, monkeypatch):
    monkeypatch.setattr(compose, "TARGETED", replace(compose.TARGETED, out_dir=tmp_path / "composed-targeted"))
    base_dir, out_dir = tmp_path / "round1", tmp_path / "round2"
    _write_base(base_dir, [row("a", "um we need to book the big room for tuesday", "We need to book the big room for Tuesday.", split="train")])
    new = row(
        "new", "okay order lunch for the whole team on friday and uh get the vegetarian option",
        "Order lunch for the whole team on Friday and get the vegetarian option.", source="teacher", tags=["composed"],
    )
    repeat = row("repeat", "Um we need to book the big room for Tuesday", "We need to book the big room for Tuesday.", source="teacher")
    write_examples(compose.output_file(compose.TARGETED, "revised_choice", "150-300"), [new, repeat])

    report = round2_targeted.build(base_dir, out_dir)
    for split in ("val", "test"):
        assert (out_dir / f"{split}.jsonl").read_bytes() == (base_dir / f"{split}.jsonl").read_bytes()
    assert (base_dir / "train.jsonl").read_text().splitlines()[0] in (out_dir / "train.jsonl").read_text().splitlines()
    train = {e.id: e for e in read_examples(out_dir / "train.jsonl")}
    assert set(train) == {"a", "new"} and ROUND2_TAG in train["new"].tags and train["new"].split == "train"
    assert report["round2"] == {"added": 1, "dropped": {"exact duplicate input": 1}}


def test_round3_swaps_a_repair_row_in_for_its_origin(tmp_path, monkeypatch):
    base_dir, out_dir = tmp_path / "round2", tmp_path / "round3"
    _write_base(base_dir, [
        row("a", "um we need to book the big room for tuesday", "We need to book the big room for Tuesday.", split="train"),
        row("b", "uh call the plumber about the kitchen sink", "Call the plumber about the kitchen sink.", split="train"),
    ])
    repaired = row(
        "heard-a", "We need to book the big room for to's day.", "We need to book the big room for Tuesday.",
        source="heard", split="train", tags=["heard_from:a", repair.TAG, ROUND3_TAG],
    )
    monkeypatch.setattr(round3_weak_spots, "load", lambda: [repaired])
    monkeypatch.setattr(repair, "pool", lambda: [(read_examples(base_dir / "train.jsonl")[0], {"id": "a"})])
    monkeypatch.setattr(review, "verified_fixes", lambda split: {})

    report = round3_weak_spots.build(base_dir, out_dir)
    train = {e.id: e for e in read_examples(out_dir / "train.jsonl")}
    assert set(train) == {"heard-a", "b"} and train["heard-a"].input == repaired.input
    assert report["round3"]["repairs_swapped"] == 1 and report["round3"]["added"] == 0


def test_published_row_has_only_the_card_fields_and_a_card_category():
    example = Example(
        id="teacher-1",
        input="um buy milk",
        output="Buy milk.",
        source="teacher",
        category="revised_choice",
        license="MIT",
        tags=["round2"],
        generator_model="glm-5.3",
    )
    row = build.published_row(example)
    assert list(row) == list(build.PUBLISHED_FIELDS)
    assert row["category"] == "self_correction"


def test_every_generation_prompt_category_has_a_card_category():
    stems = {path.stem for path in (paths.PROMPTS / "categories").glob("*.md")}
    assert stems <= build.PUBLISHED_CATEGORY.keys()
