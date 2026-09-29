from whispernow_training.datagen import build
from whispernow_training.datagen.schema import EMPTY, Example, write_examples


def row(id_, input_, output, source="owner/src", split="", seed_group="", tags=None):
    return Example(
        id=id_, input=input_, output=output, source=source, category="c", license="MIT",
        split=split, seed_group=seed_group, tags=tags or [],
    )


def test_markdown_is_caught_but_content_asterisks_are_not():
    assert build.has_markdown("This is **bold** text.")
    assert build.has_markdown("Run `ls` now.")
    assert build.has_markdown("## Heading\nText")
    assert not build.has_markdown("Use the 6-31G** basis set and x**2.")
    assert not build.has_markdown("Edit __init__.py and #general.")


def test_malformed_lists():
    assert build.malformed_list("We need:\n* Milk") == "bullet other than '- '"
    assert build.malformed_list("Steps:\n1. Build.\n3. Ship.") == "numbering skips"
    assert build.malformed_list("We need:\n- \n- Eggs") == "empty list item"
    assert build.malformed_list("Steps:\n1. Build.\n2. Ship.\n\nThen:\n1. Rest.") is None
    assert build.malformed_list("Plain text.") is None


def test_answered_outputs():
    assert build.answered(row("a", "what's the capital of France", "Sure! The capital of France is Paris."))
    assert not build.answered(row("b", "sure I can make it at 3", "Sure, I can make it at 3."))
    assert not build.answered(row("c", "um here's a thought about news", "Here's a thought about news."))
    assert not build.answered(row("d", "yeah so I can't connect to the vpn", "I can't connect to the VPN."))


def test_changed_protected_tokens():
    assert build.changed_protected_token(row("a", "open readme.md and main_test.py", "Open README.md and main_test.py.")) == "readme.md"
    assert build.changed_protected_token(row("b", "the x25 protocol", "The X.25 protocol.")) == "x25"
    assert build.changed_protected_token(row("c", "send to a@b.com no wait c@d.com", "Send to c@d.com.")) is None
    assert build.changed_protected_token(row("d", "run npm install --save-dev", "Run npm install --save-dev.")) is None
    assert build.changed_protected_token(row("e", "co2 levels in q2 on k8s", "CO2 levels in Q2 on K8s.")) is None
    assert build.changed_protected_token(row("f", "the PD-L1— marker", "The PD-L1 marker.")) is None


def test_length_ratio():
    long_input = " ".join(["word"] * 40)
    assert build.implausible_length(row("a", long_input, "Word."))
    assert build.implausible_length(row("b", "hi there", "Hi there, this is a much longer reply than anything the speaker said out loud."))
    assert not build.implausible_length(row("c", "um so we need milk and eggs", "We need:\n- Milk\n- Eggs"))
    assert not build.implausible_length(row("d", "um uh", EMPTY))
    corrected = "the character is james no wait marcus no theodore and he is a baker no a clockmaker " * 2
    assert not build.implausible_length(row("e", corrected, "Theodore is a clockmaker."))
    listed = "okay so for the party I think we need chips and salsa and um also some soda and napkins " * 2
    assert not build.implausible_length(row("f", listed, "We need:\n- Chips\n- Salsa\n- Soda"))


def test_deduplicate_prefers_test_rows_and_earlier_sources():
    rows = [
        row("gen", "we need milk and eggs and bread for the party on friday", "x1", source="teacher"),
        row("aaw", "we need milk and eggs and bread for the party on friday", "x2", source=build.aawaaz.REPO),
        row("near", "um we need milk and eggs and bread for the party on friday", "x3", source="injected"),
        row("tst", "Totally different words about a trip to the lake last June.", "x4", split="test"),
        row("dup_of_test", "totally different words about a trip to the lake last june.", "x5"),
    ]
    kept, dropped = build.deduplicate(rows)
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
    assert build.near_duplicates(rows) == {}


def test_repeated_outputs_are_capped():
    rows = [row(str(i), f"input number {i} " * 3, "Forward it to the team.") for i in range(6)]
    kept, dropped = build.deduplicate(rows)
    assert len(kept) == build.MAX_SAME_OUTPUT
    assert all(reason == "output repeated too often" for _, reason in dropped)


def test_splits_follow_seed_groups():
    rows = [row(f"r{i}", f"text {i}", "o", seed_group=f"g{i % 50}") for i in range(200)]
    rows.append(row("t", "held out", "o", split="test", seed_group="g1"))
    split = build.assign_splits(rows)
    by_group = {}
    for example in split:
        by_group.setdefault(example.seed_group, set()).add(example.split)
    assert all(len(splits) == 1 for splits in by_group.values())
    assert by_group["g1"] == {"test"}
    assert build.assign_splits(rows) == split


def test_stats_report_counts_everything():
    rows = [row("a", "um hi", "Hi.", split="train"), row("b", "list", "We need:\n- Milk", split="test", tags=["t"])]
    report = build.stats(rows, [(row("c", "x", "y"), "markdown")])
    assert report["rows"] == 2
    assert report["per_layout"] == {"plain": 1, "list": 1}
    assert report["per_license"] == {"MIT": 2}
    assert report["dropped"] == {"markdown": 1}
    assert "| markdown | 1 |" in build.stats_markdown(report)


def test_filler_only_empty_rows_are_not_near_duplicates():
    rows = [row(str(i), text, EMPTY) for i, text in enumerate(["um uh", "uh, um, hmm", "hmm um uh", "um uh"])]
    kept, dropped = build.deduplicate(rows)
    assert [e.id for e in kept] == ["0", "1", "2"]
    assert [(e.id, reason) for e, reason in dropped] == [("3", "exact duplicate input")]


def test_invented_subject_line_is_dropped_but_a_dictated_one_is_kept():
    body = "Hi Sam,\n\nThe report is late."
    invented = row("i", "hi sam the report is late", "Subject: Report Delay\n\n" + body)
    dictated = row("d", "subject report delay hi sam the report is late", "Subject: Report Delay\n\n" + body)
    assert build.filter_reason(invented) == "invented subject line"
    assert build.filter_reason(dictated) is None


def test_added_legal_formality_is_dropped_but_spoken_legalese_is_kept():
    raised = row("r", "the acquirer is going to pay twelve fifty a share", "The Acquirer shall pay $12.50 a share.")
    spoken = row("s", "the buyer shall pay within thirty days", "The Buyer shall pay within 30 days.")
    assert build.filter_reason(raised) == "added legal formality"
    assert build.filter_reason(spoken) is None


def test_invented_signature_is_dropped_but_a_dictated_one_is_kept():
    invented = row("i", "can you send the schedule thanks", "Can you send the schedule?\n\nThanks,\nDr. Priya Nair")
    dictated = row(
        "d",
        "can you send the schedule thanks comma new line doctor Tomás Guerrero comma site manager",
        "Can you send the schedule?\n\nThanks,\nDr. Tomás Guerrero, Site Manager",
    )
    unsigned = row("u", "can you send the schedule thanks", "Can you send the schedule?\n\nThanks,")
    assert build.filter_reason(invented) == "invented signature"
    assert build.filter_reason(dictated) is None
    assert build.filter_reason(unsigned) is None


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
