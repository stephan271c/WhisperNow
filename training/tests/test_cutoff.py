import asyncio
import json
import random
from collections import Counter
from types import SimpleNamespace

from whispernow_training.datagen.speech import probe, cutoff
from whispernow_training.datagen.schema import Example
from whispernow_training.model import evaluate

VOCAB = cutoff.Vocabulary(
    Counter({"default": 40, "settings": 40, "invoice": 40, "person": 40, "per": 400, "set": 400, "not": 5000}),
    Counter({"per person": 5}),
    Counter({"SEC": 24}),
)


def row(id_, input_, output, tags=None):
    return Example(id=id_, input=input_, output=output, source="s", category="c", license="MIT", split="train", tags=tags or [])


def test_splice_takes_the_heard_piece_only_when_the_words_around_it_were_heard_as_said():
    tokens = "Is it the fact that our default settings for SQL are not good?".split()
    at = tokens.index("settings")
    assert cutoff.splice(tokens, at, "Is it the fact that our default set, settings for SQL are not good?") == ["set,", "settings"]
    assert cutoff.splice(tokens, at, "Is it the fact that our the fault set settings for SQL are not good?") is None
    assert cutoff.splice(tokens, at, "Is it the fact that our default set settings four SQL are not good?") is None


def test_a_piece_is_a_stumble_only_when_it_starts_like_the_word_and_is_no_pair_said_on_purpose():
    assert cutoff.stumble_reason(["set,"], "settings", VOCAB) is None
    assert cutoff.stumble_reason(["sett", "um"], "settings", VOCAB) is None
    assert cutoff.stumble_reason(["new"], "number", VOCAB) == "no shared start"
    assert cutoff.stumble_reason(["per,"], "person", VOCAB) == "a pair said on purpose"
    assert cutoff.stumble_reason(["set", "s"], "settings", VOCAB) is None
    assert cutoff.stumble_reason(["not"], "notify", VOCAB) == "a common word"
    assert cutoff.stumble_reason(["nine"], "night", VOCAB) == "a number"
    assert cutoff.stumble_reason(["SEC"], "security", VOCAB) == "an acronym the references use"
    assert cutoff.stumble_reason(["INF"], "influenced", VOCAB) is None
    assert cutoff.stumble_reason(["tab,", "a"], "tablet", VOCAB) == "more than one word"
    assert cutoff.stumble_reason(["me", "a"], "message", VOCAB) == "more than one word"


def _without(text, piece):
    tokens = text.split()
    for i in range(len(tokens)):
        if tokens[i : i + len(piece)] == piece:
            return " ".join(tokens[:i] + tokens[i + len(piece) :])
    raise AssertionError(f"{piece} not in {text!r}")


def test_a_written_piece_goes_outside_quotes_and_the_row_keeps_every_word(monkeypatch):
    monkeypatch.setattr(cutoff, "TTS_SHARES", {"written": 1})
    example = row(
        "r1", 'tell priya the default settings are "settings as written" for the invoice',
        'Tell Priya the default settings are "settings as written" for the invoice.', tags=["style_raw"],
    )
    record = cutoff.cut_row(example, None, None, VOCAB)
    assert record["kept"] and record["tts"] == "written" and len(record["cuts"]) == 1
    cut = record["cuts"][0]
    assert cut["word"] in {"default", "settings", "invoice"} and "".join(cut["piece"])[:2] == cut["word"][:2]
    assert _without(record["input"], cut["piece"]) == example.input
    assert '"settings as written"' in record["input"]


def test_a_recognizer_row_takes_what_parakeet_heard(monkeypatch):
    monkeypatch.setattr(cutoff, "TTS_SHARES", {"phonon": 1})
    example = row("p1", "Can you check the invoice before Friday?", "Can you check the invoice before Friday?")
    speaker = SimpleNamespace(speak=lambda text, rng: SimpleNamespace(samples=text, voice="Zoey"))
    recognizer = SimpleNamespace(transcribe=lambda samples: (samples, []))
    record = cutoff.cut_row(example, {"phonon": speaker}, recognizer, VOCAB)
    assert record["kept"] and record["voice"] == "Zoey" and record["tts"] == "phonon"
    [cut] = record["cuts"]
    assert cut["word"] == "invoice" and cut["piece"][0] == cut["prefix"] + ","
    assert _without(record["input"], cut["piece"]) == example.input


def test_the_cut_goes_into_the_input_as_written(monkeypatch):
    monkeypatch.setattr(cutoff, "TTS_SHARES", {"written": 1})
    example = row("n1", "Notes for Friday:\n\ncheck  the invoice", "Notes for Friday:\n\nCheck the invoice.")
    record = cutoff.cut_row(example, None, None, VOCAB)
    piece = " ".join(record["cuts"][0]["piece"])
    assert record["input"] == f"Notes for Friday:\n\ncheck  the {piece} invoice"


def test_a_raw_row_takes_parakeets_piece_lowercased_and_unpunctuated(monkeypatch):
    monkeypatch.setattr(cutoff, "TTS_SHARES", {"kokoro": 1})
    example = row("w1", "can you check the invoice before friday", "Can you check the invoice before Friday?", tags=["style_raw"])
    speaker = SimpleNamespace(speak=lambda text, rng: SimpleNamespace(samples=text, voice=3))
    recognizer = SimpleNamespace(transcribe=lambda samples: (samples.replace("inv,", "Inv,"), []))
    record = cutoff.cut_row(example, {"kokoro": speaker}, recognizer, VOCAB)
    assert record["kept"] and record["input"] == example.input.replace("invoice", f"{record['cuts'][0]['prefix']} invoice")


def test_score_counts_pieces_left_words_lost_and_look_alike_pairs_kept(tmp_path, monkeypatch):
    cuts = [
        {"id": "c1", "origin": "t1", "tts": "kokoro", "cuts": [{"word": "settings", "prefix": "sett", "piece": ["sett,"]}]},
        {"id": "c2", "origin": "t2", "tts": "written", "cuts": [{"word": "invoice", "prefix": "inv", "piece": ["inv", "uh"]}]},
    ]
    monkeypatch.setattr(cutoff, "BENCH_CUTS", tmp_path / "cuts.jsonl")
    cutoff.BENCH_CUTS.write_text("".join(json.dumps(c) + "\n" for c in cuts), encoding="utf-8")
    test_rows = [row("t1", "we were at the default settings", "We were at the default settings."),
                 row("t2", "send the invoice", "Send the invoice.")]
    monkeypatch.setattr(evaluate, "load_rows", lambda split: test_rows if split == "test" else [])
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    outputs = {
        "test": {"t1": "We were at the default settings.", "t2": "Send the invoice."},
        "bench-cutoff": {"c1": "We were at the default sett, settings.", "c2": "Send the."},
        "bench-long": {}, "bench-noise": {},
    }
    for split, rows in outputs.items():
        (eval_dir / f"{split}.outputs.jsonl").write_text(
            "".join(json.dumps({"id": id_, "output": text}) + "\n" for id_, text in rows.items()), encoding="utf-8"
        )

    line = cutoff.score([eval_dir]).splitlines()[-1]
    assert "| 2 | 2 | 50.0% | 50.0% | 0.0% | 100.0% of 1 |" in line


def _rows(short, mid):
    return [row(f"s{i:02d}", "word " * 10, "Words.") for i in range(short)] + [
        row(f"m{i:02d}", "word " * 80, "Words.") for i in range(mid)
    ]


def test_targets_split_the_rows_over_length_buckets_in_proportion():
    rows = _rows(6, 3) + [row("l", "word " * 200, "Words.")]
    assert cutoff.targets(rows, 10) == {"<50": 6, "50-150": 3, "150-300": 1}
    assert sum(cutoff.targets(rows, 7).values()) == 7


def test_a_short_share_cuts_the_short_bucket_and_the_longer_buckets_split_the_rest():
    rows = _rows(6, 3) + [row("l", "word " * 200, "Words.")]
    assert cutoff.targets(rows, 10, short_share=0.5) == {"<50": 3, "50-150": 5, "150-300": 2}


def test_run_fills_each_length_bucket_to_its_target_past_rows_that_fail(tmp_path, monkeypatch):
    rows = _rows(12, 6)
    tried = []

    def cut_row(example, speakers, recognizer, vocab, seed=cutoff.SEED):
        tried.append(example.id)
        kept = int(example.id[1:]) % 3 != 0
        return {"id": example.id, "tts": "written", "kept": kept, "rejected": {}, "reason": None if kept else "0 of 1 cuts",
                "input": example.input, "cuts": []}

    monkeypatch.setattr(cutoff, "OUT_DIR", tmp_path)
    monkeypatch.setattr(cutoff, "ROWS", 6)
    monkeypatch.setattr(cutoff, "SHORT_SHARE", 1.0)
    monkeypatch.setattr(cutoff, "pool", lambda: rows)
    monkeypatch.setattr(cutoff, "cut_row", cut_row)
    monkeypatch.setattr(cutoff, "_speakers", lambda provider: {})
    monkeypatch.setattr(probe, "Recognizer", lambda *args: None)

    for shard in (0, 1):
        cutoff.run(shard=shard, shards=2)
    loaded = cutoff.load()
    assert Counter(r["id"][0] for r in loaded) == {"s": 4, "m": 2}
    assert [r["id"] for r in loaded] == ["s01", "s02", "s04", "s05", "m01", "m02"]
    assert "s11" not in tried and "m05" not in tried


def test_check_state_shows_the_passage_with_and_without_the_piece():
    state = cutoff.check_state("so call me at nine night? Thanks.", {"word": "night", "prefix": "ni", "piece": ["nine"]})
    assert "TRANSCRIPT:\nso call me at nine night? Thanks." in state and "CLEAN:\nso call me at night? Thanks." in state
    assert state.endswith("PIECE: nine\nNEXT WORD: night")
    assert cutoff.check_state("Stan, standing there.", {"word": "standing", "prefix": "sta", "piece": ["stan,"]}).endswith(
        "PIECE: Stan,\nNEXT WORD: standing"
    )
    lost = {"id": "c", "kept": True, "input": "sta stairs", "cuts": [{"word": "steps", "prefix": "ste", "piece": ["ste"]}]}
    assert cutoff.screen(lost, {}) == "piece not found: 'ste' before 'steps'"


def test_check_asks_each_question_once_per_cut_and_a_flag_fails_the_row(tmp_path, monkeypatch):
    asked = []

    async def decide(client, key, state, question):
        asked.append(question)
        return 0.9 if "PIECE: nine" in state and question == cutoff._check_spec()["questions"]["fact"] else 0.01

    monkeypatch.setattr(cutoff, "CHECK", tmp_path / "check.jsonl")
    monkeypatch.setattr(cutoff.mercury, "decide", decide)
    monkeypatch.setattr(cutoff.mercury, "openrouter_key", lambda: "k")
    bad = {"id": "a", "kept": True, "input": "come back at nine night and sta stairs",
           "cuts": [{"word": "night", "prefix": "ni", "piece": ["nine"]}, {"word": "stairs", "prefix": "sta", "piece": ["sta"]}]}
    good = {"id": "b", "kept": True, "input": "take the sta stairs", "cuts": [{"word": "stairs", "prefix": "sta", "piece": ["sta"]}]}

    _, new = asyncio.run(cutoff.check([bad, good]))
    assert len(asked) == 3 * 3
    seen = cutoff.verdicts()
    assert new == seen
    assert asyncio.run(cutoff.check([bad, good], seen)) == (Counter(), {})
    assert len(asked) == 9
    assert cutoff.screen(bad, seen) == "fact: 'nine' before 'night'" and cutoff.screen(good, seen) is None
    assert cutoff.screen({**good, "input": "take the sta stairs now"}, seen) == "unchecked"


def test_top_up_replaces_a_row_the_check_failed_and_loads_the_models_once(tmp_path, monkeypatch):
    rows = _rows(12, 6)
    loads = Counter()

    def cut_row(example, speakers, recognizer, vocab, seed=cutoff.SEED):
        kept = int(example.id[1:]) % 3 != 0
        return {"id": example.id, "tts": "written", "kept": kept, "rejected": {}, "reason": None if kept else "0 of 1 cuts",
                "input": example.input, "cuts": []}

    monkeypatch.setattr(cutoff, "OUT_DIR", tmp_path)
    monkeypatch.setattr(cutoff, "CHECK", tmp_path / "check.jsonl")
    monkeypatch.setattr(cutoff, "ROWS", 6)
    monkeypatch.setattr(cutoff, "SHORT_SHARE", 1.0)
    monkeypatch.setattr(cutoff, "pool", lambda: rows)
    monkeypatch.setattr(cutoff, "cut_row", cut_row)
    monkeypatch.setattr(cutoff, "_speakers", lambda provider: loads.update(["speakers"]) or {})
    monkeypatch.setattr(probe, "Recognizer", lambda *args: loads.update(["recognizer"]))
    monkeypatch.setattr(cutoff, "vocabulary", lambda rows: loads.update(["vocabulary"]) or VOCAB)
    monkeypatch.setattr(cutoff, "screen", lambda record, seen: "fact: 'x' before 'y'" if record["id"] == "s01" else None)

    async def check(rows=None, seen=None, concurrency=8):
        loads["round"] += 1
        return Counter(), {}

    monkeypatch.setattr(cutoff, "check", check)
    report = cutoff.top_up(shards=2)
    assert [r["id"] for r in cutoff.load()] == ["s02", "s04", "s05", "s07", "m01", "m02"]
    assert report["by length"] == {"<50": "4/4", "50-150": "2/2"}
    # Two rounds cut rows (the second replaces s01); the third finds every bucket full.
    assert loads == {"round": 3, "speakers": 1, "recognizer": 1, "vocabulary": 1}
    cutoff.top_up(shards=2)
    assert loads == {"round": 4, "speakers": 1, "recognizer": 1, "vocabulary": 1}


def test_one_shard_run_skips_rows_another_shard_cut_and_load_takes_each_row_once(tmp_path, monkeypatch):
    rows = _rows(12, 6)
    tried = []

    def cut_row(example, speakers, recognizer, vocab, seed=cutoff.SEED):
        tried.append(example.id)
        return {"id": example.id, "tts": "written", "kept": True, "rejected": {}, "reason": None,
                "input": example.input, "cuts": []}

    monkeypatch.setattr(cutoff, "OUT_DIR", tmp_path)
    monkeypatch.setattr(cutoff, "CHECK", tmp_path / "check.jsonl")
    monkeypatch.setattr(cutoff, "ROWS", 6)
    monkeypatch.setattr(cutoff, "SHORT_SHARE", 1.0)
    monkeypatch.setattr(cutoff, "pool", lambda: rows)
    monkeypatch.setattr(cutoff, "cut_row", cut_row)
    monkeypatch.setattr(cutoff, "_speakers", lambda provider: {})
    monkeypatch.setattr(probe, "Recognizer", lambda *args: None)

    cutoff.run(shard=1, shards=2)
    first = list(tried)
    (tmp_path / "cutoff.0.jsonl").write_text((tmp_path / "cutoff.1.jsonl").read_text().splitlines()[0] + "\n")
    cutoff.run(shard=0, shards=1)
    assert first == ["s01", "s03", "m01"] and not set(tried[3:]) & set(first)
    loaded = [r["id"] for r in cutoff.load()]
    assert len(loaded) == len(set(loaded)) == 6


def test_bench_checks_candidates_in_chunks_and_chooses_the_rows_one_at_a_time_checks_would(tmp_path, monkeypatch):
    test_rows = [row(f"t{i:02d}", "take the sta stairs " * 4, "Take the stairs.") for i in range(40)]
    cut, clients = [], []

    def kept(id_):
        return int(id_[1:]) % 4 != 0

    def flagged(id_):
        return int(id_[1:]) % 3 == 0

    def cut_row(example, speakers, recognizer, vocab, seed=cutoff.SEED):
        cut.append(example.id)
        return {"id": example.id, "tts": "written", "kept": kept(example.id), "rejected": {}, "reason": "0 of 1 cuts",
                "input": f"{example.id} take the sta stairs", "cuts": [{"word": "stairs", "prefix": "sta", "piece": ["sta"]}]}

    async def decide(client, key, state, question):
        clients.append(client)
        return 0.9 if flagged(state.split("TRANSCRIPT:\n")[1].split()[0]) else 0.01

    monkeypatch.setattr(cutoff, "read_examples", lambda path: test_rows if path.name == "test.jsonl" else [])
    monkeypatch.setattr(cutoff, "cut_row", cut_row)
    monkeypatch.setattr(cutoff, "_speakers", lambda provider: {})
    monkeypatch.setattr(probe, "Recognizer", lambda *args: None)
    monkeypatch.setattr(cutoff.mercury, "decide", decide)
    monkeypatch.setattr(cutoff.mercury, "openrouter_key", lambda: "k")
    for name, file in (("CHECK", "check.jsonl"), ("BENCH", "bench.jsonl"), ("BENCH_CUTS", "cuts.jsonl")):
        monkeypatch.setattr(cutoff, name, tmp_path / file)
    monkeypatch.setattr(cutoff, "BENCH_CHUNK", 4)

    assert cutoff.bench(n=6) == 6
    order = sorted(test_rows, key=lambda e: e.id)
    random.Random(cutoff.SEED + 1).shuffle(order)
    order = [e.id for e in order]
    chosen = [json.loads(line)["origin"] for line in cutoff.BENCH_CUTS.read_text().splitlines()]
    assert chosen == [id_ for id_ in order if kept(id_) and not flagged(id_)][:6]
    assert cut == order[: len(cut)]
    checked = [id_ for id_ in cut if kept(id_)]
    assert len({id(client) for client in clients}) == len(checked) / 4
    assert checked.index(chosen[-1]) >= len(checked) - 4
