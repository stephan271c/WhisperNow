import json
import sys

from whispernow_training.datagen.schema import Example
from whispernow_training.model import chunking, cli, evaluate


def _row(id_, transcript):
    return Example(id=id_, input=transcript, output=transcript, source="s", category="c", license="x")


def _sentence(words, opener="We"):
    return " ".join([opener, *["word"] * (words - 2), "end."])


def test_short_transcripts_stay_whole():
    assert chunking.split_transcript("  One two three.  ", 3) == ["One two three."]


def test_cuts_at_sentence_ends_into_even_pieces():
    text = " ".join([_sentence(4), _sentence(6), _sentence(4), _sentence(6)])
    pieces = chunking.split_transcript(text, 12)
    assert pieces == [" ".join([_sentence(4), _sentence(6)]), " ".join([_sentence(4), _sentence(6)])]


def test_pieces_keep_every_word_in_order_within_the_bounds():
    text = " ".join(_sentence(words) for words in (7, 3, 9, 5, 11, 4, 6, 8, 2, 10, 3, 7))
    pieces = chunking.split_transcript(text, 20)
    assert " ".join(pieces).split() == text.split()
    assert all(10 <= len(piece.split()) <= 20 for piece in pieces)
    assert all(piece.endswith("end.") for piece in pieces)


def test_a_correction_stays_with_what_it_corrects():
    # Two pieces would need a cut within 20 words of "Actually", so the split takes a third.
    correction = "Meet on Tuesday. Actually make it Wednesday."
    text = " ".join([_sentence(20), correction, _sentence(20), _sentence(20)])
    assert chunking.split_transcript(text, 40) == [_sentence(20), f"{correction} {_sentence(20)}", _sentence(20)]


def test_unpunctuated_speech_is_cut_before_a_new_thought():
    text = "we need the remote to be small and cheap so the case should be plastic um not metal because of cost"
    first, second = chunking.split_transcript(text, 12)
    assert first == "we need the remote to be small and cheap"
    assert second.startswith("so the case")


def test_rows_over_the_limit_are_cleaned_in_pieces_and_joined():
    long = _row("long", " ".join([_sentence(5), _sentence(5)]))
    short = _row("short", "Just this.")
    requests = []

    def produce(rows):
        requests.extend((row.id, row.input) for row in rows)
        return [{"id": row.id, "output": f"Clean {row.input}", "hit_limit": row.id == "long#1", "tokens": 3}
                for row in reversed(rows)]

    outputs = chunking.chunked(produce, 6)([long, short])
    assert requests == [("long#0", _sentence(5)), ("long#1", _sentence(5)), ("short", "Just this.")]
    joined, whole = outputs
    assert joined["id"] == "long" and joined["output"] == f"Clean {_sentence(5)}\n\nClean {_sentence(5)}"
    assert joined["hit_limit"] and joined["tokens"] == 6
    assert [piece["input"] for piece in joined["pieces"]] == [_sentence(5), _sentence(5)]
    assert whole == {"id": "short", "output": "Clean Just this.", "hit_limit": False, "tokens": 3}


def test_eval_chunk_words_writes_its_own_directory(tmp_path, monkeypatch):
    rows = [_row("long", " ".join([_sentence(5), _sentence(5)]))]
    ids = tmp_path / "ids.txt"
    ids.write_text("")
    monkeypatch.setattr(evaluate, "TEST_IDS_109", ids)
    monkeypatch.setattr(evaluate, "load_rows", lambda split, limit: rows)
    monkeypatch.setattr(cli, "RUNS", tmp_path)
    monkeypatch.setattr(evaluate, "server_generation", lambda url, *args: {"source": "server", "revision": "q8"})
    requests = []

    async def generate_server(url, pending, *args):
        requests.append([row.input for row in pending])
        return [{"id": row.id, "output": row.input, "hit_limit": False, "tokens": 1} for row in pending]

    monkeypatch.setattr(evaluate, "generate_server", generate_server)
    command = ["cleanup", "eval", "run", "--server", "http://127.0.0.1:8080/v1", "--label", "q8"]
    for extra in ([], ["--chunk-words", "6"]):
        monkeypatch.setattr(sys, "argv", command + extra)
        cli.main()

    assert requests == [[rows[0].input], [_sentence(5), _sentence(5)]]
    whole = json.loads((tmp_path / "run" / "eval-q8" / "test.outputs.jsonl").read_text())
    pieces = json.loads((tmp_path / "run" / "eval-q8-chunk6" / "test.outputs.jsonl").read_text())
    assert "pieces" not in whole and whole["output"] == rows[0].input
    assert pieces["output"] == f"{_sentence(5)}\n\n{_sentence(5)}" and whole["request_key"] != pieces["request_key"]
