"""Hear what Parakeet writes: speak dataset inputs with Kokoro, transcribe them as the app does.

Each sampled input is spoken by a random English Kokoro voice in chunks with pauses between
them, then segmented and decoded the way WhisperNow does: the same Silero VAD asset and
segmentation constants (`whispernow-core/src/workflow/segmentation.rs`), Parakeet TDT 0.6B v2
through sherpa-onnx 1.13.6 with greedy search on the CPU, and the same segment join
(`workflow/join.rs`). The results show Parakeet's punctuation, casing, number and filler habits
so the dataset's punctuated inputs can match them.

Synthetic speech is flatter than a person: Kokoro reads unpunctuated text with little prosody,
and every chunk ends with a falling tone. `chunk_ends` records where the chunks end so the
report can tell Parakeet's own sentence breaks from ones the chunking caused.
"""

from __future__ import annotations

import json
import random
import re
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import paths
from .schema import EMPTY, read_examples

APP_MODELS = Path.home() / ".local/share/WhisperNow/models"
PARAKEET_DIR = APP_MODELS / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16"
# From the k2-fsa/sherpa-onnx `asr-models` release, unpacked in `.cache/asr/`. The app offers
# the Whisper model; Qwen3-ASR is not in the app yet.
WHISPER_DIR = paths.TRAINING_ROOT / ".cache/asr/sherpa-onnx-whisper-distil-large-v3.5"
QWEN3_DIR = paths.TRAINING_ROOT / ".cache/asr/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
MODELS = ("parakeet", "whisper", "qwen3")
VAD_MODEL = APP_MODELS / "silero_vad.onnx"
KOKORO_DIR = paths.TRAINING_ROOT / ".cache/tts/kokoro-int8-multi-lang-v1_0"
AUDIO_DIR = paths.TRAINING_ROOT / ".cache/probe_audio"
OUT_DIR = paths.DATA / "probe"

SAMPLE_RATE = 16_000
# kokoro-multi-lang-v1_0 speaker ids 0-27 are the American and British English voices.
ENGLISH_VOICES = range(28)
THREADS = 5
# Kokoro reads at most about 510 phoneme tokens at once; 45 words stays well inside that.
MAX_CHUNK_WORDS = 45
# Share of rows per source; aawaaz is sampled with its long inputs over-represented.
SOURCE_SHARES = {
    "shantanugoel/aawaaz-transcript-cleanup-dataset": 0.55,
    "teacher": 0.2,
    "MagicNoThief/handy-dictation-editing": 0.15,
    "google-research-datasets/disfl_qa": 0.1,
}


# The app's segmenter and join, ported from whispernow-core.
def _ms(milliseconds: int) -> int:
    return milliseconds * SAMPLE_RATE // 1000


FRAME = 512
END_SILENCE = _ms(600)
PADDING = _ms(200)
MIN_SEGMENT_SPEECH = _ms(1000)
MIN_KEPT_SPEECH = _ms(250)
MAX_MERGE_GAP = _ms(3000)
MAX_SEGMENT = _ms(25_000)
CUT_SEARCH = _ms(2000)


@dataclass
class _Open:
    start: int
    speech_end: int
    speech_samples: int = 0
    held: bool = False
    recent: deque = field(default_factory=deque)  # (start, len, energy, speech)


def segment(samples: np.ndarray, is_speech) -> list[tuple[int, int]]:
    """`SpeechSegmenter` over a whole recording: (start, end) sample ranges to decode."""
    segments: list[tuple[int, int]] = []
    position = 0
    emitted_end = 0
    open_: _Open | None = None

    def emit(start: int, end: int, speech_samples: int) -> None:
        nonlocal emitted_end
        emitted_end = end
        if speech_samples >= MIN_KEPT_SPEECH:
            segments.append((start, end))

    def close(o: _Open) -> None:
        emit(o.start, min(o.speech_end + PADDING, position), o.speech_samples)

    for frame_start in range(0, len(samples), FRAME):
        frame = samples[frame_start : frame_start + FRAME]
        length = len(frame)
        if length < FRAME:
            frame = np.pad(frame, (0, FRAME - length))
        start, end = position, position + length
        position = end
        speech = is_speech(frame)
        if open_ is None:
            if not speech:
                continue
            open_ = _Open(start=max(start - PADDING, emitted_end, 0), speech_end=start)
        if speech:
            open_.speech_end = end
            open_.speech_samples += length
            open_.held = False
        open_.recent.append((start, length, float(np.mean(frame[:length] ** 2)), speech))
        while open_.recent and open_.recent[0][0] < end - CUT_SEARCH:
            open_.recent.popleft()
        pause = end - open_.speech_end
        if not speech and not open_.held and pause >= END_SILENCE:
            if open_.speech_samples >= MIN_SEGMENT_SPEECH:
                close(open_)
                open_ = None
                continue
            open_.held = True
        if open_.held and pause >= MAX_MERGE_GAP:
            close(open_)
            open_ = None
            continue
        if end - open_.start >= MAX_SEGMENT:
            candidates = [f for f in reversed(open_.recent) if f[0] > open_.start]
            cut = min(candidates, key=lambda f: f[2])[0] if candidates else position
            remainder = sum(f[1] for f in open_.recent if f[3] and f[0] >= cut)
            emit(open_.start, cut, open_.speech_samples - remainder)
            if cut < position:
                open_.recent = deque(f for f in open_.recent if f[0] >= cut)
                open_.start = cut
                open_.speech_end = max(open_.speech_end, cut)
                open_.speech_samples = remainder
            else:
                open_ = None
    if open_ is not None:
        close(open_)
    return segments


_SENTENCE_ENDINGS = (".", "!", "?", "…")
_CLOSING = "\"')]”’»"


def _keeps_capital(text: str) -> bool:
    word = re.sub(r"[^\w]+$", "", text.split()[0]) if text.split() else ""
    if word == "I" or word.startswith(("I'", "I’")):
        return True
    letters = [c for c in word if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def join_segment_texts(texts) -> str:
    """`join_segment_texts`: a segment that continues a sentence starts in lower case."""
    joined = ""
    for text in (t.strip() for t in texts):
        if not text:
            continue
        if not joined:
            joined = text
            continue
        continues = not joined.rstrip(_CLOSING).endswith(_SENTENCE_ENDINGS)
        joined += " " + (text[0].lower() + text[1:] if continues and not _keeps_capital(text) else text)
    return joined


def chunks(text: str, rng: random.Random) -> list[str]:
    """Pieces Kokoro reads in one breath: sentences when the text has them, else runs of 20-45 words."""
    pieces = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]
    out: list[str] = []
    for piece in pieces:
        words = piece.split()
        while words:
            take = len(words) if len(words) <= MAX_CHUNK_WORDS else rng.randint(20, MAX_CHUNK_WORDS)
            out.append(" ".join(words[:take]))
            words = words[take:]
    return out


def pause_seconds(rng: random.Random) -> float:
    """Mostly short breaths; some pauses long enough for the app to end a segment."""
    return rng.uniform(0.15, 0.45) if rng.random() < 0.7 else rng.uniform(0.65, 1.4)


def resample(samples: np.ndarray, rate: int) -> np.ndarray:
    """Band-limited resampling to 16 kHz by truncating the spectrum."""
    if rate == SAMPLE_RATE:
        return samples
    count = round(len(samples) * SAMPLE_RATE / rate)
    spectrum = np.fft.rfft(samples)[: count // 2 + 1]
    return (np.fft.irfft(spectrum, count) * count / len(samples)).astype(np.float32)


@dataclass
class Spoken:
    samples: np.ndarray
    voice: int
    speed: float
    chunks: list[str]
    pauses: list[float]


class Speaker:
    def __init__(self) -> None:
        import sherpa_onnx as so

        d = KOKORO_DIR
        config = so.OfflineTtsConfig(
            model=so.OfflineTtsModelConfig(
                kokoro=so.OfflineTtsKokoroModelConfig(
                    model=str(d / "model.int8.onnx"),
                    voices=str(d / "voices.bin"),
                    tokens=str(d / "tokens.txt"),
                    data_dir=str(d / "espeak-ng-data"),
                    dict_dir=str(d / "dict"),
                    lexicon=str(d / "lexicon-us-en.txt"),
                ),
                num_threads=THREADS,
                provider="cpu",
            )
        )
        self.tts = so.OfflineTts(config)

    def speak(self, text: str, rng: random.Random) -> Spoken:
        voice = rng.choice(ENGLISH_VOICES)
        speed = round(rng.uniform(0.95, 1.15), 2)
        pieces = chunks(text, rng)
        parts, pauses = [], []
        for index, piece in enumerate(pieces):
            audio = self.tts.generate(piece, sid=voice, speed=speed)
            parts.append(resample(np.asarray(audio.samples, dtype=np.float32), audio.sample_rate))
            if index < len(pieces) - 1:
                pauses.append(round(pause_seconds(rng), 2))
                parts.append(np.zeros(int(pauses[-1] * SAMPLE_RATE), dtype=np.float32))
        lead = np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32)
        return Spoken(np.concatenate([lead, *parts, lead]), voice, speed, pieces, pauses)


class Recognizer:
    def __init__(self, model: str = "parakeet") -> None:
        import sherpa_onnx as so

        if model == "parakeet":
            self.recognizer = so.OfflineRecognizer.from_transducer(
                encoder=str(PARAKEET_DIR / "encoder.fp16.onnx"),
                decoder=str(PARAKEET_DIR / "decoder.fp16.onnx"),
                joiner=str(PARAKEET_DIR / "joiner.fp16.onnx"),
                tokens=str(PARAKEET_DIR / "tokens.txt"),
                num_threads=THREADS,
                sample_rate=SAMPLE_RATE,
                feature_dim=80,
                decoding_method="greedy_search",
                model_type="nemo_transducer",
                provider="cpu",
            )
        elif model == "whisper":
            # The app's `offline_whisper_config`: automatic language, no tail padding, token timestamps.
            self.recognizer = so.OfflineRecognizer.from_whisper(
                encoder=str(WHISPER_DIR / "distil-large-v3.5-encoder.int8.onnx"),
                decoder=str(WHISPER_DIR / "distil-large-v3.5-decoder.int8.onnx"),
                tokens=str(WHISPER_DIR / "distil-large-v3.5-tokens.txt"),
                language="",
                task="transcribe",
                num_threads=THREADS,
                decoding_method="greedy_search",
                tail_paddings=0,
                enable_token_timestamps=True,
            )
        elif model == "qwen3":
            # A 25 s segment is about 325 audio tokens, more than the default 512-token budget allows with the text.
            self.recognizer = so.OfflineRecognizer.from_qwen3_asr(
                conv_frontend=str(QWEN3_DIR / "conv_frontend.onnx"),
                encoder=str(QWEN3_DIR / "encoder.int8.onnx"),
                decoder=str(QWEN3_DIR / "decoder.int8.onnx"),
                tokenizer=str(QWEN3_DIR / "tokenizer"),
                num_threads=THREADS,
                max_total_len=1024,
                max_new_tokens=256,
            )
        else:
            raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
        vad_config = so.VadModelConfig()
        vad_config.silero_vad.model = str(VAD_MODEL)
        vad_config.silero_vad.threshold = 0.5
        vad_config.silero_vad.min_silence_duration = 0.1
        vad_config.silero_vad.min_speech_duration = 0.064
        vad_config.silero_vad.window_size = FRAME
        vad_config.silero_vad.max_speech_duration = 30.0
        vad_config.sample_rate = SAMPLE_RATE
        vad_config.num_threads = 1
        self._so = so
        self._vad_config = vad_config

    def transcribe(self, samples: np.ndarray) -> tuple[str, list[str]]:
        detector = self._so.VoiceActivityDetector(self._vad_config, buffer_size_in_seconds=30)

        def is_speech(frame: np.ndarray) -> bool:
            detector.accept_waveform(frame)
            speech = detector.is_speech_detected()
            while not detector.empty():  # the app's `clear()`: drop the native segments
                detector.pop()
            return speech

        texts = []
        for start, end in segment(samples, is_speech):
            stream = self.recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, samples[start:end])
            self.recognizer.decode_stream(stream)
            texts.append(stream.result.text)
        return join_segment_texts(texts), texts


def sample_rows(n: int, seed: int) -> list:
    rows = [e for split in ("train", "val", "test") for e in read_examples(paths.BUILD / f"{split}.jsonl")]
    rows = [e for e in rows if e.output != EMPTY and e.source in SOURCE_SHARES]
    rng = random.Random(seed)
    picked = []
    for source, share in SOURCE_SHARES.items():
        pool = sorted((e for e in rows if e.source == source), key=lambda e: e.id)
        count = round(n * share)
        if source.endswith("aawaaz-transcript-cleanup-dataset"):
            # Half from the longest quarter, so long dictations are well represented.
            pool.sort(key=lambda e: len(e.input))
            long_pool = pool[3 * len(pool) // 4 :]
            chosen = rng.sample(long_pool, count // 2)
            taken = {e.id for e in chosen}
            rest = [e for e in pool if e.id not in taken]
            chosen += rng.sample(rest, count - count // 2)
        else:
            chosen = rng.sample(pool, count)
        picked += chosen
    rng.shuffle(picked)
    return picked


def _write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(SAMPLE_RATE)
        sink.writeframes(pcm.tobytes())


def _out_files(model: str = "parakeet") -> list[Path]:
    return sorted(OUT_DIR.glob(f"{model}.jsonl")) + sorted(OUT_DIR.glob(f"{model}.*.jsonl"))


def run(n: int, seed: int, keep_audio: bool = True, shard: int = 0, shards: int = 1) -> int:
    """Speaks and transcribes this shard's share of `n` sampled inputs; resumes past rows already done."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for path in _out_files() for line in path.open(encoding="utf-8")}
    todo = [e for index, e in enumerate(sample_rows(n, seed)) if index % shards == shard and e.id not in done]
    speaker, recognizer = Speaker(), Recognizer()
    written = 0
    out = OUT_DIR / (f"parakeet.{shard}.jsonl" if shards > 1 else "parakeet.jsonl")
    with out.open("a", encoding="utf-8") as sink:
        for example in todo:
            rng = random.Random(f"{seed}-{example.id}")
            spoken = speaker.speak(example.input, rng)
            text, segments = recognizer.transcribe(spoken.samples)
            if keep_audio:
                _write_wav(AUDIO_DIR / f"{example.id}.wav", spoken.samples)
            record = {
                "id": example.id,
                "source": example.source,
                "category": example.category,
                "layout": example.layout,
                "tags": example.tags,
                "spoken": example.input,
                "model": "parakeet",
                "heard": text,
                "segments": segments,
                "chunk_ends": [len(" ".join(spoken.chunks[: i + 1]).split()) for i in range(len(spoken.chunks))],
                "pauses": spoken.pauses,
                "voice": spoken.voice,
                "speed": spoken.speed,
                "seconds": round(len(spoken.samples) / SAMPLE_RATE, 1),
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            written += 1
            print(f"shard {shard}: {written}/{len(todo)} {example.id} {record['seconds']}s", flush=True)
    return written


def probe_texts(items: list[tuple[str, str]], out: Path, seed: int) -> int:
    """Speaks and transcribes (id, text) pairs into `out`, one JSON line each; resumes past ids already there."""
    done = {json.loads(line)["id"] for line in out.open(encoding="utf-8")} if out.exists() else set()
    todo = [(id_, text) for id_, text in items if id_ not in done]
    if not todo:
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    speaker, recognizer = Speaker(), Recognizer()
    written = 0
    with out.open("a", encoding="utf-8") as sink:
        for id_, text in todo:
            spoken = speaker.speak(text, random.Random(f"{seed}-{id_}"))
            heard, segments = recognizer.transcribe(spoken.samples)
            record = {
                "id": id_, "spoken": text, "model": "parakeet", "heard": heard, "segments": segments,
                "pauses": spoken.pauses, "voice": spoken.voice, "speed": spoken.speed,
                "seconds": round(len(spoken.samples) / SAMPLE_RATE, 1),
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            written += 1
            print(f"{written}/{len(todo)} {id_} {record['seconds']}s", flush=True)
    return written


def redecode(model: str, shard: int = 0, shards: int = 1) -> int:
    """Transcribes the audio `run` kept with another recognizer, through the same segmenter and join."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for path in _out_files(model) for line in path.open(encoding="utf-8")}
    todo = [r for index, r in enumerate(load()) if index % shards == shard and r["id"] not in done]
    todo = [r for r in todo if (AUDIO_DIR / f"{r['id']}.wav").exists()]
    recognizer = Recognizer(model)
    written = 0
    with (OUT_DIR / (f"{model}.{shard}.jsonl" if shards > 1 else f"{model}.jsonl")).open("a", encoding="utf-8") as sink:
        for record in todo:
            with wave.open(str(AUDIO_DIR / f"{record['id']}.wav")) as audio:
                samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype=np.int16).astype(np.float32) / 32768
            text, segments = recognizer.transcribe(samples)
            sink.write(json.dumps(record | {"model": model, "heard": text, "segments": segments}, ensure_ascii=False) + "\n")
            sink.flush()
            written += 1
            print(f"{model} shard {shard}: {written}/{len(todo)} {record['id']}", flush=True)
    return written


def load(model: str = "parakeet") -> list[dict]:
    """Probe records for `model`; the transcript is `heard` (the first Parakeet rows called it `parakeet`)."""
    records = [json.loads(line) for path in _out_files(model) for line in path.open(encoding="utf-8")]
    for record in records:
        if "parakeet" in record:
            record["heard"] = record.pop("parakeet")
            record["model"] = "parakeet"
    return records


_WORD = re.compile(r"[A-Za-z0-9$%][\w'’.:,/%$-]*")
_FILLERS = {"um", "uh", "erm", "hmm", "mm", "ah", "er"}
_COMMANDS = {
    "period": ".",
    "comma": ",",
    "question mark": "?",
    "exclamation mark": "!",
    "exclamation point": "!",
    "colon": ":",
    "new line": "\n",
    "new paragraph": "\n\n",
    "dot": ".",
    "slash": "/",
    "dash": "-",
    "hyphen": "-",
    "underscore": "_",
    "at sign": "@",
}
_NUMBER_WORDS = set(
    (
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million"
    ).split()
)


def _bare(word: str) -> str:
    return re.sub(r"[^\w']", "", word.casefold().replace("’", "'"))


def _aligned_breaks(record: dict) -> tuple[int, int]:
    """Sentence ends the recognizer wrote at a TTS chunk end, and elsewhere."""
    import difflib

    spoken = [_bare(w) for w in record["spoken"].split()]
    heard = _WORD.findall(record["heard"])
    matcher = difflib.SequenceMatcher(a=spoken, b=[_bare(w) for w in heard], autojunk=False)
    heard_to_spoken = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            heard_to_spoken[block.b + offset] = block.a + offset
    chunk_ends = set(record["chunk_ends"])
    at_chunk = elsewhere = 0
    for index, word in enumerate(heard[:-1]):
        if word.rstrip("\"')”").endswith((".", "?", "!")) and index in heard_to_spoken:
            if heard_to_spoken[index] + 1 in chunk_ends:
                at_chunk += 1
            else:
                elsewhere += 1
    return at_chunk, elsewhere


def word_error_rate(spoken: str, heard: str) -> float:
    import difflib

    a = [_bare(w) for w in spoken.split() if _bare(w)]
    b = [_bare(w) for w in _WORD.findall(heard) if _bare(w)]
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return 1 - matched / max(len(a), 1)


def report(records: list[dict]) -> dict:
    """A recognizer's habits over its probe rows."""
    from collections import Counter

    out: dict = {"rows": len(records), "seconds": round(sum(r["seconds"] for r in records))}
    words = sum(len(_WORD.findall(r["heard"])) for r in records)
    text = " ".join(r["heard"] for r in records)
    out["per_100_words"] = {
        mark: round(100 * text.count(mark) / max(words, 1), 2) for mark in (".", ",", "?", "!", ":", ";", "-", '"')
    }
    at_chunk = elsewhere = 0
    for record in records:
        a, b = _aligned_breaks(record)
        at_chunk += a
        elsewhere += b
    chunk_ends = sum(len(r["chunk_ends"]) - 1 for r in records)
    out["sentence_breaks"] = {"at_tts_chunk_end": at_chunk, "elsewhere": elsewhere, "tts_chunk_ends": chunk_ends}
    sentences = [s for r in records for s in re.split(r"(?<=[.?!])\s+", r["heard"]) if s]
    lengths = sorted(len(s.split()) for s in sentences)
    out["sentence_words"] = {"p25": lengths[len(lengths) // 4], "p50": lengths[len(lengths) // 2], "p75": lengths[3 * len(lengths) // 4]}
    out["sentence_starts_capitalized"] = round(sum(s[:1].isupper() for s in sentences) / len(sentences), 3)

    fillers = Counter()
    for record in records:
        spoken = [_bare(w) for w in record["spoken"].split()]
        fillers["spoken"] += sum(w in _FILLERS for w in spoken)
        for match in re.finditer(r"(^|.)\s*\b(um+|uh+|erm|hmm|mm|ah|er)\b([.,?!]?)", record["heard"], re.IGNORECASE):
            fillers["heard"] += 1
            fillers["comma_before"] += match.group(1) == ","
            fillers["mark_after"] += bool(match.group(3))
            fillers["capitalized"] += match.group(2)[0].isupper()
    out["fillers"] = dict(fillers)

    numbers = Counter()
    for record in records:
        for token in _WORD.findall(record["heard"]):
            bare = _bare(token)
            if any(c.isdigit() for c in token):
                numbers["digits"] += 1
            elif bare in _NUMBER_WORDS:
                numbers["words"] += 1
    out["numbers"] = dict(numbers)

    commands = Counter()
    for record in records:
        spoken = record["spoken"].casefold()
        heard = record["heard"].casefold()
        for phrase in _COMMANDS:
            said = len(re.findall(rf"\b{phrase}\b", spoken))
            if said:
                commands[f"{phrase}: said"] += said
                commands[f"{phrase}: kept as words"] += min(said, len(re.findall(rf"\b{phrase}\b", heard)))
    out["spoken_commands"] = dict(sorted(commands.items()))

    # Words the recognizer respelled the British way ("canceling" heard as "cancelling").
    british = set()
    for record in records:
        spoken = set(re.findall(r"[a-z]+", record["spoken"].casefold()))
        for word in set(re.findall(r"[a-z]+", record["heard"].casefold())) - spoken:
            for gb, us in (("our", "or"), ("ise", "ize"), ("ised", "ized"), ("tre", "ter"), ("lled", "led"), ("lling", "ling"), ("yse", "yze")):
                if word.endswith(gb) and word[: -len(gb)] + us in spoken:
                    british.add(word)
            if word.replace("our", "or", 1) in spoken:
                british.add(word)
    out["british_spellings"] = sorted(british)

    # How the app's segments meet: the constants in `restyle.recognizer_quirks` come from here.
    bounds = [(before, after) for r in records for before, after in zip(r["segments"], r["segments"][1:])]
    out["segments"] = {
        "words_per_boundary": round(words / max(len(bounds), 1), 1),
        "end_with_period": round(sum(b.rstrip().endswith((".", "?", "!")) for b, _ in bounds) / max(len(bounds), 1), 3),
        "start_lower_case": round(sum(a[:1].islower() for _, a in bounds) / max(len(bounds), 1), 3),
    }
    dollars = [m.group(1) for r in records for m in re.finditer(r"(\S?)\$\d", r["heard"])]
    out["dollar_glued"] = f"{sum(bool(c) and c.isalnum() for c in dollars)} of {len(dollars)}"

    punctuated = [r for r in records if len(re.findall(r"[.,?!]", r["spoken"])) > 0.03 * len(r["spoken"].split())]
    plain = [r for r in records if r not in punctuated]
    out["by_spoken_text"] = {}
    for label, group in (("punctuated", punctuated), ("unpunctuated", plain)):
        group_words = sum(len(_WORD.findall(r["heard"])) for r in group)
        group_text = " ".join(r["heard"] for r in group)
        out["by_spoken_text"][label] = {"rows": len(group)} | {
            mark: round(100 * group_text.count(mark) / max(group_words, 1), 2) for mark in (".", ",", "?")
        }
    kept = {}
    for filler in ("um", "uh", "like", "so"):
        said = sum(len(re.findall(rf"\b{filler}\b", r["spoken"], re.IGNORECASE)) for r in records)
        heard = sum(len(re.findall(rf"\b{filler}\b", r["heard"], re.IGNORECASE)) for r in records)
        kept[filler] = f"{heard} of {said}"
    out["fillers_kept"] = kept
    errors = sorted(word_error_rate(r["spoken"], r["heard"]) for r in records)
    out["word_error_rate"] = {"p50": round(errors[len(errors) // 2], 3), "p90": round(errors[9 * len(errors) // 10], 3), "mean": round(sum(errors) / len(errors), 3)}
    return out
