"""Hear what Parakeet writes: speak dataset inputs with Kokoro (or Phonon), transcribe them as the app does.

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
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import paths
from ...text_rules import FILLER, FILLER_FORMS
from ..schema import EMPTY, read_examples

APP_MODELS = Path.home() / ".local/share/WhisperNow/models"
PARAKEET_DIR = APP_MODELS / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16"
# From the k2-fsa/sherpa-onnx `asr-models` release, unpacked in `.cache/asr/`. The app offers
# the Whisper model; Qwen3-ASR is not in the app yet.
WHISPER_DIR = paths.TRAINING_ROOT / ".cache/asr/sherpa-onnx-whisper-distil-large-v3.5"
QWEN3_DIR = paths.TRAINING_ROOT / ".cache/asr/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
MODELS = ("parakeet", "whisper", "qwen3")
VAD_MODEL = APP_MODELS / "silero_vad.onnx"
KOKORO_DIR = paths.TRAINING_ROOT / ".cache/tts/kokoro-int8-multi-lang-v1_0"
# The int8 Kokoro runs its quantized ops on the CPU even under CUDA (1.7x real time on an RTX 5070);
# the float model runs at about 27x. The voices are the same file.
KOKORO_GPU_DIR = paths.CACHE / "tts" / "kokoro-multi-lang-v1_0"
# Pinned revisions; Apache-2.0 weights and voices.
PHONON_MODEL = "hf://gradium/phonon-31m-en@8c64b55ec301728e2d1695bfea87a08a267d0c2d"
PHONON_VOICES = "hf://gradium/phonon-voices@da6c9ec0d47be8df5c87c730a3aa97187f3229c3"
# Phonon's character voices that are no person dictating; its human characters add old and young voices.
PHONON_NOT_HUMAN = {"Azgorath", "The Thing Under the Stairs", "Vyrmathrax the Ancient", "Grub the Ogre"}
TTS_MODELS = ("kokoro", "phonon")
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
END_SILENCE = _ms(2000)
PADDING = _ms(200)
MIN_SEGMENT_SPEECH = _ms(1000)
MIN_KEPT_SPEECH = _ms(250)
MAX_MERGE_GAP = _ms(3000)


def max_segment(model: str) -> int:
    """`max_segment_samples`: Parakeet's segments run to 45 s, every other recognizer's to 28 s."""
    return _ms(45_000) if model.startswith("parakeet") else _ms(28_000)


@dataclass
class _Open:
    start: int
    speech_end: int
    speech_samples: int = 0
    held: bool = False
    recent: deque = field(default_factory=deque)  # (start, len, energy, speech)


def _longest_pause_middle(frames) -> int | None:
    longest = run = None
    for start, length, _, speech in frames:
        if speech:
            run = None
            continue
        run = (run[0] if run else start, start + length)
        if longest is None or run[1] - run[0] >= longest[1] - longest[0]:
            longest = run
    return None if longest is None else (longest[0] + longest[1]) // 2


def segment(samples: np.ndarray, is_speech, max_samples: int) -> list[tuple[int, int]]:
    """`SpeechSegmenter` over a whole recording: (start, end) sample ranges to decode."""
    segments: list[tuple[int, int]] = []
    position = 0
    emitted_end = 0
    active: _Open | None = None

    def emit(start: int, end: int, speech_samples: int) -> None:
        nonlocal emitted_end
        emitted_end = end
        if speech_samples >= MIN_KEPT_SPEECH:
            segments.append((start, end))

    def close(segment: _Open) -> None:
        emit(segment.start, min(segment.speech_end + PADDING, position), segment.speech_samples)

    for frame_start in range(0, len(samples), FRAME):
        frame = samples[frame_start : frame_start + FRAME]
        length = len(frame)
        if length < FRAME:
            frame = np.pad(frame, (0, FRAME - length))
        start, end = position, position + length
        position = end
        speech = is_speech(frame)
        if active is None:
            if not speech:
                continue
            active = _Open(start=max(start - PADDING, emitted_end, 0), speech_end=start)
        if speech:
            active.speech_end = end
            active.speech_samples += length
            active.held = False
        active.recent.append((start, length, float(np.mean(frame[:length] ** 2)), speech))
        while active.recent and active.recent[0][0] < end - max_samples // 2:
            active.recent.popleft()
        pause = end - active.speech_end
        if not speech and not active.held and pause >= END_SILENCE:
            if active.speech_samples >= MIN_SEGMENT_SPEECH:
                close(active)
                active = None
                continue
            active.held = True
        if active.held and pause >= MAX_MERGE_GAP:
            close(active)
            active = None
            continue
        if end - active.start >= max_samples:
            window = [f for f in active.recent if f[0] > active.start]
            cut = _longest_pause_middle(window)
            if cut is None:
                cut = min(reversed(window), key=lambda f: f[2])[0] if window else position
            remaining_speech = sum(f[1] for f in active.recent if f[3] and f[0] >= cut)
            emit(active.start, cut, active.speech_samples - remaining_speech)
            if cut < position:
                active.recent = deque(f for f in active.recent if f[0] >= cut)
                active.start = cut
                active.speech_end = max(active.speech_end, cut)
                active.speech_samples = remaining_speech
            else:
                active = None
    if active is not None:
        close(active)
    return segments


def vad_config():
    import sherpa_onnx as so

    config = so.VadModelConfig()
    config.silero_vad.model = str(VAD_MODEL)
    config.silero_vad.threshold = 0.5
    config.silero_vad.min_silence_duration = 0.1
    config.silero_vad.min_speech_duration = 0.064
    config.silero_vad.window_size = FRAME
    config.silero_vad.max_speech_duration = 30.0
    config.sample_rate = SAMPLE_RATE
    config.num_threads = 1
    return config


def segments(samples: np.ndarray, max_samples: int) -> list[np.ndarray]:
    import sherpa_onnx as so

    detector = so.VoiceActivityDetector(vad_config(), buffer_size_in_seconds=30)

    def is_speech(frame: np.ndarray) -> bool:
        detector.accept_waveform(frame)
        speech = detector.is_speech_detected()
        while not detector.empty():  # the app's `clear()`: drop the native segments
            detector.pop()
        return speech

    return [samples[start:end] for start, end in segment(samples, is_speech, max_samples)]


def decode_segment(recognizer, samples: np.ndarray) -> tuple[str, float]:
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    start = time.perf_counter()
    recognizer.decode_stream(stream)
    elapsed = time.perf_counter() - start
    return stream.result.text, elapsed


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
    voice: int | str
    speed: float
    chunks: list[str]
    pauses: list[float]


class Speaker:
    """Speaks a text in chunks with pauses between them, in one voice and speed drawn per text."""

    name: str

    def _voice(self, rng: random.Random) -> tuple[int | str, float]:
        raise NotImplementedError

    def _say(self, piece: str, voice: int | str, speed: float, rng: random.Random) -> np.ndarray:
        raise NotImplementedError

    def speak(self, text: str, rng: random.Random) -> Spoken:
        voice, speed = self._voice(rng)
        pieces = chunks(text, rng)
        parts, pauses = [], []
        for index, piece in enumerate(pieces):
            parts.append(self._say(piece, voice, speed, rng))
            if index < len(pieces) - 1:
                pauses.append(round(pause_seconds(rng), 2))
                parts.append(np.zeros(int(pauses[-1] * SAMPLE_RATE), dtype=np.float32))
        lead = np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32)
        return Spoken(np.concatenate([lead, *parts, lead]), voice, speed, pieces, pauses)


class KokoroSpeaker(Speaker):
    def __init__(self, provider: str = "cpu", model_dir: Path = KOKORO_DIR) -> None:
        import sherpa_onnx as so

        config = so.OfflineTtsConfig(
            model=so.OfflineTtsModelConfig(
                kokoro=so.OfflineTtsKokoroModelConfig(
                    model=str(next(model_dir.glob("model*.onnx"))),
                    voices=str(model_dir / "voices.bin"),
                    tokens=str(model_dir / "tokens.txt"),
                    data_dir=str(model_dir / "espeak-ng-data"),
                    dict_dir=str(model_dir / "dict"),
                    lexicon=str(model_dir / "lexicon-us-en.txt"),
                ),
                num_threads=THREADS,
                provider=provider,
            )
        )
        self.tts = so.OfflineTts(config)
        self.name = f"{model_dir.name}-{provider}"

    def _voice(self, rng: random.Random) -> tuple[int, float]:
        return rng.choice(ENGLISH_VOICES), round(rng.uniform(0.95, 1.15), 2)

    def _say(self, piece: str, voice: int | str, speed: float, rng: random.Random) -> np.ndarray:
        audio = self.tts.generate(piece, sid=voice, speed=speed)
        return resample(np.asarray(audio.samples, dtype=np.float32), audio.sample_rate)


class PhononSpeaker(Speaker):
    """gradium's phonon-31m-en through its `ptts` runtime, on the CPU (ptts 0.4.1 has no GPU build).

    Half the texts get an American or British voice, half a French, German, Spanish or Portuguese one
    reading English, which Parakeet hears as accented English. Phonon has no speed setting.
    """

    def __init__(self) -> None:
        import ptts

        ptts.set_num_threads(THREADS)
        self.tts = ptts.TTS(PHONON_MODEL, lang="en", quant="q8", voice_catalog=PHONON_VOICES)
        voices = ptts.catalog_voices(PHONON_VOICES)
        self.english = sorted(v["name"] for v in voices if v["language"] == "en" and v["name"] not in PHONON_NOT_HUMAN)
        self.accented = sorted(v["name"] for v in voices if v["language"] != "en")
        self.name = "phonon-31m-en-q8"

    def _voice(self, rng: random.Random) -> tuple[str, float]:
        return rng.choice(self.english if rng.random() < 0.5 else self.accented), 1.0

    def _say(self, piece: str, voice: int | str, speed: float, rng: random.Random) -> np.ndarray:
        audio = self.tts.synth(piece, voice=voice, seed=rng.randrange(2**32))
        return resample(np.asarray(audio, dtype=np.float32), self.tts.sample_rate)


def speaker(tts: str, provider: str = "cpu") -> Speaker:
    """A Kokoro (on `provider`) or Phonon speaker."""
    if tts == "kokoro":
        return KokoroSpeaker(provider, KOKORO_GPU_DIR if provider == "cuda" else KOKORO_DIR)
    if tts == "phonon":
        return PhononSpeaker()
    raise ValueError(f"unknown text-to-speech model {tts!r}; expected one of {TTS_MODELS}")


class Recognizer:
    def __init__(self, model: str = "parakeet", provider: str = "cpu") -> None:
        import sherpa_onnx as so

        self.max_segment = max_segment(model)
        self.name = f"{model}-{provider}"

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
                provider=provider,
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
                provider=provider,
            )
        elif model == "qwen3":
            # A 28 s segment is about 364 audio tokens, more than the default 512-token budget allows with the text.
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

    def transcribe(self, samples: np.ndarray) -> tuple[str, list[str]]:
        texts = [decode_segment(self.recognizer, part)[0] for part in segments(samples, self.max_segment)]
        return join_segment_texts(texts), texts


def sample_rows(n: int, seed: int) -> list:
    rows = [e for split in ("train", "val", "test") for e in read_examples(paths.ROUND1 / f"{split}.jsonl")]
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
    out = OUT_DIR / (f"parakeet.{shard}.jsonl" if shards > 1 else "parakeet.jsonl")
    extra_fields = {
        e.id: {"source": e.source, "category": e.category, "layout": e.layout, "tags": e.tags}
        for e in todo
    }
    return probe_texts(
        [(e.id, e.input) for e in todo], out, seed,
        extra_fields=extra_fields, keep_audio=keep_audio, progress_prefix=f"shard {shard}: ",
    )


def probe_texts(
    items: list[tuple[str, str]], out: Path, seed: int, *,
    extra_fields: dict[str, dict] | None = None, keep_audio: bool = False, progress_prefix: str = "",
) -> int:
    """Speaks and transcribes (id, text) pairs into `out`, one JSON line each; resumes past ids already there."""
    done = {json.loads(line)["id"] for line in out.open(encoding="utf-8")} if out.exists() else set()
    todo = [(id_, text) for id_, text in items if id_ not in done]
    if not todo:
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    speaker, recognizer = KokoroSpeaker(), Recognizer()
    written = 0
    with out.open("a", encoding="utf-8") as sink:
        for id_, text in todo:
            spoken = speaker.speak(text, random.Random(f"{seed}-{id_}"))
            heard, segment_texts = recognizer.transcribe(spoken.samples)
            if keep_audio:
                _write_wav(AUDIO_DIR / f"{id_}.wav", spoken.samples)
            record = {
                "id": id_, "spoken": text, "model": "parakeet", "heard": heard, "segments": segment_texts,
                "chunk_ends": [len(" ".join(spoken.chunks[: i + 1]).split()) for i in range(len(spoken.chunks))],
                "pauses": spoken.pauses, "voice": spoken.voice, "speed": spoken.speed,
                "seconds": round(len(spoken.samples) / SAMPLE_RATE, 1),
            }
            if extra_fields is not None:
                record |= extra_fields[id_]
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            written += 1
            print(f"{progress_prefix}{written}/{len(todo)} {id_} {record['seconds']}s", flush=True)
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
            text, segment_texts = recognizer.transcribe(samples)
            sink.write(json.dumps(record | {"model": model, "heard": text, "segments": segment_texts}, ensure_ascii=False) + "\n")
            sink.flush()
            written += 1
            print(f"{model} shard {shard}: {written}/{len(todo)} {record['id']}", flush=True)
    return written


def load(model: str = "parakeet") -> list[dict]:
    return [json.loads(line) for path in _out_files(model) for line in path.open(encoding="utf-8")]


_WORD = re.compile(r"[A-Za-z0-9$%][\w'’.:,/%$-]*")
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


ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def say(n: int) -> str:
    """An integer in English words, as Kokoro read the probe inputs."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    for size, word in ((10**9, "billion"), (10**6, "million"), (1000, "thousand"), (100, "hundred")):
        if n >= size:
            rest = n % size
            return say(n // size) + " " + word + ("" if rest == 0 else " " + say(rest))
    raise AssertionError


def normalize(text: str) -> list[str]:
    """Words for WER: lower case, digits spelled out, punctuation and fillers dropped."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    text = re.sub(r"\$(\d+(?:\.\d+)?)", r"\1 dollars", text)
    text = text.replace("%", " percent").replace("&", " and ")
    text = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", text)

    def number(m):
        whole, _, frac = m.group(0).partition(".")
        spoken = say(int(whole))
        return spoken + (" point " + " ".join(ONES[int(d)] for d in frac) if frac else "")

    text = re.sub(r"\d+(?:\.\d+)?", number, text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    words = [w.strip("'") for w in text.split()]
    return [w.replace("okay", "ok") for w in words if w and not FILLER.fullmatch(w)]


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
        fillers["spoken"] += sum(FILLER.fullmatch(w) is not None for w in spoken)
        for match in re.finditer(rf"(^|.)\s*\b({FILLER_FORMS})\b([.,?!]?)", record["heard"], re.IGNORECASE):
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
