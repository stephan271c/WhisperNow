"""CPU speed and English WER of each catalog speech model, decoded as the app does.

The audio is the probe's Kokoro dictations (`asr_probe.py`), cut by the app's segmenter and decoded
with the app's recognizer settings: CPU, four threads, greedy search, Whisper with automatic
language. Speed is seconds of segment audio per second of decoding. The synthetic speech is
cleaner than a person, so WER here only ranks the models; the app's accuracy ratings come from
the Open ASR Leaderboard where a model is listed.
"""

from __future__ import annotations

import json
import wave

import numpy as np
import sherpa_onnx as so

from . import asr_probe as probe
from . import paths

ASR = paths.CACHE / "asr"
THREADS = 4
CLIPS = 40
OUT = paths.DATA / "probe" / "asr_speed.json"


def transducer(d, precision):
    return lambda: so.OfflineRecognizer.from_transducer(
        encoder=str(d / f"encoder.{precision}.onnx"), decoder=str(d / f"decoder.{precision}.onnx"),
        joiner=str(d / f"joiner.{precision}.onnx"), tokens=str(d / "tokens.txt"), num_threads=THREADS,
        sample_rate=16000, feature_dim=80, decoding_method="greedy_search", model_type="nemo_transducer",
        provider="cpu")


def whisper(d, prefix):
    return lambda: so.OfflineRecognizer.from_whisper(
        encoder=str(d / f"{prefix}-encoder.int8.onnx"), decoder=str(d / f"{prefix}-decoder.int8.onnx"),
        tokens=str(d / f"{prefix}-tokens.txt"), language="", task="transcribe", num_threads=THREADS,
        decoding_method="greedy_search", tail_paddings=0)


MODELS = {
    "parakeet-v2-fp16": transducer(probe.APP_MODELS / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16", "fp16"),
    "parakeet-v2-int8": transducer(ASR / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8", "int8"),
    "parakeet-unified": transducer(ASR / "sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming", "int8"),
    "parakeet-v3": transducer(ASR / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8", "int8"),
    "parakeet-110m": transducer(ASR / "sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000-int8", "int8"),
    "sense-voice": lambda: so.OfflineRecognizer.from_sense_voice(
        model=str(ASR / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx"),
        tokens=str(ASR / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt"),
        num_threads=THREADS, language="auto", use_itn=True),
    "qwen3-asr": lambda: so.OfflineRecognizer.from_qwen3_asr(
        conv_frontend=str(ASR / "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx"),
        encoder=str(ASR / "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx"),
        decoder=str(ASR / "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx"),
        tokenizer=str(ASR / "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer"),
        num_threads=THREADS, max_total_len=1024, max_new_tokens=256),
    "whisper-distil-v3.5": whisper(ASR / "sherpa-onnx-whisper-distil-large-v3.5", "distil-large-v3.5"),
    "whisper-turbo": whisper(ASR / "sherpa-onnx-whisper-turbo", "turbo"),
}

def edits(ref: list[str], hyp: list[str]) -> int:
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1]


def clips() -> list[tuple[str, np.ndarray]]:
    """The spoken text and audio of every n-th probe dictation."""
    records = probe.load()
    records = sorted(records, key=lambda r: r["id"])[:: max(1, len(records) // CLIPS)][:CLIPS]
    out = []
    for r in records:
        with wave.open(str(probe.AUDIO_DIR / f"{r['id']}.wav")) as w:
            samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        out.append((r["spoken"], samples))
    return out


def run(only: list[str] | None = None) -> dict:
    """Measures each model in `only` (default all) not already in `OUT`; returns every result."""
    recordings = clips()
    by_cap: dict[int, list[tuple[str, list[np.ndarray]]]] = {}
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, make in MODELS.items():
        if (only and name not in only) or name in results:
            continue
        cap = probe.max_segment(name)
        if cap not in by_cap:
            by_cap[cap] = [(spoken, probe.segments(samples, cap)) for spoken, samples in recordings]
        data = by_cap[cap]
        audio = sum(len(s) for _, segs in data for s in segs) / 16000
        recognizer = make()
        probe.decode_segment(recognizer, data[0][1][0])
        spent = 0.0
        errors = words = 0
        for spoken, segments in data:
            texts = []
            for segment in segments:
                text, elapsed = probe.decode_segment(recognizer, segment)
                spent += elapsed
                texts.append(text)
            ref, hyp = probe.normalize(spoken), probe.normalize(probe.join_segment_texts(texts))
            errors += edits(ref, hyp)
            words += len(ref)
        results[name] = {"times_real_time": round(audio / spent, 1), "wer": round(100 * errors / words, 2)}
        print(name, results[name], flush=True)
        OUT.write_text(json.dumps(results, indent=1) + "\n")
    return results
