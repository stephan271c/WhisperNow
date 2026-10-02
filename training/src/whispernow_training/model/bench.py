"""Latency against a running `llama-server`: time to first token and total time, one request at a time.

The short dictation is one sentence; the long one is the test row nearest 300 words. Prompt caching
is off so every request pays for its prompt, as a new dictation does. `bench_teacher` times the
hosted teacher on the same two dictations, whole replies only.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time

import httpx

from . import prompt
from .evaluate import load_rows

SHORT = "um so can you remind me to call the dentist tomorrow morning about moving my appointment"
LONG_WORDS = 300


def long_dictation() -> str:
    return min(load_rows("test"), key=lambda row: abs(len(row.input.split()) - LONG_WORDS)).input


def _time_one(client: httpx.Client, url: str, transcript: str) -> dict:
    body = {
        "messages": prompt.messages(transcript),
        "max_tokens": prompt.max_tokens(transcript),
        **prompt.decoding(),
        "stream": True,
        "cache_prompt": False,
    }
    started = time.perf_counter()
    first = None
    pieces = []
    with client.stream("POST", f"{url.rstrip('/')}/chat/completions", json=body) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
            if delta and first is None:
                first = time.perf_counter() - started
            pieces.append(delta)
    return {"ttft": first or 0.0, "total": time.perf_counter() - started, "output": "".join(pieces)}


def bench(url: str, repeats: int = 5) -> dict:
    cases = {"short": SHORT, "long": long_dictation()}
    result: dict = {}
    # A fresh connection per request: llama-server closes kept-alive connections after a few requests.
    with httpx.Client(timeout=600, limits=httpx.Limits(max_keepalive_connections=0)) as client:
        _time_one(client, url, SHORT)  # warm-up
        for name, transcript in cases.items():
            runs = [_time_one(client, url, transcript) for _ in range(repeats)]
            result[name] = {
                "input_words": len(transcript.split()),
                "output_words": len(runs[-1]["output"].split()),
                "ttft_ms": round(1000 * statistics.median(run["ttft"] for run in runs)),
                "total_ms": round(1000 * statistics.median(run["total"] for run in runs)),
            }
    return result


def bench_teacher(repeats: int = 5) -> dict:
    """`LLM_MODEL` with the app's hosted prompt, through the datagen client with its cache skipped."""
    from ..datagen.llm import LlmClient, LlmSettings
    from ..datagen.teacher import clean_teacher_output, teacher_messages

    async def run() -> dict:
        settings = LlmSettings.from_env()
        result: dict = {"model": settings.model, "reasoning": settings.reasoning}
        async with LlmClient(settings) as llm:
            await llm.complete(teacher_messages(SHORT), cached=False)  # warm-up
            for name, transcript in {"short": SHORT, "long": long_dictation()}.items():
                seconds, output = [], ""
                for _ in range(repeats):
                    started = time.perf_counter()
                    output = clean_teacher_output(await llm.complete(teacher_messages(transcript), cached=False))
                    seconds.append(time.perf_counter() - started)
                result[name] = {
                    "input_words": len(transcript.split()),
                    "output_words": len(output.split()),
                    "total_ms": round(1000 * statistics.median(seconds)),
                }
        return result

    return asyncio.run(run())
