"""Mercury Decide, Inception's free yes/no model on OpenRouter's System One endpoint: it answers with P(yes)."""

from __future__ import annotations

import asyncio
import os
import random

import httpx
from dotenv import load_dotenv

from . import paths

MODEL = "inception/mercury-decide:free"
URL = "https://openrouter.ai/api/v1/systemone"


async def decide(client: httpx.AsyncClient, key: str, state: str, question: str) -> float | None:
    """Mercury Decide's probability that the answer to `question` about `state` is yes; None if it never answered."""
    body = {"model": MODEL, "state": state, "questions": {"q": {"type": "noul", "instructions": question}}}
    for attempt in range(8):
        try:
            response = await client.post(URL, json=body, headers={"Authorization": f"Bearer {key}"})
        except httpx.TransportError:
            pass
        else:
            if response.status_code == 200:
                return response.json()["answers"]["q"]["noul"]
            if response.status_code not in (408, 429, 500, 502, 503, 504):
                print(f"Mercury Decide: HTTP {response.status_code}: {response.text[:200]}", flush=True)
                return None
        await asyncio.sleep(min(2**attempt + random.random(), 60))
    return None


def openrouter_key() -> str:
    load_dotenv(paths.REPO_ROOT / ".env")
    if not (key := os.environ.get("OPENROUTER_API_KEY")):
        raise SystemExit("set OPENROUTER_API_KEY in the repo-root .env")
    return key
