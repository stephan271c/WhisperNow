"""Rate-limited, cached LLM calls configured from the repo-root `.env`.

Two backends share the concurrency cap, the retry policy and the response cache:

- `opencode` (default) sends each request through a private `opencode serve`
  process. The server runs with its own empty config directory, so the user's
  global opencode instructions and plugins never reach the prompt. opencode
  still adds a short environment note before our system message.
- `openai` sends requests to any OpenAI-compatible endpoint with `AsyncOpenAI`.

Only the final message text is returned and cached; reasoning is dropped.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv

from .paths import CACHE, REPO_ROOT

Backend = Literal["opencode", "openai"]
Role = Literal["teacher", "judge"]
Message = dict[str, str]

_MAX_ATTEMPTS = 7
_BACKOFF_BASE_SECONDS = 4.0
_BACKOFF_CAP_SECONDS = 180.0
_REQUEST_TIMEOUT_SECONDS = 900.0


class LlmError(RuntimeError):
    """A request failed. `retryable` marks rate limits, overloads and timeouts."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class LlmSettings:
    backend: Backend
    model: str
    judge_model: str
    concurrency: int
    # opencode `variant` or OpenAI `reasoning_effort`; empty leaves the provider default.
    reasoning: str
    judge_reasoning: str
    opencode_provider: str
    base_url: str | None

    @classmethod
    def from_env(cls) -> LlmSettings:
        load_dotenv(REPO_ROOT / ".env")
        backend = os.environ.get("LLM_BACKEND", "opencode")
        if backend not in ("opencode", "openai"):
            raise SystemExit(f"LLM_BACKEND must be opencode or openai, not {backend!r}")
        model = os.environ.get("LLM_MODEL", "glm-5.3-flash")
        return cls(
            backend=backend,
            model=model,
            judge_model=os.environ.get("LLM_JUDGE_MODEL") or model,
            concurrency=int(os.environ.get("LLM_CONCURRENCY", "3")),
            reasoning=os.environ.get("LLM_REASONING", "low"),
            judge_reasoning=os.environ.get("LLM_JUDGE_REASONING") or os.environ.get("LLM_REASONING", "low"),
            opencode_provider=os.environ.get("LLM_OPENCODE_PROVIDER", "zai-coding-plan"),
            base_url=os.environ.get("LLM_BASE_URL"),
        )


def cache_key(model: str, reasoning: str, messages: list[Message]) -> str:
    request = json.dumps(
        {"model": model, "reasoning": reasoning, "messages": messages},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(request.encode()).hexdigest()


class ResponseCache:
    """One file per request hash, written atomically so a killed run never leaves half a file."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["content"]

    def put(self, key: str, content: str, meta: dict) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".tmp")
        partial.write_text(json.dumps({"content": content, **meta}, ensure_ascii=False), encoding="utf-8")
        partial.replace(path)


def _exit_with_parent() -> None:
    """On Linux, ends the server when datagen dies, even from a signal that skips cleanup."""
    if sys.platform.startswith("linux"):
        import ctypes

        pr_set_pdeathsig = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(pr_set_pdeathsig, signal.SIGTERM)


class _OpencodeServer:
    """A private `opencode serve` with an isolated config dir and a tool-less agent."""

    AGENT = "datagen"

    def __init__(self, settings: LlmSettings) -> None:
        self.settings = settings
        self.root = CACHE / "opencode"
        self.process: subprocess.Popen[bytes] | None = None
        self.url = ""

    def _write_config(self) -> Path:
        config_home = self.root / "config"
        (config_home / "opencode").mkdir(parents=True, exist_ok=True)
        config = {
            "$schema": "https://opencode.ai/config.json",
            "autoupdate": False,
            "share": "disabled",
            "agent": {
                self.AGENT: {
                    "description": "Plain completions for WhisperNow data generation",
                    "mode": "primary",
                    # Must be non-empty, or opencode substitutes its coding-agent prompt.
                    "prompt": "Follow the instructions at the end of this system message.",
                    "tools": {"*": False},
                }
            },
        }
        (config_home / "opencode" / "opencode.json").write_text(json.dumps(config, indent=2))
        return config_home

    async def start(self) -> None:
        opencode = shutil.which("opencode")
        if opencode is None:
            raise SystemExit("LLM_BACKEND=opencode needs the opencode CLI on PATH")
        config_home = self._write_config()
        work = self.root / "work"
        work.mkdir(parents=True, exist_ok=True)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = {**os.environ, "XDG_CONFIG_HOME": str(config_home), "OPENCODE_DISABLE_CLAUDE_CODE": "1"}
        log = (self.root / "serve.log").open("wb")
        self.process = subprocess.Popen(
            [opencode, "serve", "--pure", "--port", str(port)],
            cwd=work,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=_exit_with_parent,
        )
        self.url = f"http://127.0.0.1:{port}"
        async with httpx.AsyncClient() as client:
            for _ in range(120):
                if self.process.poll() is not None:
                    raise SystemExit(f"opencode serve exited; see {self.root / 'serve.log'}")
                try:
                    agents = (await client.get(f"{self.url}/agent", timeout=2)).json()
                    if any(agent["name"] == self.AGENT for agent in agents):
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)
        raise SystemExit("opencode serve did not become ready")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()

    async def complete(
        self, client: httpx.AsyncClient, model: str, reasoning: str, messages: list[Message]
    ) -> str:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        users = [m["content"] for m in messages if m["role"] == "user"]
        if len(users) != 1 or len(messages) != len(users) + (1 if system else 0):
            raise ValueError("the opencode backend sends one optional system message and one user message")
        session = await client.post(f"{self.url}/session", json={"title": "datagen"})
        session.raise_for_status()
        session_id = session.json()["id"]
        body: dict = {
            "agent": self.AGENT,
            "model": {"providerID": self.settings.opencode_provider, "modelID": model},
            "parts": [{"type": "text", "text": users[0]}],
        }
        if system:
            body["system"] = system
        if reasoning:
            body["variant"] = reasoning
        try:
            response = await client.post(
                f"{self.url}/session/{session_id}/message", json=body, timeout=_REQUEST_TIMEOUT_SECONDS
            )
            if response.status_code >= 400:
                raise LlmError(
                    f"opencode HTTP {response.status_code}: {response.text[:300]}",
                    retryable=response.status_code == 429 or response.status_code >= 500,
                )
            reply = response.json()
        finally:
            await client.delete(f"{self.url}/session/{session_id}")
        error = reply["info"].get("error")
        if error:
            data = error.get("data", {})
            status = data.get("statusCode")
            retryable = bool(data.get("isRetryable")) or status == 429 or (status or 0) >= 500
            raise LlmError(f"{error.get('name')}: {str(data.get('message', ''))[:300]}", retryable=retryable)
        return "".join(part["text"] for part in reply["parts"] if part["type"] == "text")


class LlmClient:
    """Use as `async with LlmClient(settings) as llm:`."""

    def __init__(self, settings: LlmSettings) -> None:
        self.settings = settings
        self.cache = ResponseCache(CACHE / "llm")
        self._semaphore = asyncio.Semaphore(settings.concurrency)
        self._opencode: _OpencodeServer | None = None
        self._http: httpx.AsyncClient | None = None
        self._openai = None
        self._connect_lock = asyncio.Lock()
        self.calls = 0
        self.cache_hits = 0

    async def __aenter__(self) -> LlmClient:
        return self

    async def _connect(self) -> None:
        async with self._connect_lock:
            await self._connect_once()

    async def _connect_once(self) -> None:
        if self.settings.backend == "opencode":
            if self._opencode is None:
                server = _OpencodeServer(self.settings)
                await server.start()
                self._http = httpx.AsyncClient(timeout=60)
                self._opencode = server
        elif self._openai is None:
            from openai import AsyncOpenAI

            # The key is read by the SDK from LLM_API_KEY and never logged.
            self._openai = AsyncOpenAI(
                base_url=self.settings.base_url, api_key=os.environ["LLM_API_KEY"], max_retries=0
            )

    async def __aexit__(self, *_exc: object) -> None:
        if self._http is not None:
            await self._http.aclose()
        if self._opencode is not None:
            self._opencode.stop()

    async def _request(self, model: str, reasoning: str, messages: list[Message]) -> str:
        if self.settings.backend == "opencode":
            assert self._opencode is not None and self._http is not None
            try:
                return await self._opencode.complete(self._http, model, reasoning, messages)
            except httpx.TimeoutException as error:
                raise LlmError(f"timeout: {error}", retryable=True) from error
            except httpx.TransportError as error:
                raise LlmError(f"transport: {error}", retryable=True) from error
        import openai

        extra = {"reasoning_effort": reasoning} if reasoning else {}
        try:
            response = await self._openai.chat.completions.create(
                model=model, messages=messages, timeout=_REQUEST_TIMEOUT_SECONDS, **extra
            )
        except (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError, openai.APIConnectionError) as error:
            raise LlmError(str(error)[:300], retryable=True) from error
        except openai.APIError as error:
            raise LlmError(str(error)[:300], retryable=False) from error
        return response.choices[0].message.content or ""

    async def complete(self, messages: list[Message], *, role: Role = "teacher") -> str:
        """Returns the final message text, from the cache when this exact request was made before.

        The teacher role uses `LLM_MODEL`, the judge role `LLM_JUDGE_MODEL`, each with its reasoning setting.
        """
        if role == "teacher":
            model, reasoning = self.settings.model, self.settings.reasoning
        else:
            model, reasoning = self.settings.judge_model, self.settings.judge_reasoning
        key = cache_key(model, reasoning, messages)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        async with self._semaphore:
            await self._connect()
            for attempt in range(1, _MAX_ATTEMPTS + 1):
                started = time.monotonic()
                try:
                    content = await self._request(model, reasoning, messages)
                    break
                except LlmError as error:
                    if not error.retryable or attempt == _MAX_ATTEMPTS:
                        raise
                    delay = min(_BACKOFF_CAP_SECONDS, _BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))
                    await asyncio.sleep(delay * random.uniform(0.5, 1.0))
            self.calls += 1
        content = strip_think(content)
        self.cache.put(key, content, {"model": model, "seconds": round(time.monotonic() - started, 2)})
        return content


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Drops inline reasoning some providers leave in the message content."""
    return _THINK_BLOCK.sub("", text).strip()


def parse_json_reply(text: str) -> dict | list:
    """Parses a JSON reply, tolerating a surrounding code fence or stray prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        # The outermost object or array, then the first array alone: long replies sometimes drop
        # the closing brace of {"items": [...]}.
        pairs = sorted((("{", "}"), ("[", "]")), key=lambda pair: (text.find(pair[0]) < 0, text.find(pair[0])))
        for opener, closer in pairs:
            start, end = text.find(opener), text.rfind(closer)
            if 0 <= start < end:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
        raise error
