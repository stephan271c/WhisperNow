"""Rate-limited, cached LLM calls configured from the repo-root `.env`.

Three backends share the retry policy and the response cache, and each has its own concurrency cap.
Each role (generator, teacher, judge) can use a different backend and model:

- `opencode` (default) sends each request through a private `opencode serve`
  process. The server runs with its own empty config directory, so the user's
  global opencode instructions and plugins never reach the prompt. opencode
  still adds a short environment note before our system message.
- `openai` sends requests to any OpenAI-compatible endpoint with `AsyncOpenAI`.
- `codex` runs one `codex exec` per request. Our system message replaces
  codex's own instructions, and the user's codex config, rules and any AGENTS.md
  are ignored. codex still declares its tools and adds its environment note; the
  README records this.

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
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv

from .paths import CACHE, REPO_ROOT

Backend = Literal["opencode", "openai", "codex"]
Role = Literal["generator", "teacher", "judge"]
Message = dict[str, str]

_MAX_ATTEMPTS = 7
_BACKOFF_BASE_SECONDS = 4.0
_BACKOFF_CAP_SECONDS = 180.0
_REQUEST_TIMEOUT_SECONDS = 900.0
# How long a subprocess's pipes may stay open after it exits, and how long to wait for them to
# close once its process group is killed.
_ORPHAN_GRACE_SECONDS = 10.0
_KILL_WAIT_SECONDS = 30.0
# A request refused for the account's usage limit waits and retries without end, so the run resumes
# once the limit resets or another account signs in; the log says so at most this often.
_LIMIT_WAIT_SECONDS = 60.0
_LIMIT_REPORT_SECONDS = 600.0
# Retried failures are logged at most this often, so a throttled run says why it is quiet.
_RETRY_REPORT_SECONDS = 60.0


class LlmError(RuntimeError):
    """A request failed. `retryable` marks rate limits, overloads and timeouts; `limited` an account that
    cannot take requests until its usage limit resets or the user signs in again."""

    def __init__(self, message: str, *, retryable: bool, limited: bool = False, throttled: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.limited = limited
        self.throttled = throttled


def _backend(name: str, default: str) -> Backend:
    backend = os.environ.get(name) or default
    if backend not in ("opencode", "openai", "codex"):
        raise SystemExit(f"{name} must be opencode, openai or codex, not {backend!r}")
    return backend


@dataclass(frozen=True)
class LlmSettings:
    """`LLM_*` for the teacher; `LLM_JUDGE_*` and `LLM_GENERATOR_*` default to the teacher's settings."""

    backend: Backend
    model: str
    judge_model: str
    concurrency: int
    # opencode `variant`, OpenAI `reasoning_effort` or codex `model_reasoning_effort`; empty leaves the
    # provider default.
    reasoning: str
    judge_reasoning: str
    opencode_provider: str
    base_url: str | None
    judge_backend: Backend
    generator_backend: Backend
    generator_model: str
    generator_reasoning: str
    # Requests in flight on the codex backend, which allows many more than the others.
    codex_concurrency: int = 16

    @classmethod
    def from_env(cls) -> LlmSettings:
        load_dotenv(REPO_ROOT / ".env")
        backend = _backend("LLM_BACKEND", "opencode")
        model = os.environ.get("LLM_MODEL", "glm-5.3-flash")
        reasoning = os.environ.get("LLM_REASONING", "low")
        return cls(
            backend=backend,
            model=model,
            judge_model=os.environ.get("LLM_JUDGE_MODEL") or model,
            concurrency=int(os.environ.get("LLM_CONCURRENCY", "3")),
            reasoning=reasoning,
            judge_reasoning=os.environ.get("LLM_JUDGE_REASONING") or reasoning,
            opencode_provider=os.environ.get("LLM_OPENCODE_PROVIDER", "zai-coding-plan"),
            base_url=os.environ.get("LLM_BASE_URL"),
            judge_backend=_backend("LLM_JUDGE_BACKEND", backend),
            generator_backend=_backend("LLM_GENERATOR_BACKEND", backend),
            generator_model=os.environ.get("LLM_GENERATOR_MODEL") or model,
            generator_reasoning=os.environ.get("LLM_GENERATOR_REASONING") or reasoning,
            codex_concurrency=int(os.environ.get("LLM_CODEX_CONCURRENCY", "16")),
        )

    def target(self, role: Role) -> tuple[Backend, str, str]:
        return {
            "teacher": (self.backend, self.model, self.reasoning),
            "judge": (self.judge_backend, self.judge_model, self.judge_reasoning),
            "generator": (self.generator_backend, self.generator_model, self.generator_reasoning),
        }[role]

    def request(self, role: Role) -> dict:
        backend, model, reasoning = self.target(role)
        fields = {"backend": backend, "model": model, "reasoning": reasoning}
        if backend == "openai":
            fields["base_url"] = self.base_url or os.environ.get("OPENAI_BASE_URL")
        elif backend == "opencode":
            fields["provider"] = self.opencode_provider
        return fields


def request_key(fields: dict) -> str:
    request = json.dumps(fields, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(request.encode()).hexdigest()


R112_TARGET = ("codex", "gpt-6.1-sol", "high")


def require_r112(settings: LlmSettings, *roles: Role) -> None:
    wrong = {role: target for role in roles if (target := settings.target(role)) != R112_TARGET}
    if wrong:
        found = ", ".join(f"{role}: {'/'.join(target)}" for role, target in wrong.items())
        raise SystemExit(f"R112 requires {'/'.join(R112_TARGET)}; found {found}")


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


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


async def run_process_group(command: list[str], data: bytes, cwd: Path, timeout: float) -> tuple[int, bytes]:
    """Runs `command` in its own process group with `data` on stdin; returns its exit code and stderr.

    `codex` is a node wrapper around a native binary. Killing only the wrapper leaves the binary
    holding stderr, and asyncio's `wait()` and `communicate()` return only once every pipe closes,
    so a stray binary would hang the caller for good. So the whole group is killed: on timeout, on
    cancellation, `_ORPHAN_GRACE_SECONDS` after the wrapper exits if the pipes are still open,
    and at the end, so nothing outlives the call. Raises `TimeoutError` after `timeout` seconds.
    """
    process = await asyncio.create_subprocess_exec(
        *command, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        start_new_session=True, preexec_fn=_exit_with_parent,
    )
    io_task = asyncio.ensure_future(process.communicate(data))
    deadline = time.monotonic() + timeout
    exited_at = None
    try:
        while not io_task.done():
            now = time.monotonic()
            if process.returncode is not None and exited_at is None:
                exited_at = now
            if now >= deadline or (exited_at is not None and now - exited_at >= _ORPHAN_GRACE_SECONDS):
                break
            await asyncio.wait({io_task}, timeout=min(1.0, max(0.01, deadline - now)))
        timed_out = not io_task.done() and process.returncode is None
    finally:
        _kill_group(process.pid)
        if not io_task.done():
            try:
                await asyncio.wait_for(io_task, _KILL_WAIT_SECONDS)
            except TimeoutError:
                pass
    if timed_out:
        raise TimeoutError(f"{command[0]} ran over {timeout:.0f}s")
    stderr = io_task.result()[1] if io_task.done() and not io_task.cancelled() and not io_task.exception() else b""
    return process.returncode if process.returncode is not None else -signal.SIGKILL, stderr


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
        self, client: httpx.AsyncClient, model: str, reasoning: str, prompt: tuple[str, str]
    ) -> str:
        system, user = prompt
        session = await client.post(f"{self.url}/session", json={"title": "datagen"})
        session.raise_for_status()
        session_id = session.json()["id"]
        body: dict = {
            "agent": self.AGENT,
            "model": {"providerID": self.settings.opencode_provider, "modelID": model},
            "parts": [{"type": "text", "text": user}],
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


_CODEX_LIMITED = re.compile(r"usage.?limit|insufficient_quota|budget_limited", re.I)
# Signed out, as while the user switches accounts with `codex login`.
_CODEX_SIGNED_OUT = re.compile(r"\b401\b|unauthori[sz]ed|not logged in|log ?in again|codex login|refresh token|token.{0,20}expired", re.I)
# `codex exec` ends its stderr log with the request's token count, as "tokens used" and the number.
_CODEX_TOKENS = re.compile(r"tokens used\s*:?\s*([\d,]+)", re.I)
# The service refusing a connection while many requests run: back off and retry, however long it lasts.
_CODEX_THROTTLED = re.compile(r"403 Forbidden|failed to connect to websocket|model is at capacity", re.I)
_CODEX_RETRYABLE = re.compile(r"429|rate.?limit|usage limit|overloaded|timed? ?out|5\d\d|stream disconnected|try again", re.I)


def codex_command(codex: str, model: str, reasoning: str, instructions: Path, reply: Path) -> list[str]:
    """The `codex exec` call for one request; the user message goes on stdin."""
    command = [
        codex, "exec", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules",
        "--sandbox", "read-only", "--color", "never", "--model", model,
        "-c", f"model_instructions_file={json.dumps(str(instructions))}",
        "-c", "project_doc_max_bytes=0",
        "-c", 'web_search="disabled"',
        # Read the login from ~/.codex/auth.json, never the desktop keyring: Secret Service
        # requests from codex were crashing gnome-keyring-daemon.
        "-c", 'cli_auth_credentials_store="file"',
        "--output-last-message", str(reply),
    ]
    if reasoning:
        command += ["-c", f"model_reasoning_effort={json.dumps(reasoning)}"]
    return [*command, "-"]


class _CodexRunner:
    """Runs `codex exec` from an empty directory outside the repo, with our system message as its instructions."""

    def __init__(self) -> None:
        codex = shutil.which("codex")
        if codex is None:
            raise SystemExit("LLM_BACKEND=codex needs the codex CLI on PATH")
        self.codex = codex
        self.instructions = CACHE / "codex" / "instructions"
        self.instructions.mkdir(parents=True, exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix="datagen-codex-"))
        # Tokens the requests used, as codex reports them: the quota a run spends.
        self.tokens = 0

    def _instructions_file(self, system: str) -> Path:
        path = self.instructions / f"{hashlib.sha256(system.encode()).hexdigest()[:16]}.md"
        if not path.exists():
            partial = path.with_suffix(f".{os.getpid()}.tmp")
            partial.write_text(system, encoding="utf-8")
            partial.replace(path)
        return path

    async def complete(self, model: str, reasoning: str, prompt: tuple[str, str]) -> str:
        system, user = prompt
        instructions = self._instructions_file(system or "Follow the user's instructions.")
        with tempfile.NamedTemporaryFile(dir=self.work, suffix=".txt", delete=False) as handle:
            reply = Path(handle.name)
        try:
            try:
                returncode, stderr = await run_process_group(
                    codex_command(self.codex, model, reasoning, instructions, reply), user.encode(), self.work,
                    _REQUEST_TIMEOUT_SECONDS,
                )
            except TimeoutError as error:
                raise LlmError("codex exec timed out", retryable=True) from error
            if returncode != 0:
                tail = stderr.decode(errors="replace").strip()[-600:]
                raise LlmError(
                    f"codex exit {returncode}: {tail}",
                    retryable=bool(_CODEX_RETRYABLE.search(tail) or _CODEX_THROTTLED.search(tail)),
                    limited=bool(_CODEX_LIMITED.search(tail) or _CODEX_SIGNED_OUT.search(tail))
                    and not _CODEX_THROTTLED.search(tail),
                    throttled=bool(_CODEX_THROTTLED.search(tail)),
                )
            content = reply.read_text(encoding="utf-8")
            log = stderr.decode(errors="replace")
            token_counts = _CODEX_TOKENS.findall(log)
            if token_counts:
                self.tokens += int(token_counts[-1].replace(",", ""))
        finally:
            reply.unlink(missing_ok=True)
        if not content.strip():
            # codex can exit 0 without a reply when the service refuses the request; its log says why.
            tail = log.strip()[-600:]
            raise LlmError(
                f"codex returned an empty reply: {tail}", retryable=True,
                # A refused connection's log can mention tokens and auth, but it is throttling, not a sign-out.
                limited=bool(_CODEX_LIMITED.search(tail))
                or (bool(_CODEX_SIGNED_OUT.search(tail)) and not _CODEX_THROTTLED.search(tail)),
                throttled=bool(_CODEX_THROTTLED.search(tail)),
            )
        return content

    def stop(self) -> None:
        shutil.rmtree(self.work, ignore_errors=True)


class LlmClient:
    """Use as `async with LlmClient(settings) as llm:`."""

    def __init__(self, settings: LlmSettings) -> None:
        self.settings = settings
        self.cache = ResponseCache(CACHE / "llm")
        self._semaphores = {
            backend: asyncio.Semaphore(settings.codex_concurrency if backend == "codex" else settings.concurrency)
            for backend in ("opencode", "openai", "codex")
        }
        self._opencode: _OpencodeServer | None = None
        self._http: httpx.AsyncClient | None = None
        self._openai = None
        self._codex: _CodexRunner | None = None
        self._connect_lock = asyncio.Lock()
        self.calls = 0
        self.cache_hits = 0
        # Seconds the answered requests took, summed, and usage-limit refusals waited out.
        self.request_seconds = 0.0
        self.limit_waits = 0
        self.retries = 0
        self._retry_reported = 0.0
        self._limit_reported = 0.0
        self._limited = False

    @property
    def tokens(self) -> int:
        """Tokens the codex requests used so far; the other backends do not report them."""
        return self._codex.tokens if self._codex is not None else 0

    async def __aenter__(self) -> LlmClient:
        return self

    async def _connect(self, backend: Backend) -> None:
        async with self._connect_lock:
            if backend == "opencode":
                if self._opencode is None:
                    server = _OpencodeServer(self.settings)
                    await server.start()
                    self._http = httpx.AsyncClient(timeout=60)
                    self._opencode = server
            elif backend == "codex":
                if self._codex is None:
                    self._codex = _CodexRunner()
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
        if self._codex is not None:
            self._codex.stop()

    async def _request(
        self, backend: Backend, model: str, reasoning: str, messages: list[Message] | tuple[str, str]
    ) -> str:
        if backend == "opencode":
            assert self._opencode is not None and self._http is not None
            try:
                return await self._opencode.complete(self._http, model, reasoning, messages)
            except httpx.TimeoutException as error:
                raise LlmError(f"timeout: {error}", retryable=True) from error
            except httpx.TransportError as error:
                raise LlmError(f"transport: {error}", retryable=True) from error
        if backend == "codex":
            assert self._codex is not None
            return await self._codex.complete(model, reasoning, messages)
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

    async def complete(
        self,
        messages: list[Message],
        *,
        role: Role = "teacher",
        cached: bool = True,
        attempt: int = 0,
        reasoning: str | None = None,
    ) -> str:
        """Returns the final message text, from the cache when this exact request was made before.

        The teacher role uses `LLM_MODEL`, the judge role `LLM_JUDGE_MODEL` and the generator role
        `LLM_GENERATOR_MODEL`, each with its backend and reasoning setting.
        `cached=False` always sends the request, for timing it. `attempt` N > 0 caches the reply apart from
        the plain request's, so a second attempt at the same request gets a new sample that a rerun reuses.
        `reasoning` overrides the role's reasoning setting for this request.
        """
        backend, model, default_reasoning = self.settings.target(role)
        payload: list[Message] | tuple[str, str] = messages
        if backend in ("opencode", "codex"):
            system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            users = [m["content"] for m in messages if m["role"] == "user"]
            if len(users) != 1 or len(messages) != len(users) + (1 if system else 0):
                raise ValueError(f"the {backend} backend sends one optional system message and one user message")
            payload = (system, users[0])
        reasoning = default_reasoning if reasoning is None else reasoning
        key = request_key({**self.settings.request(role), "model": model, "reasoning": reasoning, "messages": messages,
                           **({"attempt": attempt} if attempt else {})})
        cached_content = self.cache.get(key) if cached else None
        if cached_content is not None:
            self.cache_hits += 1
            return cached_content
        async with self._semaphores[backend]:
            await self._connect(backend)
            failures = 0
            refused = False
            while True:
                started = time.monotonic()
                try:
                    content = await self._request(backend, model, reasoning, payload)
                    break
                except LlmError as error:
                    if error.limited:
                        refused = True
                        self.limit_waits += 1
                        self._report_limit(backend, error)
                        await asyncio.sleep(_LIMIT_WAIT_SECONDS * random.uniform(0.75, 1.0))
                        continue
                    failures += 1
                    if not error.retryable or (failures >= _MAX_ATTEMPTS and not error.throttled):
                        raise
                    self._report_retry(backend, error, failures)
                    delay = min(_BACKOFF_CAP_SECONDS, _BACKOFF_BASE_SECONDS * 2 ** min(failures - 1, 8))
                    await asyncio.sleep(delay * random.uniform(0.5, 1.0))
            # Only a request that was refused itself shows the account is back; one sent before the
            # limit can still finish after it.
            if refused and self._limited:
                self._limited = False
                print(f"{time.strftime('%H:%M')} ACCOUNT OK again on {backend}; requests go through", file=sys.stderr, flush=True)
            self.calls += 1
            self.request_seconds += time.monotonic() - started
        content = strip_think(content)
        self.cache.put(key, content, {"model": model, "seconds": round(time.monotonic() - started, 2)})
        return content


    def _report_retry(self, backend: Backend, error: LlmError, attempt: int) -> None:
        """Logs a retried failure (a rate limit, overload or timeout), once per `_RETRY_REPORT_SECONDS`."""
        self.retries += 1
        if time.monotonic() - self._retry_reported >= _RETRY_REPORT_SECONDS:
            self._retry_reported = time.monotonic()
            print(
                f"{time.strftime('%H:%M')} RETRY on {backend} ({self.retries} so far, attempt {attempt}): {str(error)[-300:]}",
                file=sys.stderr, flush=True,
            )

    def _report_limit(self, backend: Backend, error: LlmError) -> None:
        """Logs a usage-limit or signed-out refusal, once per `_LIMIT_REPORT_SECONDS` while it lasts."""
        self._limited = True
        if time.monotonic() - self._limit_reported >= _LIMIT_REPORT_SECONDS or not self._limit_reported:
            self._limit_reported = time.monotonic()
            what = "USAGE LIMIT" if _CODEX_LIMITED.search(str(error)) else "SIGNED OUT"
            print(
                f"{time.strftime('%H:%M')} {what} on {backend}, waiting and retrying every minute: {str(error)[-300:]}",
                file=sys.stderr, flush=True,
            )


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
        # A reply that ends before its last closing brackets, as the judge's one-row replies
        # sometimes do ({"results": [{...}] with no "]}"), is completed.
        closed = _close_brackets(text)
        if closed is not None:
            try:
                return json.loads(closed)
            except json.JSONDecodeError:
                pass
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


def _close_brackets(text: str) -> str | None:
    """`text` from its first `{` or `[` with the brackets still open at its end closed, or None when
    nothing is open, a bracket is mismatched, or the text ends inside a string."""
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        return None
    text = text[min(starts) :]
    closers = {"{": "}", "[": "]"}
    stack: list[str] = []
    in_string = escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in closers:
            stack.append(closers[character])
        elif character in "}]":
            if not stack or stack.pop() != character:
                return None
    if in_string or not stack:
        return None
    return text.rstrip() + "".join(reversed(stack))
