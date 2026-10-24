"""Rate-limited, cached LLM calls configured from the repo-root `.env`.

Four backends share the retry policy and the response cache, and each has its own concurrency cap.
Each role (generator, teacher, judge) can use a different backend and model:

- `opencode` (default) sends each request through a private `opencode serve`
  process (opencode 2). The server runs with its own empty config directory, and
  requests use its stateless generate endpoint, so no agent prompt, tools,
  instructions or plugins reach the model. That endpoint takes one prompt, so our
  system message leads it.
- `openai` sends requests to any OpenAI-compatible endpoint with `AsyncOpenAI`.
  `LlmClient.complete_batch` sends many at once to an OpenRouter `:batch` model.
- `codex` runs one `codex exec` per request. Our system message replaces
  codex's own instructions, and the user's codex config, rules and any AGENTS.md
  are ignored. codex still declares its tools and adds its environment note; the
  README records this.
- `claude` runs one `claude -p` per request, with our system message in place of
  Claude Code's and no tools. Claude Code still adds an identity line, a reminder
  with the account's email, and an environment note after the user message; the
  README records this.

Only the final message text is returned and cached; reasoning is dropped.

`LLM_GENERATORS` spreads the generator role over several routes instead (`GeneratorPool`), each
`provider:model#reasoning` with a provider from `PROVIDERS`, and `LLM_GENERATOR_FALLBACK` names a
route used only while every other route rests.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv

from .paths import CACHE, REPO_ROOT

Backend = Literal["opencode", "openai", "codex", "claude"]
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
_BATCH_ROUNDS = 3
_BATCH_POLL_SECONDS = 30.0
_BATCH_REPORT_SECONDS = 600.0
_BATCH_TERMINAL = frozenset({"completed", "failed", "expired", "cancelled"})
_BATCH_DEAD = frozenset({"failed", "cancelled"})


class LlmError(RuntimeError):
    """A request failed. `retryable` marks rate limits, overloads and timeouts; `limited` an account that
    cannot take requests until its usage limit resets or the user signs in again; `retry_after` the
    seconds a rate-limited provider asked for."""

    def __init__(
        self, message: str, *, retryable: bool, limited: bool = False, throttled: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.limited = limited
        self.throttled = throttled
        self.retry_after = retry_after


def _backend(name: str, default: str) -> Backend:
    backend = os.environ.get(name) or default
    if backend not in ("opencode", "openai", "codex", "claude"):
        raise SystemExit(f"{name} must be opencode, openai, codex or claude, not {backend!r}")
    return backend


@dataclass(frozen=True)
class Provider:
    """`kind` is the backend that sends a route's requests, `key` the `.env` name of its API key, and
    `concurrency` its requests in flight, shared by its routes (`LLM_<NAME>_CONCURRENCY` overrides it).
    An `agent` provider sends through opencode's coding agent rather than a plain completion."""

    kind: Backend
    concurrency: int
    base_url: str = ""
    key: str = ""
    max_tokens: int | None = None
    agent: bool = False


PROVIDERS = {
    # glm-5.3-flash at its default reasoning spent 5,500 tokens on one 150-word dictation; a batch of
    # long transcripts needs room for that and the reply.
    "nvidia": Provider("openai", 8, "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY", max_tokens=32768),
    "orca": Provider("openai", 4, "https://api.orcarouter.ai/v1", "ORCAROUTER_API_KEY"),
    "openrouter": Provider("openai", 4, "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "opencode": Provider("opencode", 4),
    # `opencode-agent` sends through opencode's own coding agent, so the transcripts come from a model
    # that also read that agent's prompt and tool list.
    "opencode-agent": Provider("opencode", 4, agent=True),
    "codex": Provider("codex", 8),
}


@dataclass(frozen=True)
class Route:
    """A generator route, written `provider:model#reasoning`; the reasoning is optional."""

    provider: str
    model: str
    reasoning: str = ""

    @classmethod
    def parse(cls, text: str) -> Route:
        provider, colon, rest = text.strip().partition(":")
        model, hash_, reasoning = rest.rpartition("#")
        if not hash_:
            model, reasoning = rest, ""
        if not colon or not model or provider not in PROVIDERS:
            raise SystemExit(f"a generator route is provider:model#reasoning, provider one of {', '.join(PROVIDERS)}; got {text!r}")
        return cls(provider, model, reasoning)

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}" + (f"#{self.reasoning}" if self.reasoning else "")


@dataclass(frozen=True)
class Generated:
    text: str
    route: Route


@dataclass(frozen=True)
class LlmSettings:
    """`LLM_*` for the teacher; `LLM_JUDGE_*` and `LLM_GENERATOR_*` default to the teacher's settings."""

    backend: Backend
    model: str
    judge_model: str
    concurrency: int
    # opencode `variant`, OpenAI `reasoning_effort`, codex `model_reasoning_effort` or claude `--effort`; empty leaves the
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
    generators: tuple[Route, ...] = ()
    generator_fallback: Route | None = None

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
            generators=tuple(Route.parse(spec) for spec in os.environ.get("LLM_GENERATORS", "").split(",") if spec.strip()),
            generator_fallback=Route.parse(fallback) if (fallback := os.environ.get("LLM_GENERATOR_FALLBACK")) else None,
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


# The teacher and judge every reference in the dataset was written and checked by.
REFERENCE_TARGET = ("codex", "gpt-6.1-sol", "high")


def require_reference_target(settings: LlmSettings, *roles: Role) -> None:
    wrong = {role: target for role in roles if (target := settings.target(role)) != REFERENCE_TARGET}
    if wrong:
        found = ", ".join(f"{role}: {'/'.join(target)}" for role, target in wrong.items())
        raise SystemExit(f"references need {'/'.join(REFERENCE_TARGET)}; found {found}")


class ResponseCache:
    """One file per request hash, written atomically so a killed run never leaves half a file."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        entry = self.entry(key)
        return entry["content"] if entry is not None else None

    def entry(self, key: str) -> dict | None:
        """The stored reply with what was saved beside it."""
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

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


async def run_process_group(
    command: list[str], data: bytes, cwd: Path, timeout: float, stdout: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    """Runs `command` in its own process group with `data` on stdin; returns its exit code and stderr,
    and writes its stdout to `stdout` when given.

    `codex` is a node wrapper around a native binary. Killing only the wrapper leaves the binary
    holding stderr, and asyncio's `wait()` and `communicate()` return only once every pipe closes,
    so a stray binary would hang the caller for good. So the whole group is killed: on timeout, on
    cancellation, `_ORPHAN_GRACE_SECONDS` after the wrapper exits if the pipes are still open,
    and at the end, so nothing outlives the call. Raises `TimeoutError` after `timeout` seconds.
    """
    sink = stdout.open("wb") if stdout is not None else subprocess.DEVNULL
    try:
        process = await asyncio.create_subprocess_exec(
            *command, cwd=cwd, stdin=subprocess.PIPE, stdout=sink, stderr=subprocess.PIPE,
            start_new_session=True, preexec_fn=_exit_with_parent, env=env,
        )
    finally:
        if stdout is not None:
            sink.close()
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
    """A private `opencode serve` (opencode 2) with an isolated config dir and a password of its own."""

    def __init__(self) -> None:
        self.root = CACHE / "opencode"
        self.agent_work = self.root / "agent-work"
        self.process: subprocess.Popen[bytes] | None = None
        self.opencode = ""
        self.url = ""
        self.auth = ("opencode", secrets.token_urlsafe(24))

    def _write_config(self) -> Path:
        config_home = self.root / "config"
        (config_home / "opencode").mkdir(parents=True, exist_ok=True)
        config = {
            "$schema": "https://opencode.ai/config.json",
            "update": "disable",
            "share": "disabled",
            "instructions": [],
            "plugins": [],
            # Every tool asks, and a non-interactive `opencode run` rejects every ask, so the coding
            # agent behind `run_agent` can call no tool. Denying them instead removes them from the
            # request, and the provider then refuses it.
            "permissions": [{"action": "*", "resource": "*", "effect": "ask"}],
        }
        (config_home / "opencode" / "opencode.json").write_text(json.dumps(config, indent=2))
        return config_home

    async def start(self) -> None:
        opencode = shutil.which("opencode")
        if opencode is None:
            raise SystemExit("LLM_BACKEND=opencode needs the opencode CLI on PATH")
        self.opencode = opencode
        config_home = self._write_config()
        work = self.root / "work"
        work.mkdir(parents=True, exist_ok=True)
        self.agent_work.mkdir(parents=True, exist_ok=True)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = {
            **os.environ, "XDG_CONFIG_HOME": str(config_home), "OPENCODE_DISABLE_CLAUDE_CODE": "1",
            "OPENCODE_SERVER_PASSWORD": self.auth[1],
        }
        log = (self.root / "serve.log").open("wb")
        self.process = subprocess.Popen(
            [opencode, "serve", "--port", str(port)],
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
                    if (await client.get(f"{self.url}/api/info", auth=self.auth, timeout=2)).status_code == 200:
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
        self, client: httpx.AsyncClient, provider: str, model: str, variant: str, prompt: tuple[str, str]
    ) -> str:
        system, user = prompt
        body = {
            "prompt": f"{system}\n\n{user}" if system else user,
            "model": {"id": model, "providerID": provider, **({"variant": variant} if variant else {})},
        }
        response = await client.post(
            f"{self.url}/api/experimental/generate", json=body, auth=self.auth, timeout=_REQUEST_TIMEOUT_SECONDS
        )
        if response.status_code >= 400:
            raise route_error("opencode", response.status_code, response.text)
        return response.json()["data"]["text"]

    async def run_agent(self, provider: str, model: str, variant: str, prompt: tuple[str, str]) -> str:
        """One `opencode run` on opencode's own coding agent, whose system prompt and tool list come
        with the request, from an empty directory; the text of its last step that wrote any."""
        system, user = prompt
        target = f"{provider}/{model}" + (f"#{variant}" if variant else "")
        with tempfile.NamedTemporaryFile(dir=self.root, suffix=".jsonl", delete=False) as handle:
            events = Path(handle.name)
        try:
            try:
                returncode, stderr = await run_process_group(
                    [self.opencode, "run", "--server", self.url, "--format", "json", "-m", target],
                    (f"{system}\n\n{user}" if system else user).encode(), self.agent_work, _REQUEST_TIMEOUT_SECONDS,
                    stdout=events, env={**os.environ, "OPENCODE_SERVER_PASSWORD": self.auth[1]},
                )
            except TimeoutError as error:
                raise LlmError("opencode run timed out", retryable=True) from error
            lines = events.read_text(encoding="utf-8").splitlines()
        finally:
            events.unlink(missing_ok=True)
        steps: list[list[str]] = []
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "error":
                error = event.get("error") or {}
                body = (error.get("response") or {}).get("body", "")
                raise route_error("opencode-agent", error.get("status"), f"{error.get('message', '')} {body}".strip())
            if event.get("type") == "step_start":
                steps.append([])
            elif event.get("type") == "text" and steps:
                steps[-1].append(event["part"]["text"])
        written = [texts for texts in steps if texts]
        if not written and returncode != 0:
            raise LlmError(f"opencode run exit {returncode}: {stderr.decode(errors='replace').strip()[-300:]}", retryable=True)
        return "".join(written[-1]) if written else ""


# A provider refusing this account, for now or for good.
_ROUTE_GATED = re.compile(r"only available|can only be used|not available to this account|link .{0,40}github", re.I)
_ROUTE_QUOTA = re.compile(r"per.?day|daily|quota|insufficient|usage.?limit", re.I)
# Orca's busy free tier says "capacity is limited right now. Retry shortly, or add credits": a short
# wait, so a billing nag in the message is not read as a quota.
_ROUTE_RATE = re.compile(r"\b429\b|rate.?limit|too many requests|capacity|retry shortly|try again", re.I)


def route_error(source: str, status: int | None, text: str, retry_after: str | None = None) -> LlmError:
    """A provider's refusal as the pool reads it: `limited` rests the route for long, `throttled` for
    `retry_after` or a backoff, a retryable failure counts toward a short rest, and anything else (a bad
    model name or parameter) is a configuration error."""
    message = f"{source} HTTP {status}: {text[:300]}" if status else f"{source}: {text[:300]}"
    if status in (401, 402, 403) or _ROUTE_GATED.search(text) or _ROUTE_QUOTA.search(text):
        return LlmError(message, retryable=True, limited=True)
    if status == 429 or _ROUTE_RATE.search(text):
        seconds = float(retry_after) if retry_after and retry_after.strip().isdigit() else None
        return LlmError(message, retryable=True, throttled=True, retry_after=seconds)
    return LlmError(message, retryable=status is None or status == 408 or status >= 500)


_ROUTE_FAILURES = 3
_ROUTE_REST_SECONDS = 120.0
# An account or quota refusal (a daily cap, a free tier the account does not qualify for yet) is
# tried again this often, so a route joins once its provider takes requests.
_ROUTE_LIMIT_SECONDS = 1800.0
_ROUTE_THROTTLE_SECONDS = 30.0
_ROUTE_THROTTLE_CAP_SECONDS = 600.0


@dataclass
class _RouteState:
    route: Route
    started: int = 0
    rest_until: float = 0.0
    failures: int = 0
    throttles: int = 0
    disabled: str = ""


class GeneratorPool:
    """Spreads generator requests over routes.

    A request goes to the least-used route that is not resting and whose provider has a free slot,
    else waits for one. A route rests after a rate limit (`Retry-After`, else 30 s doubling to 10
    minutes), an account or quota refusal (30 minutes) or three failures in a row (2 minutes); a
    configuration error takes it out of the run. The fallback route takes requests only while every
    other route rests or is out, not while they are merely busy.
    """

    def __init__(self, routes: Sequence[Route], fallback: Route | None, capacity: dict[str, int]) -> None:
        self.states = [_RouteState(route) for route in routes]
        self.fallback = _RouteState(fallback) if fallback is not None else None
        self.capacity = capacity
        self.in_flight: Counter[str] = Counter()
        self._changed = asyncio.Event()

    def _ready(self, state: _RouteState, now: float) -> bool:
        provider = state.route.provider
        return not state.disabled and state.rest_until <= now and self.in_flight[provider] < self.capacity[provider]

    def _pick(self, now: float) -> _RouteState | None:
        ready = [state for state in self.states if self._ready(state, now)]
        if not ready and self.fallback is not None and self._ready(self.fallback, now):
            if all(state.disabled or state.rest_until > now for state in self.states):
                ready = [self.fallback]
        if not ready:
            return None
        state = min(ready, key=lambda state: (state.started, random.random()))
        state.started += 1
        self.in_flight[state.route.provider] += 1
        return state

    async def acquire(self) -> _RouteState:
        while True:
            # Cleared before the check, so a release between the check and the wait still wakes it.
            self._changed.clear()
            now = time.monotonic()
            state = self._pick(now)
            if state is not None:
                return state
            live = [s for s in (*self.states, *([self.fallback] if self.fallback else [])) if not s.disabled]
            if not live:
                out = "; ".join(f"{s.route.label}: {s.disabled}" for s in self.states)
                raise LlmError(f"every generator route is out of the run: {out}", retryable=False)
            wait = min([s.rest_until - now for s in live if s.rest_until > now] + [30.0])
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=max(wait, 0.1))
            except TimeoutError:
                pass

    def release(self, state: _RouteState, error: LlmError | None = None) -> None:
        self.in_flight[state.route.provider] -= 1
        if error is None:
            state.failures = state.throttles = 0
        elif error.limited:
            self._rest(state, _ROUTE_LIMIT_SECONDS, error)
        elif error.throttled:
            state.throttles += 1
            backoff = min(_ROUTE_THROTTLE_CAP_SECONDS, _ROUTE_THROTTLE_SECONDS * 2 ** (state.throttles - 1))
            self._rest(state, error.retry_after or backoff, error)
        elif not error.retryable:
            if not state.disabled:
                state.disabled = str(error)[:300]
                _log(f"ROUTE {state.route.label} is out of the run: {state.disabled}")
        else:
            state.failures += 1
            if state.failures >= _ROUTE_FAILURES:
                state.failures = 0
                self._rest(state, _ROUTE_REST_SECONDS, error)
        self._changed.set()

    @staticmethod
    def _rest(state: _RouteState, seconds: float, error: LlmError) -> None:
        now = time.monotonic()
        if state.rest_until <= now:
            _log(f"ROUTE {state.route.label} rests {seconds:.0f}s: {str(error)[-300:]}")
        state.rest_until = max(state.rest_until, now + seconds)

    def summary(self) -> str:
        states = [*self.states, *([self.fallback] if self.fallback else [])]
        return ", ".join(f"{s.route.label} {s.started}" + (" (out)" if s.disabled else "") for s in states)


def _log(line: str) -> None:
    print(f"{time.strftime('%H:%M')} {line}", file=sys.stderr, flush=True)


_CODEX_LIMITED = re.compile(r"usage.?limit|insufficient_quota|budget_limited", re.I)
# Signed out, as while the user switches accounts with `codex login`.
_CODEX_SIGNED_OUT = re.compile(r"\b401\b|unauthori[sz]ed|not logged in|log ?in again|codex login|refresh token|token.{0,20}expired", re.I)
# `codex exec` ends its stderr log with the request's token count, as "tokens used" and the number.
_CODEX_TOKENS = re.compile(r"tokens used\s*:?\s*([\d,]+)", re.I)
# The service refusing a connection while many requests run: back off and retry, however long it lasts.
_CODEX_THROTTLED = re.compile(r"403 Forbidden|failed to connect to websocket|model is at capacity", re.I)
# Requests over the account's rate limit: codex retries them itself, then gives up with this line.
# Treated as throttled unless the account's usage limit is what refused them.
_CODEX_RATE_LIMITED = re.compile(r"exceeded retry limit, last status: 429", re.I)
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
                limited = bool(_CODEX_LIMITED.search(tail) or _CODEX_SIGNED_OUT.search(tail)) and not _CODEX_THROTTLED.search(tail)
                raise LlmError(
                    f"codex exit {returncode}: {tail}",
                    retryable=bool(_CODEX_RETRYABLE.search(tail) or _CODEX_THROTTLED.search(tail)),
                    limited=limited,
                    throttled=bool(_CODEX_THROTTLED.search(tail) or (_CODEX_RATE_LIMITED.search(tail) and not limited)),
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


_CLAUDE_LIMITED = re.compile(r"usage limit|limit reached|hit your limit|out of extra usage|not logged in|/login|\b401\b", re.I)
_CLAUDE_RETRYABLE = re.compile(r"overloaded|\b529\b|\b5\d\d\b|\b429\b|rate.?limit|timed? ?out|ECONNRESET|socket hang up|try again", re.I)


def claude_command(claude: str, model: str, reasoning: str, system: str) -> list[str]:
    """The `claude -p` call for one request; the user message goes on stdin."""
    command = [
        claude, "-p", "--model", model, "--system-prompt", system, "--tools", "", "--safe-mode",
        "--strict-mcp-config", "--no-session-persistence", "--output-format", "json",
    ]
    if reasoning:
        command += ["--effort", reasoning]
    return command


class _ClaudeRunner:
    """Runs `claude -p` from an empty directory outside the repo, with our system message in place of
    Claude Code's."""

    def __init__(self) -> None:
        claude = shutil.which("claude")
        if claude is None:
            raise SystemExit("LLM_BACKEND=claude needs the claude CLI on PATH")
        self.claude = claude
        self.work = Path(tempfile.mkdtemp(prefix="datagen-claude-"))

    async def complete(self, model: str, reasoning: str, prompt: tuple[str, str]) -> str:
        system, user = prompt
        with tempfile.NamedTemporaryFile(dir=self.work, suffix=".json", delete=False) as handle:
            reply = Path(handle.name)
        try:
            try:
                returncode, stderr = await run_process_group(
                    claude_command(self.claude, model, reasoning, system or "Follow the user's instructions."),
                    user.encode(), self.work, _REQUEST_TIMEOUT_SECONDS, stdout=reply,
                )
            except TimeoutError as error:
                raise LlmError("claude -p timed out", retryable=True) from error
            text = reply.read_text(encoding="utf-8")
        finally:
            reply.unlink(missing_ok=True)
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            result = None
        if returncode != 0 or not isinstance(result, dict) or result.get("is_error") or "result" not in result:
            detail = result.get("result") if isinstance(result, dict) else None
            tail = str(detail or text or stderr.decode(errors="replace")).strip()[-600:]
            raise LlmError(
                f"claude exit {returncode}: {tail}",
                retryable=bool(_CLAUDE_RETRYABLE.search(tail)), limited=bool(_CLAUDE_LIMITED.search(tail)),
            )
        return result["result"]

    def stop(self) -> None:
        shutil.rmtree(self.work, ignore_errors=True)


async def _get_batch(http: httpx.AsyncClient, batch_id: str) -> dict | None:
    """The batch, or None once OpenRouter has deleted it; network and server errors are retried."""
    while True:
        try:
            response = await http.get(f"batches/{batch_id}")
        except httpx.TransportError:
            await asyncio.sleep(_BATCH_POLL_SECONDS)
            continue
        if response.status_code in (404, 410):
            return None
        if response.status_code >= 500 or response.status_code == 429:
            await asyncio.sleep(_BATCH_POLL_SECONDS)
            continue
        if response.status_code >= 400:
            raise LlmError(f"batch {batch_id} ({response.status_code}): {response.text[:300]}", retryable=False)
        return response.json()


async def _wait_for_batch(http: httpx.AsyncClient, batch: dict) -> dict:
    reported = time.monotonic()
    while batch["status"] not in _BATCH_TERMINAL:
        await asyncio.sleep(_BATCH_POLL_SECONDS)
        polled = await _get_batch(http, batch["id"])
        if polled is None:
            raise LlmError(f"batch {batch['id']} was deleted before it finished", retryable=False)
        batch = polled
        if time.monotonic() - reported >= _BATCH_REPORT_SECONDS:
            reported = time.monotonic()
            print(f"{time.strftime('%H:%M')} batch {batch['id']} {batch['status']}: {batch.get('request_counts')}",
                  file=sys.stderr, flush=True)
    return batch


class LlmClient:
    """Use as `async with LlmClient(settings) as llm:`."""

    def __init__(self, settings: LlmSettings) -> None:
        self.settings = settings
        self.cache = ResponseCache(CACHE / "llm")
        self._semaphores = {
            backend: asyncio.Semaphore(settings.codex_concurrency if backend == "codex" else settings.concurrency)
            for backend in ("opencode", "openai", "codex", "claude")
        }
        self._opencode: _OpencodeServer | None = None
        self._http: httpx.AsyncClient | None = None
        self._route_http: httpx.AsyncClient | None = None
        self._pool: GeneratorPool | None = None
        self._openai = None
        self._codex: _CodexRunner | None = None
        self._claude: _ClaudeRunner | None = None
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
                    server = _OpencodeServer()
                    await server.start()
                    self._http = httpx.AsyncClient(timeout=60)
                    self._opencode = server
            elif backend == "codex":
                if self._codex is None:
                    self._codex = _CodexRunner()
            elif backend == "claude":
                if self._claude is None:
                    self._claude = _ClaudeRunner()
            elif self._openai is None:
                from openai import AsyncOpenAI

                # The key is read by the SDK from LLM_API_KEY and never logged.
                self._openai = AsyncOpenAI(
                    base_url=self.settings.base_url, api_key=os.environ["LLM_API_KEY"], max_retries=0
                )

    async def __aexit__(self, *_exc: object) -> None:
        if self._http is not None:
            await self._http.aclose()
        if self._route_http is not None:
            await self._route_http.aclose()
        if self._openai is not None:
            await self._openai.close()
        if self._opencode is not None:
            self._opencode.stop()
        if self._codex is not None:
            self._codex.stop()
        if self._claude is not None:
            self._claude.stop()

    async def _request(
        self, backend: Backend, model: str, reasoning: str, messages: list[Message] | tuple[str, str]
    ) -> str:
        if backend == "opencode":
            assert self._opencode is not None and self._http is not None
            try:
                return await self._opencode.complete(self._http, self.settings.opencode_provider, model, reasoning, messages)
            except httpx.TimeoutException as error:
                raise LlmError(f"timeout: {error}", retryable=True) from error
            except httpx.TransportError as error:
                raise LlmError(f"transport: {error}", retryable=True) from error
        if backend == "codex":
            assert self._codex is not None
            return await self._codex.complete(model, reasoning, messages)
        if backend == "claude":
            assert self._claude is not None
            return await self._claude.complete(model, reasoning, messages)
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
        if role == "generator" and self.settings.generators:
            return (await self.generate(messages, cached=cached, attempt=attempt)).text
        backend, model, default_reasoning = self.settings.target(role)
        payload: list[Message] | tuple[str, str] = messages
        if backend in ("opencode", "codex", "claude"):
            payload = _single_prompt(messages, backend)
        reasoning = default_reasoning if reasoning is None else reasoning
        key = self._key(role, messages, reasoning, attempt)
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

    async def generate(self, messages: list[Message], *, cached: bool = True, attempt: int = 0) -> Generated:
        """The generator's reply and the route that wrote it.

        With `LLM_GENERATORS` set, the pool picks the route, and a cached reply keeps the route that
        wrote it, so a rerun reuses it whatever the pool now holds. A route's rate limits and refusals
        move the request to another route without counting against it; other failures count, up to
        `_MAX_ATTEMPTS`.
        """
        if not self.settings.generators:
            backend, model, reasoning = self.settings.target("generator")
            text = await self.complete(messages, role="generator", cached=cached, attempt=attempt)
            return Generated(text, Route(backend, model, reasoning))
        prompt = _single_prompt(messages, "generator pool")
        key = request_key({"role": "generator", "routes": "pool", "messages": messages,
                           **({"attempt": attempt} if attempt else {})})
        entry = self.cache.entry(key) if cached else None
        if entry is not None:
            self.cache_hits += 1
            return Generated(entry["content"], Route.parse(entry["route"]))
        pool = self._generator_pool()
        failures = 0
        while True:
            state = await pool.acquire()
            started = time.monotonic()
            try:
                content = strip_think(await self._send(state.route, prompt))
                if not content:
                    raise LlmError(f"{state.route.label} returned an empty reply", retryable=True)
            except LlmError as error:
                pool.release(state, error)
                if error.limited or error.throttled:
                    continue
                failures += 1
                if failures >= _MAX_ATTEMPTS:
                    raise
                self._report_retry(state.route.label, error, failures)
                continue
            except BaseException:
                pool.release(state)
                raise
            pool.release(state)
            break
        self.calls += 1
        self.request_seconds += time.monotonic() - started
        self.cache.put(key, content, {"model": state.route.model, "route": state.route.label,
                                      "seconds": round(time.monotonic() - started, 2)})
        return Generated(content, state.route)

    def _generator_pool(self) -> GeneratorPool:
        if self._pool is None:
            routes = [*self.settings.generators, *([self.settings.generator_fallback] if self.settings.generator_fallback else [])]
            for route in routes:
                key = PROVIDERS[route.provider].key
                if key and not os.environ.get(key):
                    raise SystemExit(f"generator route {route.label} needs {key} in .env")
            capacity = {
                name: int(os.environ.get(f"LLM_{name.upper().replace('-', '_')}_CONCURRENCY") or provider.concurrency)
                for name, provider in PROVIDERS.items()
            }
            self._pool = GeneratorPool(self.settings.generators, self.settings.generator_fallback, capacity)
        return self._pool

    async def _send(self, route: Route, prompt: tuple[str, str]) -> str:
        provider = PROVIDERS[route.provider]
        if provider.kind == "codex":
            await self._connect("codex")
            assert self._codex is not None
            async with self._semaphores["codex"]:
                return await self._codex.complete(route.model, route.reasoning, prompt)
        if provider.kind == "opencode":
            await self._connect("opencode")
            assert self._opencode is not None and self._http is not None
            source, _, model = route.model.partition("/")
            if provider.agent:
                return await self._opencode.run_agent(source, model, route.reasoning, prompt)
            try:
                return await self._opencode.complete(self._http, source, model, route.reasoning, prompt)
            except httpx.TimeoutException as error:
                raise LlmError(f"{route.label} timed out", retryable=True) from error
            except httpx.TransportError as error:
                raise LlmError(f"{route.label} transport: {error}", retryable=True) from error
        system, user = prompt
        body: dict = {
            "model": route.model,
            "messages": [*([{"role": "system", "content": system}] if system else []), {"role": "user", "content": user}],
        }
        if route.reasoning:
            body["reasoning_effort"] = route.reasoning
        if provider.max_tokens:
            body["max_tokens"] = provider.max_tokens
        if self._route_http is None:
            self._route_http = httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS)
        try:
            # The key is read from the environment per request and never logged.
            response = await self._route_http.post(
                f"{provider.base_url}/chat/completions", json=body,
                headers={"Authorization": f"Bearer {os.environ[provider.key]}"},
            )
        except httpx.TimeoutException as error:
            raise LlmError(f"{route.label} timed out", retryable=True) from error
        except httpx.TransportError as error:
            raise LlmError(f"{route.label} transport: {error}", retryable=True) from error
        if response.status_code >= 400:
            raise route_error(route.label, response.status_code, response.text, response.headers.get("retry-after"))
        try:
            return response.json()["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as error:
            # Some providers answer 200 with an error object in place of choices.
            raise route_error(route.label, None, response.text) from error

    def _key(self, role: Role, messages: list[Message], reasoning: str, attempt: int = 0) -> str:
        _, model, _ = self.settings.target(role)
        return request_key({**self.settings.request(role), "model": model, "reasoning": reasoning, "messages": messages,
                            **({"attempt": attempt} if attempt else {})})

    async def complete_batch(self, requests: Sequence[list[Message]], *, role: Role = "teacher") -> None:
        """Caches the replies to `requests` through OpenRouter's batch API, so `complete` then finds them.

        For a `:batch` model, which OpenRouter serves only this way, at half the per-token price. A batch can
        take up to 24 hours. Its id is kept under the cache, so a rerun waits for the batch already sent
        instead of paying for another. Failed requests go again in a new batch, twice at most; a batch that
        expires unanswered raises instead, since another batch could wait as long again.
        """
        backend, model, reasoning = self.settings.target(role)
        if backend != "openai" or not model.endswith(":batch"):
            raise ValueError(f"{model} on the {backend} backend is not an OpenRouter :batch model")
        pending: dict[str, list[Message]] = {}
        for messages in requests:
            key = self._key(role, messages, reasoning)
            if self.cache.get(key) is None:
                pending[key] = messages
        extra = {"reasoning_effort": reasoning} if reasoning else {}
        headers = {"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"}
        errors: list[str] = []
        async with httpx.AsyncClient(base_url=f"{(self.settings.base_url or '').rstrip('/')}/", headers=headers, timeout=300) as http:
            for _ in range(_BATCH_ROUNDS):
                if not pending:
                    return
                keys = sorted(pending)
                record = CACHE / "llm-batches" / f"{request_key(keys)}.json"
                batch = await _get_batch(http, json.loads(record.read_text())["id"]) if record.exists() else None
                if batch is None or batch["status"] in _BATCH_DEAD:
                    # OpenRouter requires `endpoint` and `model` before `requests`.
                    body = {"endpoint": "/v1/chat/completions", "model": model, "requests": [
                        {"custom_id": f"r{index}", "body": {"messages": pending[key], **extra}} for index, key in enumerate(keys)
                    ]}
                    response = await http.post("batches", json=body)
                    if response.status_code >= 400:
                        raise LlmError(f"batch refused ({response.status_code}): {response.text[:300]}", retryable=False)
                    batch = response.json()
                    record.parent.mkdir(parents=True, exist_ok=True)
                    record.write_text(json.dumps({"id": batch["id"], "model": model}), encoding="utf-8")
                    print(f"{time.strftime('%H:%M')} sent batch {batch['id']}: {len(keys)} requests", file=sys.stderr, flush=True)
                batch = await _wait_for_batch(http, batch)
                errors = []
                for result in batch.get("results") or []:
                    key = keys[int(result["custom_id"][1:])]
                    response = result.get("response") or {}
                    if result.get("error") or response.get("status_code") != 200:
                        errors.append(str(result.get("error") or response)[:300])
                        continue
                    content = strip_think(response["body"]["choices"][0]["message"]["content"] or "")
                    self.cache.put(key, content, {"model": model, "batch": batch["id"]})
                    pending.pop(key, None)
                    self.calls += 1
                print(
                    f"{time.strftime('%H:%M')} batch {batch['id']} {batch['status']}: {len(keys) - len(pending)} of {len(keys)}"
                    f" answered, cost ${(batch.get('usage') or {}).get('cost')}",
                    file=sys.stderr, flush=True,
                )
                if batch["status"] == "expired" and pending:
                    raise LlmError(
                        f"batch {batch['id']} expired with {len(pending)} of {len(keys)} requests unanswered;"
                        " send them without :batch", retryable=False,
                    )
        if pending:
            raise LlmError(f"{len(pending)} batch requests failed; first error: {errors[:1]}", retryable=False)


    def _report_retry(self, backend: str, error: LlmError, attempt: int) -> None:
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


def _single_prompt(messages: list[Message], what: str) -> tuple[str, str]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    users = [m["content"] for m in messages if m["role"] == "user"]
    if len(users) != 1 or len(messages) != len(users) + (1 if system else 0):
        raise ValueError(f"the {what} backend sends one optional system message and one user message")
    return system, users[0]


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
