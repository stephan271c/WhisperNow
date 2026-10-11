#!/usr/bin/env python3
"""Smoke-test the frontend command and URL configured for ``tauri dev``.

This deliberately tests the HTTP/bootstrap boundary only. A successful run
proves that Tauri's configured development command serves the compiled Dioxus
JavaScript and WebAssembly assets at its configured URL; it does not claim that
a native WebKit/WebView2 window rendered them.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from bounded_process import terminate_tree

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "apps" / "desktop" / "tauri.conf.json"
MAX_HTML_BYTES = 4 * 1024 * 1024
MAX_JAVASCRIPT_BYTES = 8 * 1024 * 1024
WASM_MAGIC = b"\x00asm"


class SmokeError(RuntimeError):
    pass


@dataclass(frozen=True)
class DevFrontend:
    command: str
    command_directory: Path
    request_url: str
    expected_title: str


class ShellParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_title = False
        self.title_parts: list[str] = []
        self.has_mount = False
        self.module_scripts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "title":
            self._in_title = True
        elif attributes.get("id") == "main":
            self.has_mount = True
        elif (
            tag == "script"
            and attributes.get("type") == "module"
            and attributes.get("src")
        ):
            self.module_scripts.append(attributes["src"] or "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)

    @property
    def title(self) -> str:
        return "".join(self.title_parts).strip()


def _is_loopback(hostname: str | None) -> bool:
    if hostname == "localhost":
        return True
    if hostname is None:
        return False
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def load_dev_frontend(path: Path) -> DevFrontend:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SmokeError(f"cannot read Tauri configuration: {error}") from error

    build = document.get("build")
    if not isinstance(build, dict):
        raise SmokeError("Tauri build configuration is missing")
    hook = build.get("beforeDevCommand")
    dev_url = build.get("devUrl")
    title = document.get("productName")
    command_directory = path.resolve().parent
    if isinstance(hook, str):
        command = hook
    elif isinstance(hook, dict):
        command = hook.get("script")
        cwd = hook.get("cwd")
        if cwd is not None:
            if not isinstance(cwd, str) or not cwd.strip():
                raise SmokeError("build.beforeDevCommand.cwd must be a directory")
            command_directory = (command_directory / cwd).resolve()
    else:
        command = None
    if not isinstance(command, str) or not command.strip():
        raise SmokeError("build.beforeDevCommand must be a non-empty command")
    if not command_directory.is_dir():
        raise SmokeError("build.beforeDevCommand.cwd must be an existing directory")
    if not isinstance(dev_url, str):
        raise SmokeError("build.devUrl must be a URL")
    if not isinstance(title, str) or not title.strip():
        raise SmokeError("productName must be a non-empty string")

    parsed = urlparse(dev_url)
    if parsed.scheme != "http" or not _is_loopback(parsed.hostname):
        raise SmokeError("build.devUrl must use HTTP on a loopback host")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SmokeError(
            "build.devUrl must not contain credentials, a query, or a fragment"
        )

    route = "/"
    app = document.get("app")
    windows = app.get("windows") if isinstance(app, dict) else None
    if isinstance(windows, list):
        settings = next(
            (
                window
                for window in windows
                if isinstance(window, dict) and window.get("label") == "settings"
            ),
            None,
        )
        if isinstance(settings, dict) and isinstance(settings.get("url"), str):
            route = settings["url"]
    route_url = urlparse(route)
    if route_url.scheme or route_url.netloc:
        raise SmokeError("the Settings window must use a path relative to build.devUrl")

    return DevFrontend(
        command=command,
        command_directory=command_directory,
        request_url=urljoin(f"{dev_url.rstrip('/')}/", route),
        expected_title=title.strip(),
    )


def _read_url(url: str, max_bytes: int) -> tuple[bytes, str]:
    request = Request(
        url,
        headers={"Accept": "text/html,*/*", "Cache-Control": "no-cache"},
    )
    try:
        with urlopen(request, timeout=2.0) as response:
            body = response.read(max_bytes + 1)
            content_type = response.headers.get_content_type()
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise SmokeError("configured development URL is not ready") from error
    if len(body) > max_bytes:
        raise SmokeError("development response exceeded the smoke-test size limit")
    return body, content_type


def _read_magic(url: str) -> bytes:
    request = Request(url, headers={"Cache-Control": "no-cache"})
    try:
        with urlopen(request, timeout=2.0) as response:
            return response.read(len(WASM_MAGIC))
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise SmokeError("compiled WebAssembly asset is not ready") from error


def _same_origin_url(base_url: str, candidate: str) -> str:
    resolved = urljoin(base_url, candidate)
    base = urlparse(base_url)
    asset = urlparse(resolved)
    if (asset.scheme, asset.hostname, asset.port) != (
        base.scheme,
        base.hostname,
        base.port,
    ):
        raise SmokeError("compiled frontend asset escaped the configured devUrl origin")
    return resolved


def inspect_served_frontend(frontend: DevFrontend) -> dict[str, str]:
    html_bytes, content_type = _read_url(frontend.request_url, MAX_HTML_BYTES)
    if content_type not in {"text/html", "application/xhtml+xml"}:
        raise SmokeError("configured development URL did not return HTML")
    try:
        html = html_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SmokeError("development HTML was not UTF-8") from error

    parser = ShellParser()
    parser.feed(html)
    if parser.title != frontend.expected_title:
        raise SmokeError(
            "development HTML did not contain the expected application title"
        )
    if not parser.has_mount:
        raise SmokeError("development HTML did not contain the Dioxus mount element")
    if not parser.module_scripts:
        raise SmokeError("development HTML did not load a compiled module")

    module_url = _same_origin_url(frontend.request_url, parser.module_scripts[-1])
    module, _ = _read_url(module_url, MAX_JAVASCRIPT_BYTES)
    try:
        module_text = module.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SmokeError("compiled frontend module was not UTF-8 JavaScript") from error
    wasm_matches = re.findall(
        r"[\"'`]([^\"'`]+\.wasm(?:\?[^\"'`]*)?)[\"'`]", module_text
    )
    if not wasm_matches:
        raise SmokeError("compiled frontend module did not reference WebAssembly")

    wasm_url = _same_origin_url(module_url, wasm_matches[0])
    if _read_magic(wasm_url) != WASM_MAGIC:
        raise SmokeError("compiled frontend WebAssembly asset had invalid magic bytes")
    return {"module_url": module_url, "wasm_url": wasm_url}


def run_smoke(frontend: DevFrontend, timeout_seconds: float) -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("CI", "true")
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(
            frontend.command,
            cwd=frontend.command_directory,
            env=environment,
            shell=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
        )
        deadline = time.monotonic() + timeout_seconds
        last_error = SmokeError("development server did not become ready")
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise SmokeError(
                        f"build.beforeDevCommand exited with status {process.returncode}"
                    )
                try:
                    return inspect_served_frontend(frontend)
                except SmokeError as error:
                    last_error = error
                    time.sleep(0.25)
            raise SmokeError(f"development server smoke timed out: {last_error}")
        except SmokeError as error:
            log.flush()
            log.seek(0)
            tail = log.read()[-12_000:].decode("utf-8", errors="replace").strip()
            if tail:
                raise SmokeError(
                    f"{error}\n--- beforeDevCommand log tail ---\n{tail}"
                ) from error
            raise
        finally:
            terminate_tree(process, grace_seconds=10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    try:
        frontend = load_dev_frontend(args.config)
        assets = run_smoke(frontend, args.timeout_seconds)
    except SmokeError as error:
        print(f"development frontend smoke failed: {error}", file=sys.stderr)
        return 1
    print(
        "development frontend smoke passed: "
        f"url={frontend.request_url} module={assets['module_url']} wasm={assets['wasm_url']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
