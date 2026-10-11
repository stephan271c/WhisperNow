#!/usr/bin/env python3
"""Run one command with a portable deadline and process-tree termination."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


TIMEOUT_EXIT = 124
MISSING_EXECUTABLE_EXIT = 127


def terminate_tree(process: subprocess.Popen[bytes], grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=max(grace_seconds, 1.0),
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
    try:
        process.wait(timeout=max(grace_seconds, 1.0))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_bounded(
    command: list[str],
    *,
    timeout_seconds: float,
    grace_seconds: float,
    cwd: Path | None,
    label: str,
) -> int:
    if not command:
        print(f"bounded process {label}: missing command", file=sys.stderr)
        return MISSING_EXECUTABLE_EXIT
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=os.name != "nt",
            creationflags=creationflags,
        )
    except (FileNotFoundError, NotADirectoryError):
        print(f"bounded process {label}: executable unavailable", file=sys.stderr)
        return MISSING_EXECUTABLE_EXIT
    except OSError:
        print(f"bounded process {label}: launch failed", file=sys.stderr)
        return MISSING_EXECUTABLE_EXIT
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        terminate_tree(process, grace_seconds)
        print(f"bounded process {label}: deadline exceeded", file=sys.stderr)
        return TIMEOUT_EXIT
    if return_code == 0:
        print(f"bounded process {label}: completed")
        return 0
    if return_code < 0:
        print(
            f"bounded process {label}: terminated by signal {-return_code}",
            file=sys.stderr,
        )
        return 128 - return_code
    else:
        print(
            f"bounded process {label}: exited nonzero ({return_code})", file=sys.stderr
        )
    return return_code if 0 < return_code < 256 else 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--timeout-seconds", type=float, required=True)
    result.add_argument("--grace-seconds", type=float, default=5.0)
    result.add_argument("--cwd", type=Path)
    result.add_argument("--label", required=True)
    result.add_argument("command", nargs=argparse.REMAINDER)
    return result


def main() -> int:
    args = parser().parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.timeout_seconds <= 0 or args.grace_seconds <= 0:
        parser().error("timeouts must be positive")
    if not args.label or any(character in args.label for character in "\r\n"):
        parser().error("label must be non-empty and single-line")
    return run_bounded(
        command,
        timeout_seconds=args.timeout_seconds,
        grace_seconds=args.grace_seconds,
        cwd=args.cwd,
        label=args.label,
    )


if __name__ == "__main__":
    raise SystemExit(main())
