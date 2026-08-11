#!/usr/bin/env python3
"""Fail unless the active compiler and every workspace package use Rust 1.96."""

from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]
TOOLCHAIN = "1.96.0"
MSRV = "1.96"


def main() -> int:
    toolchain = tomllib.loads((ROOT / "rust-toolchain.toml").read_text(encoding="utf-8"))
    workspace = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    if toolchain.get("toolchain", {}).get("channel") != TOOLCHAIN:
        raise SystemExit(f"rust-toolchain.toml must pin {TOOLCHAIN}")
    if workspace.get("workspace", {}).get("package", {}).get("rust-version") != MSRV:
        raise SystemExit(f"workspace rust-version must be {MSRV}")
    version = subprocess.run(
        ["rustc", "--version"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    if not version.startswith(f"rustc {TOOLCHAIN} "):
        raise SystemExit(f"active compiler must be rustc {TOOLCHAIN}, found {version}")
    metadata = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    )
    members = set(metadata["workspace_members"])
    mismatches = sorted(
        package["name"]
        for package in metadata["packages"]
        if package["id"] in members and package.get("rust_version") != MSRV
    )
    if mismatches:
        raise SystemExit(
            f"workspace packages must declare rust-version {MSRV}: {', '.join(mismatches)}"
        )
    print(f"verified rustc {TOOLCHAIN} and workspace MSRV {MSRV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
