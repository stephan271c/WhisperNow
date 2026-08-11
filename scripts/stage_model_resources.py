#!/usr/bin/env python3
"""Verify the pinned Silero VAD resources bundled by Tauri (size, SHA-256, license)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

MODEL_FILENAME = "silero_vad.onnx"
LICENSE_FILENAME = "LICENSE-MIT"
MANIFEST_FILENAME = "manifest.json"
RESOURCE_NAMES = (MODEL_FILENAME, LICENSE_FILENAME, MANIFEST_FILENAME)
EXPECTED_MODEL = {
    "name": "Silero VAD",
    "version": "v4",
    "filename": MODEL_FILENAME,
    "bytes": 643854,
    "sha256": "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
    "license": "MIT",
    "license_file": LICENSE_FILENAME,
    "license_bytes": 1075,
    "license_sha256": "2e63e9a38b6e8fc0c7bc37ce174caca1862870856c6daf5697cfb785e925520b",
    "source": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
    "upstream_project": "https://github.com/snakers4/silero-vad",
    "upstream_license_snapshot": "https://github.com/snakers4/silero-vad/blob/76e3dc408eb2a5c655c34e230d2d5459b4439daa/LICENSE",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_and_validate(resource_dir: Path) -> dict:
    if not resource_dir.is_dir():
        raise ValueError(f"model resource directory does not exist: {resource_dir}")
    entries = list(resource_dir.iterdir())
    actual = {entry.name for entry in entries if entry.is_file()}
    if any(not entry.is_file() for entry in entries):
        raise ValueError("model resource directory contains a non-file entry")
    if actual != set(RESOURCE_NAMES):
        raise ValueError(
            "model resource directory must contain the exact pinned file set"
        )
    try:
        manifest = json.loads(
            (resource_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"model resource manifest is invalid: {error}") from error
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "model"}:
        raise ValueError("model resource manifest must contain the exact field set")
    if manifest.get("schema_version") != 1 or manifest.get("model") != EXPECTED_MODEL:
        raise ValueError(
            "model resource manifest does not match the pinned Silero VAD identity"
        )
    for field in ("source", "upstream_project", "upstream_license_snapshot"):
        parsed = urlparse(EXPECTED_MODEL[field])
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError(f"model resource {field} must be an HTTPS URL")
    model_path = resource_dir / MODEL_FILENAME
    license_path = resource_dir / LICENSE_FILENAME
    if model_path.stat().st_size != EXPECTED_MODEL["bytes"]:
        raise ValueError("Silero VAD model size does not match the pinned manifest")
    if digest(model_path) != EXPECTED_MODEL["sha256"]:
        raise ValueError("Silero VAD model SHA-256 does not match the pinned manifest")
    if license_path.stat().st_size != EXPECTED_MODEL["license_bytes"]:
        raise ValueError("Silero VAD license size does not match the pinned manifest")
    if digest(license_path) != EXPECTED_MODEL["license_sha256"]:
        raise ValueError(
            "Silero VAD license SHA-256 does not match the pinned manifest"
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a VAD resource directory (source tree or extracted package)."
    )
    parser.add_argument(
        "--resource-dir",
        type=Path,
        default=Path(__file__).parents[1] / "resources" / "vad",
    )
    args = parser.parse_args()
    try:
        load_and_validate(args.resource_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"verified pinned VAD resources: {args.resource_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
