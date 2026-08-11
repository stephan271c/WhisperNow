#!/usr/bin/env python3
"""Verify a Tauri base64-wrapped minisign signature with system minisign."""

from __future__ import annotations

import argparse
import base64
import binascii
import subprocess
import tempfile
from pathlib import Path


SYSTEM_MINISIGN = Path("/usr/bin/minisign")


def _decode_envelope(value: str, label: str) -> bytes:
    try:
        decoded = base64.b64decode(value.strip(), validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError(f"{label} is not a valid Tauri base64 envelope") from error
    if not decoded or b"\x00" in decoded:
        raise ValueError(f"{label} envelope is empty or malformed")
    return decoded


def verify(payload: Path, signature: Path, public_key: str) -> bool:
    if not payload.is_file() or not signature.is_file():
        return False
    if not SYSTEM_MINISIGN.is_file():
        raise ValueError("trusted system minisign is unavailable")
    try:
        signature_bytes = _decode_envelope(
            signature.read_text(encoding="utf-8"), "signature"
        )
        public_key_bytes = _decode_envelope(public_key, "public key")
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    with tempfile.TemporaryDirectory(prefix="whispernow-minisign-") as directory:
        root = Path(directory)
        raw_signature = root / "signature.minisig"
        raw_public_key = root / "public.key"
        raw_signature.write_bytes(signature_bytes)
        raw_public_key.write_bytes(public_key_bytes)
        try:
            result = subprocess.run(
                [
                    str(SYSTEM_MINISIGN),
                    "-V",
                    "-H",
                    "-m",
                    str(payload),
                    "-x",
                    str(raw_signature),
                    "-p",
                    str(raw_public_key),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
    return result.returncode == 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--payload", required=True, type=Path)
    root.add_argument("--signature", required=True, type=Path)
    root.add_argument("--public-key", required=True)
    return root


def main() -> int:
    arguments = parser().parse_args()
    try:
        valid = verify(arguments.payload, arguments.signature, arguments.public_key)
    except ValueError as error:
        parser().error(str(error))
    if not valid:
        parser().error("signature authentication failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
