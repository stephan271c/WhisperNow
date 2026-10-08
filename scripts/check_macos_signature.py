#!/usr/bin/env python3
"""Check that a macOS app is signed with WhisperNow's pinned self-signed certificate.

The app bundle, each executable in `Contents/MacOS`, and each library in
`Contents/Frameworks` must be signed by exactly the certificate in
`release/macos-signing-certificate.pem`. The bundle's designated requirement must name
that certificate by hash: macOS keeps the user's privacy grants (Microphone,
Accessibility) for an update only while the requirement stays the
same, which an ad-hoc signature or another certificate would not.

Uses `codesign --display`, which reads signatures without evaluating trust, so it also
runs on a Mac that does not trust the certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import ssl
import subprocess
import tempfile
from pathlib import Path

REQUIREMENT = re.compile(r'^designated => .*\bcertificate (?:leaf|root) = H"([0-9a-fA-F]{40})"', re.MULTILINE)


def pinned_certificate(path: Path) -> bytes:
    return ssl.PEM_cert_to_DER_cert(path.read_text(encoding="ascii"))


def requirement_pins(display: str, certificate: bytes) -> bool:
    """Whether `codesign -d -r-` output's designated requirement names the certificate."""
    match = REQUIREMENT.search(display)
    return bool(match) and match.group(1).lower() == hashlib.sha1(certificate).hexdigest()


def signed_code(app: Path) -> list[Path]:
    """The bundle and the code inside it that must carry the pinned signature."""
    nested = [
        path
        for directory in ("MacOS", "Frameworks")
        for path in sorted((app / "Contents" / directory).glob("*"))
        if path.is_file()
    ]
    return [app, *nested]


def signer(path: Path) -> bytes | None:
    """The DER bytes of the certificate that signed `path`, or None for ad-hoc code."""
    with tempfile.TemporaryDirectory() as directory:
        prefix = Path(directory) / "certificate"
        subprocess.run(
            ["codesign", "--display", f"--extract-certificates={prefix}", str(path)],
            check=True,
            capture_output=True,
        )
        leaf = prefix.with_name("certificate0")
        return leaf.read_bytes() if leaf.exists() else None


def problems(app: Path, certificate: bytes) -> list[str]:
    found = []
    for path in signed_code(app):
        if signer(path) != certificate:
            found.append(f"{path.relative_to(app.parent)} is not signed with the pinned certificate")
    display = subprocess.run(
        ["codesign", "--display", "--requirements", "-", str(app)],
        check=True,
        capture_output=True,
        text=True,
    )
    if not requirement_pins(display.stdout + display.stderr, certificate):
        found.append(f"{app.name}'s designated requirement does not name the pinned certificate")
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("app", type=Path, help="the .app bundle")
    parser.add_argument("--certificate", type=Path, required=True, help="the pinned PEM certificate")
    args = parser.parse_args()
    try:
        certificate = pinned_certificate(args.certificate)
        found = problems(args.app, certificate)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    if found:
        parser.error("\n".join(found))
    print(f"{args.app.name} and its code are signed with the pinned certificate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
