#!/usr/bin/env python3
"""Stage signed WhisperNow packages and generate the updater `latest.json`.

`stage` runs once per platform on the bundle runner: it verifies each updater
payload's detached signature, copies the packages under canonical names, and
writes a small `manifest.json`. `generate` runs once in the release job: it
combines the four platform manifests into `latest.json`, which carries the
signatures, so the `.sig` files are not published.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib.parse import quote

from stage_native_dependencies import sha256

STABLE_TAG = re.compile(r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

@dataclass(frozen=True)
class Target:
    platform: str
    bundle_type: str
    glob: str
    suffix: str
    label: str
    installer_glob: str | None = None
    installer_suffix: str | None = None


TARGETS = {
    "windows-x86_64": Target(
        "windows-x86_64", "nsis", "*.exe", "windows_x86_64_nsis.exe", "Windows NSIS installer"
    ),
    "linux-x86_64-deb": Target(
        "linux-x86_64", "deb", "*.deb", "linux_x86_64.deb", "Linux deb package"
    ),
    "linux-x86_64-rpm": Target(
        "linux-x86_64", "rpm", "*.rpm", "linux_x86_64.rpm", "Linux rpm package"
    ),
    "linux-x86_64-appimage": Target(
        "linux-x86_64", "appimage", "*.AppImage", "linux_x86_64.AppImage", "Linux AppImage"
    ),
    "darwin-aarch64": Target(
        "macos-aarch64", "app", "*.app.tar.gz", "darwin_aarch64.app.tar.gz",
        "macOS updater archive", "*.dmg", "darwin_aarch64.dmg"
    ),
    "darwin-x86_64": Target(
        "macos-x86_64", "app", "*.app.tar.gz", "darwin_x86_64.app.tar.gz",
        "macOS updater archive", "*.dmg", "darwin_x86_64.dmg"
    ),
}
REQUIRED_TARGETS = {name: target.bundle_type for name, target in TARGETS.items()}
PLATFORM_TARGETS = {
    platform: tuple(name for name, target in TARGETS.items() if target.platform == platform)
    for platform in sorted({target.platform for target in TARGETS.values()})
}

SignatureVerifier = Callable[[Path, Path], bool]


@dataclass(frozen=True)
class Selection:
    target: str
    updater: Path
    installer: Path


def validate_identity(repository: str, tag: str, version: str) -> None:
    if not STABLE_TAG.fullmatch(tag) or tag[1:] != version:
        raise ValueError("tag and version must be the same stable vMAJOR.MINOR.PATCH")
    if not REPOSITORY.fullmatch(repository):
        raise ValueError("repository must be an owner/name GitHub repository")


def canonical_package_name(version: str, target: str, *, installer: bool) -> str:
    if target not in TARGETS:
        raise ValueError(f"unsupported release target: {target}")
    entry = TARGETS[target]
    suffix = entry.installer_suffix if installer and entry.installer_suffix else entry.suffix
    return f"WhisperNow_{version}_{suffix}"


def signature_text(path: Path) -> str:
    """Return the base64 minisign envelope Tauri writes to `*.sig` files."""
    try:
        encoded = path.read_text(encoding="utf-8").strip()
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (OSError, UnicodeDecodeError, binascii.Error) as error:
        raise ValueError(f"detached signature is malformed: {path.name}") from error
    lines = decoded.splitlines()
    if len(lines) != 4 or not lines[0].startswith("untrusted comment: "):
        raise ValueError(f"detached signature is malformed: {path.name}")
    return encoded


def subprocess_signature_verifier(executable: Path, public_key: str) -> SignatureVerifier:
    if not public_key.strip():
        raise ValueError("updater public key is required")
    if not executable.is_file():
        raise ValueError(f"signature verifier executable is missing: {executable}")

    def verify(payload: Path, signature: Path) -> bool:
        try:
            result = subprocess.run(
                [
                    str(executable),
                    "--payload",
                    str(payload),
                    "--signature",
                    str(signature),
                    "--public-key",
                    public_key,
                ],
                stdin=subprocess.DEVNULL,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    return verify


def one(root: Path, pattern: str, label: str) -> Path:
    matches = sorted(path for path in root.rglob(pattern) if path.is_file())
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {label}, found {len(matches)}")
    return matches[0]


def select_platform(root: Path, platform: str) -> list[Selection]:
    if platform not in PLATFORM_TARGETS:
        raise ValueError(f"unsupported release platform: {platform}")
    selections = []
    for target in PLATFORM_TARGETS[platform]:
        entry = TARGETS[target]
        updater = one(root, entry.glob, entry.label)
        installer = (
            one(root, entry.installer_glob, f"{platform} installer")
            if entry.installer_glob
            else updater
        )
        selections.append(Selection(target, updater, installer))
    return selections


def asset_record(path: Path) -> dict:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def stage(
    *,
    root: Path,
    output: Path,
    platform: str,
    version: str,
    signature_verifier: SignatureVerifier,
) -> Path:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"staging output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    assets: dict[str, dict] = {}
    updates: list[dict] = []
    for selection in select_platform(root, platform):
        updater_name = canonical_package_name(version, selection.target, installer=False)
        signature = Path(f"{selection.updater}.sig")
        signature_text(signature)
        if not signature_verifier(selection.updater, signature):
            raise ValueError(f"updater signature verification failed: {selection.updater.name}")
        shutil.copyfile(signature, output / f"{updater_name}.sig")
        for source, installer in ((selection.updater, False), (selection.installer, True)):
            name = canonical_package_name(version, selection.target, installer=installer)
            if name not in assets:
                shutil.copyfile(source, output / name)
                assets[name] = asset_record(output / name)
        updates.append(
            {
                "target": selection.target,
                "bundle_type": REQUIRED_TARGETS[selection.target],
                "file": updater_name,
                "signature_file": f"{updater_name}.sig",
            }
        )
    manifest = {
        "platform": platform,
        "version": version,
        "assets": sorted(assets.values(), key=lambda entry: entry["name"]),
        "updates": sorted(updates, key=lambda entry: entry["target"]),
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def parse_pub_date(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("publication date must be RFC 3339") from error
    if not value.endswith("Z") or parsed.utcoffset() is None:
        raise ValueError("publication date must be an absolute RFC 3339 UTC timestamp")


def generate(
    *,
    manifests: list[Path],
    output: Path,
    repository: str,
    tag: str,
    version: str,
    pub_date: str,
    notes: str,
    signature_verifier: SignatureVerifier,
) -> Path:
    validate_identity(repository, tag, version)
    parse_pub_date(pub_date)
    if not notes.strip() or len(notes.encode("utf-8")) > 64 * 1024:
        raise ValueError("release notes must be non-empty and at most 64 KiB")
    platforms: set[str] = set()
    assets: dict[str, tuple[dict, Path]] = {}
    updates: dict[str, tuple[dict, Path]] = {}
    for manifest_path in manifests:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        platform = document["platform"]
        if document["version"] != version:
            raise ValueError(f"{manifest_path}: version does not match {version}")
        if platform in platforms:
            raise ValueError(f"duplicate platform manifest: {platform}")
        platforms.add(platform)
        for entry in document["assets"]:
            name = entry["name"]
            source = manifest_path.parent / name
            if name in assets or Path(name).name != name:
                raise ValueError(f"duplicate or unsafe release asset name: {name}")
            if asset_record(source) != entry:
                raise ValueError(f"release asset bytes do not match its manifest: {name}")
            assets[name] = (entry, source)
        for entry in document["updates"]:
            if entry["target"] in updates:
                raise ValueError(f"duplicate updater target: {entry['target']}")
            updates[entry["target"]] = (entry, manifest_path.parent / entry["signature_file"])
    if platforms != set(PLATFORM_TARGETS):
        raise ValueError(f"release platforms are incomplete: {sorted(platforms)}")
    if set(updates) != set(REQUIRED_TARGETS):
        raise ValueError(f"updater targets are incomplete: {sorted(updates)}")

    output.mkdir(parents=True, exist_ok=True)
    for name, (_entry, source) in assets.items():
        shutil.copyfile(source, output / name)
    base_url = f"https://github.com/{repository}/releases/download/{tag}"
    platforms_document: dict[str, dict] = {}
    for target, (entry, signature) in sorted(updates.items()):
        payload = output / entry["file"]
        if not signature_verifier(payload, signature):
            raise ValueError(f"updater signature verification failed: {target}")
        platforms_document[target] = {
            "signature": signature_text(signature),
            "url": f"{base_url}/{quote(entry['file'])}",
        }
    latest = {
        "version": version,
        "notes": notes,
        "pub_date": pub_date,
        "platforms": platforms_document,
    }
    latest_path = output / "latest.json"
    latest_path.write_text(json.dumps(latest, indent=2) + "\n", encoding="utf-8")
    return latest_path


def _verifier(args: argparse.Namespace) -> SignatureVerifier:
    return subprocess_signature_verifier(args.signature_verifier, args.public_key)


def command_stage(args: argparse.Namespace) -> None:
    print(
        stage(
            root=args.root,
            output=args.output,
            platform=args.platform,
            version=args.version,
            signature_verifier=_verifier(args),
        )
    )


def command_generate(args: argparse.Namespace) -> None:
    print(
        generate(
            manifests=args.manifest,
            output=args.output,
            repository=args.repository,
            tag=args.tag,
            version=args.version,
            pub_date=args.pub_date,
            notes=args.notes_file.read_text(encoding="utf-8"),
            signature_verifier=_verifier(args),
        )
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(required=True)

    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("--root", required=True, type=Path)
    stage_parser.add_argument("--output", required=True, type=Path)
    stage_parser.add_argument("--platform", required=True, choices=sorted(PLATFORM_TARGETS))
    stage_parser.add_argument("--version", required=True)
    stage_parser.set_defaults(function=command_stage)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--manifest", required=True, action="append", type=Path)
    generate_parser.add_argument("--output", required=True, type=Path)
    generate_parser.add_argument("--repository", required=True)
    generate_parser.add_argument("--tag", required=True)
    generate_parser.add_argument("--version", required=True)
    generate_parser.add_argument("--pub-date", required=True)
    generate_parser.add_argument("--notes-file", required=True, type=Path)
    generate_parser.set_defaults(function=command_generate)

    for subparser in (stage_parser, generate_parser):
        subparser.add_argument("--signature-verifier", required=True, type=Path)
        subparser.add_argument("--public-key", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        args.function(args)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser().error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
