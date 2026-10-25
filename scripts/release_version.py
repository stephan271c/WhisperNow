#!/usr/bin/env python3
"""Check that Cargo, Tauri, updater, tag and artifact versions agree."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

STABLE_SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
SEMVER_IN_NAME = re.compile(
    r"(?<![0-9])(?:v)?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?![0-9])"
)


def parse_stable_tag(tag: str) -> str:
    if not tag.startswith("v"):
        raise ValueError("stable release tag must start with exactly one leading 'v'")
    version = tag[1:]
    if not STABLE_SEMVER.fullmatch(version):
        raise ValueError(
            "stable release tag must be vMAJOR.MINOR.PATCH without prerelease or build metadata"
        )
    return version


def workspace_version(manifest: Path) -> str:
    with manifest.open("rb") as source:
        document = tomllib.load(source)
    version = document.get("workspace", {}).get("package", {}).get("version")
    if not isinstance(version, str) or not STABLE_SEMVER.fullmatch(version):
        raise ValueError("Cargo workspace.package.version must be stable SemVer")
    return version


def cargo_metadata_versions(cargo: str, manifest: Path) -> dict[str, str]:
    completed = subprocess.run(
        [
            cargo,
            "metadata",
            "--format-version",
            "1",
            "--no-deps",
            "--manifest-path",
            str(manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(completed.stdout)
    workspace_members = set(document["workspace_members"])
    return {
        package["name"]: package["version"]
        for package in document["packages"]
        if package["id"] in workspace_members
    }


def validate_cargo_metadata(versions: dict[str, str], expected: str) -> list[str]:
    if not versions:
        return ["Cargo metadata contains no workspace packages"]
    return [
        f"Cargo metadata package {name} is {version}, expected {expected}"
        for name, version in sorted(versions.items())
        if version != expected
    ]


def validate_json_versions(
    paths: list[Path], expected: str, *, label: str, allow_missing: bool
) -> list[str]:
    errors: list[str] = []
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            errors.append(f"{path}: {error}")
            continue
        configured = document.get("version")
        if configured != expected and not (allow_missing and configured is None):
            errors.append(f"{path}: {label} version is {configured!r}, expected {expected}")
    return errors


def artifact_version(name: str) -> str | None:
    matches = list(SEMVER_IN_NAME.finditer(Path(name).name))
    if len(matches) != 1:
        return None
    match = matches[0]
    return ".".join(match.groups())


def validate_artifacts(names: list[str], expected: str) -> list[str]:
    errors: list[str] = []
    for name in names:
        version = artifact_version(name)
        if version is None:
            errors.append(f"{name}: artifact name must contain exactly one SemVer")
        elif version != expected:
            errors.append(f"{name}: artifact version is {version}, expected {expected}")
    return errors


def validate(
    *,
    manifest: Path,
    tauri_configs: list[Path],
    tag: str | None,
    metadata_versions: dict[str, str],
    updater_metadata: list[Path],
    artifacts: list[str],
) -> list[str]:
    try:
        expected = workspace_version(manifest)
    except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
        return [str(error)]
    errors = validate_cargo_metadata(metadata_versions, expected)
    if tag is not None:
        try:
            tagged = parse_stable_tag(tag)
        except ValueError as error:
            errors.append(str(error))
        else:
            if tagged != expected:
                errors.append(f"release tag version is {tagged}, expected {expected}")
    errors.extend(
        validate_json_versions(tauri_configs, expected, label="Tauri", allow_missing=True)
    )
    errors.extend(
        validate_json_versions(updater_metadata, expected, label="updater", allow_missing=False)
    )
    errors.extend(validate_artifacts(artifacts, expected))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Cargo, Tauri, updater, tag, and artifact versions without modifying files."
    )
    parser.add_argument("--manifest", type=Path, default=Path("Cargo.toml"))
    parser.add_argument("--tauri-config", action="append", type=Path, default=[])
    parser.add_argument("--release-tag")
    parser.add_argument("--updater-metadata", action="append", type=Path, default=[])
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--cargo", default="cargo")
    args = parser.parse_args()

    tauri_configs = args.tauri_config or sorted(
        args.manifest.parent.glob("apps/desktop/tauri*.conf.json")
    )
    try:
        metadata = cargo_metadata_versions(args.cargo, args.manifest)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as error:
        print(f"release version check failed: Cargo metadata: {error}", file=sys.stderr)
        return 1
    errors = validate(
        manifest=args.manifest,
        tauri_configs=tauri_configs,
        tag=args.release_tag,
        metadata_versions=metadata,
        updater_metadata=args.updater_metadata,
        artifacts=args.artifact,
    )
    if errors:
        print("release version check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"release version contract verified: {workspace_version(args.manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
