#!/usr/bin/env python3
"""Download, verify, and extract the pinned static sherpa-onnx archive used by Cargo.

It also stages the pinned native license files (and any source offers) into
`native-compliance.zip`, which is published with each release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

PLATFORMS = ("linux-x86_64", "macos-aarch64", "macos-x86_64", "windows-x86_64")
DEFAULT_ALLOWLIST = Path(__file__).parents[1] / "release" / "native-allowlist.json"
DEFAULT_LICENSES = Path(__file__).parents[1] / "release" / "native-licenses"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matches_pin(path: Path, pin: dict) -> bool:
    return path.stat().st_size == pin["bytes"] and sha256(path) == pin["sha256"]


def load_allowlist(path: Path, platform: str) -> tuple[dict, dict, dict, list[dict]]:
    """Return the components, archive, license files, and libraries pinned for a platform."""
    document = json.loads(path.read_text(encoding="utf-8"))
    archive = document["source_archives"].get(platform)
    libraries = document["libraries"].get(platform)
    if archive is None or libraries is None:
        raise ValueError(f"native allowlist has no {platform!r} entry")
    return (
        document["components"],
        archive,
        document["license_evidence"],
        sorted(libraries, key=lambda entry: entry["name"]),
    )


def validate_archive(path: Path, archive: dict) -> None:
    if not matches_pin(path, archive):
        raise ValueError(
            f"native archive size or SHA-256 does not match the allowlist: {path.name}"
        )


def fetch(source: str, destination: Path) -> None:
    with urlopen(source) as response, destination.open("wb") as output:  # noqa: S310
        shutil.copyfileobj(response, output)


def safe_members(bundle: tarfile.TarFile, expected_root: str) -> list[tarfile.TarInfo]:
    members = bundle.getmembers()
    if not members:
        raise ValueError("native archive is empty")
    for member in members:
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[0] != expected_root
            or member.issym()
            or member.islnk()
            or member.isdev()
        ):
            raise ValueError(f"native archive contains unsafe entry: {member.name}")
    return members


def extract_archive(path: Path, output: Path) -> Path:
    root_name = path.name.removesuffix(".tar.bz2")
    extracted = output / root_name
    if extracted.exists():
        shutil.rmtree(extracted)
    with tarfile.open(path, mode="r:bz2") as bundle:
        members = safe_members(bundle, root_name)
        bundle.extractall(output, members=members, filter="data")
    lib_dir = extracted / "lib"
    if not lib_dir.is_dir() or not any(lib_dir.iterdir()):
        raise ValueError("native archive has no non-empty lib directory")
    return lib_dir


def stage_compliance(
    components: dict, license_files: dict, source: Path, output: Path
) -> Path:
    """Zip the pinned license files and source offers into `native-compliance.zip`."""
    members = [(f"licenses/{name}", name, pin) for name, pin in license_files.items()]
    members += [
        (f"source/{offer['name']}", offer["name"], offer)
        for component in components.values()
        if (offer := component.get("source_offer")) is not None
    ]
    bundle = output / "native-compliance.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_STORED) as archive:
        for member, name, pin in sorted(members, key=lambda item: item[0]):
            path = source / name
            check_pinned(path, pin)
            info = zipfile.ZipInfo(member, (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    return bundle


def check_pinned(path: Path, pin: dict) -> None:
    if not path.is_file():
        raise ValueError(f"pinned native license file is missing: {path.name}")
    if not matches_pin(path, pin):
        raise ValueError(f"native license file does not match the allowlist: {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=PLATFORMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--license-dir", type=Path, default=DEFAULT_LICENSES)
    args = parser.parse_args()
    try:
        components, archive, license_files, expected_libraries = load_allowlist(
            args.allowlist, args.platform
        )
        args.output.mkdir(parents=True, exist_ok=True)
        destination = args.output / archive["name"]
        if args.archive is None:
            with tempfile.NamedTemporaryFile(
                dir=args.output, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                fetch(archive["source"], temporary_path)
                temporary_path.replace(destination)
            finally:
                temporary_path.unlink(missing_ok=True)
        else:
            shutil.copy2(args.archive, destination)
        validate_archive(destination, archive)
        lib_dir = extract_archive(destination, args.output)
        stage_compliance(components, license_files, args.license_dir, args.output)
    except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError) as error:
        parser.error(str(error))

    libraries = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(lib_dir.iterdir())
        if path.is_file()
    ]
    if libraries != expected_libraries:
        parser.error("extracted static libraries do not match the trusted allowlist")
    # The workflow exports this as SHERPA_ONNX_LIB_DIR.
    print(lib_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
