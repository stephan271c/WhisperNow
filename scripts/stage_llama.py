#!/usr/bin/env python3
r"""Download, verify, and extract the pinned llama.cpp release libraries the app loads.

The local cleanup model runs in process on llama.cpp's public C API. The app opens these
shared libraries at run time from its `llama/` resources (`Contents/Frameworks` on macOS).
Only the files `native-allowlist.json` lists under `shared_libraries.llama` are extracted,
each renamed to the name the loader expects (the library's soname), and each must match its
pinned size and SHA-256. Their license files are staged by `stage_native_dependencies.py`.

On Windows the release's DLLs import the Visual C++ runtime, which a fresh Windows lacks.
The allowlist names those DLLs under `msvc_runtime`; they are copied into `llama/`, where
the libraries' dependencies resolve, from the build's Visual Studio (`--msvc-runtime`, its
`VC\Redist\MSVC\<version>\x64\Microsoft.VC*.CRT`). They are Microsoft's signed builds,
so CI checks their signatures rather than pinning them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from stage_native_dependencies import DEFAULT_ALLOWLIST, PLATFORMS, fetch, sha256


def load_pins(path: Path, platform: str) -> tuple[dict, list[dict], list[str]]:
    """Return the archive, the libraries pinned for a platform, and its runtime DLLs."""
    document = json.loads(path.read_text(encoding="utf-8"))
    llama = document["shared_libraries"]["llama"]
    archive = llama["archives"].get(platform)
    libraries = llama["libraries"].get(platform)
    if archive is None or libraries is None:
        raise ValueError(f"native allowlist has no llama.cpp {platform!r} entry")
    runtime = llama.get("msvc_runtime", {}).get(platform, [])
    for name in [library["name"] for library in libraries] + runtime:
        path = PurePosixPath(name)
        if len(path.parts) != 1 or path.name in ("", ".", ".."):
            raise ValueError(f"llama.cpp library name is not a plain file name: {name}")
    return archive, libraries, runtime


def read_members(path: Path, members: set[str]) -> dict[str, bytes]:
    """The named regular files of a .tar.gz or .zip archive."""
    found: dict[str, bytes] = {}
    if path.name.endswith(".zip"):
        with zipfile.ZipFile(path) as bundle:
            for info in bundle.infolist():
                if info.filename in members and not info.is_dir():
                    found[info.filename] = bundle.read(info)
    else:
        with tarfile.open(path, mode="r:gz") as bundle:
            for member in bundle.getmembers():
                if member.name in members and member.isfile():
                    source = bundle.extractfile(member)
                    if source is not None:
                        found[member.name] = source.read()
    missing = members - found.keys()
    if missing:
        raise ValueError(f"llama.cpp archive lacks pinned files: {', '.join(sorted(missing))}")
    return found


def stage(
    archive_path: Path,
    libraries: list[dict],
    output: Path,
    runtime: list[str],
    runtime_dir: Path | None,
) -> None:
    contents = read_members(archive_path, {library["member"] for library in libraries})
    available = {path.name.lower(): path for path in runtime_dir.iterdir()} if runtime_dir else {}
    missing = [name for name in runtime if name.lower() not in available]
    if missing:
        raise ValueError(f"the Visual C++ runtime directory lacks {', '.join(missing)}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for library in libraries:
        destination = output / library["name"]
        destination.write_bytes(contents[library["member"]])
        if (
            destination.stat().st_size != library["bytes"]
            or sha256(destination) != library["sha256"]
        ):
            raise ValueError(
                f"llama.cpp library does not match the allowlist: {library['member']}"
            )
        destination.chmod(0o755)
    for name in runtime:
        shutil.copyfile(available[name.lower()], output / name)


def verify(directory: Path, libraries: list[dict], runtime: list[str]) -> None:
    """Check that an installed `llama/` holds exactly the pinned libraries and runtime DLLs."""
    present = sorted(path.name for path in directory.iterdir()) if directory.is_dir() else []
    if present != sorted([library["name"] for library in libraries] + runtime):
        raise ValueError(f"packaged llama.cpp libraries are not the exact pinned set: {present}")
    for library in libraries:
        path = directory / library["name"]
        if path.stat().st_size != library["bytes"] or sha256(path) != library["sha256"]:
            raise ValueError(f"packaged llama.cpp library does not match the allowlist: {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=PLATFORMS, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path, help="the directory to fill (replaced)")
    action.add_argument("--verify", type=Path, help="a packaged directory to check")
    parser.add_argument("--archive", type=Path, help="a local copy of the pinned archive")
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument(
        "--msvc-runtime", type=Path, help="the Visual C++ runtime DLLs to copy in (Windows)"
    )
    args = parser.parse_args()
    try:
        archive, libraries, runtime = load_pins(args.allowlist, args.platform)
        if args.verify is not None:
            verify(args.verify, libraries, runtime)
            return 0
        if bool(runtime) != (args.msvc_runtime is not None):
            raise ValueError(
                f"{args.platform} llama.cpp needs --msvc-runtime"
                if runtime
                else f"{args.platform} llama.cpp takes no --msvc-runtime"
            )
        with tempfile.TemporaryDirectory() as directory:
            downloaded = Path(directory) / archive["name"]
            if args.archive is None:
                fetch(archive["source"], downloaded)
            else:
                shutil.copyfile(args.archive, downloaded)
            if (
                downloaded.stat().st_size != archive["bytes"]
                or sha256(downloaded) != archive["sha256"]
            ):
                raise ValueError(
                    f"llama.cpp archive size or SHA-256 does not match the allowlist: {archive['name']}"
                )
            stage(downloaded, libraries, args.output, runtime, args.msvc_runtime)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, tarfile.TarError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
