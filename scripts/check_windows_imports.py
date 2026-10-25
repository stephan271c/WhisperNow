#!/usr/bin/env python3
"""Check that every DLL an installed Windows payload imports is there on a fresh Windows.

Each `.exe` and `.dll` under the payload is read for its import and delay-import tables.
An import is satisfied by a file in the importer's own directory: Windows searches the
executable's directory for the executable, and the engine opens `llama/`'s libraries
with LOAD_WITH_ALTERED_SEARCH_PATH, so theirs come from `llama/`. API sets (`api-ms-*`,
`ext-ms-*`) and DLLs in the system directory also satisfy it, except the Visual C++
runtime: the build runner has it installed, a fresh Windows does not. The Vulkan loader
comes with the GPU driver; without it ggml skips its Vulkan backend.
"""

from __future__ import annotations

import argparse
import os
import re
import struct
from pathlib import Path

VISUAL_CPP_RUNTIME = re.compile(r"^(msvcp|vcruntime|vcomp|concrt|vccorlib|mfc)\d", re.IGNORECASE)
HOST_PROVIDED = {"vulkan-1.dll"}
# (data directory index, descriptor size, offset of the DLL name's RVA in a descriptor)
IMPORT_TABLES = ((1, 20, 12), (13, 32, 4))


def imported_dlls(path: Path) -> list[str]:
    data = path.read_bytes()

    def u16(offset: int) -> int:
        return struct.unpack_from("<H", data, offset)[0]

    def u32(offset: int) -> int:
        return struct.unpack_from("<I", data, offset)[0]

    if data[:2] != b"MZ" or data[u32(0x3C) : u32(0x3C) + 4] != b"PE\0\0":
        raise ValueError(f"{path.name} is not a PE file")
    coff_offset = u32(0x3C) + 4
    optional_offset = coff_offset + 20
    directories_offset = optional_offset + {0x10B: 96, 0x20B: 112}.get(u16(optional_offset), 0)
    if directories_offset == optional_offset:
        raise ValueError(f"{path.name} has an unknown optional header")
    first_section_offset = optional_offset + u16(coff_offset + 16)
    sections = [
        (
            u32(section_offset + 12),
            u32(section_offset + 8) or u32(section_offset + 16),
            u32(section_offset + 20),
        )
        for section_offset in range(
            first_section_offset, first_section_offset + 40 * u16(coff_offset + 2), 40
        )
    ]

    def offset(rva: int) -> int:
        for address, size, raw_offset in sections:
            if address <= rva < address + size:
                return rva - address + raw_offset
        raise ValueError(f"{path.name} points outside its sections")

    names = []
    for index, descriptor_size, name_offset in IMPORT_TABLES:
        if index >= u32(directories_offset - 4) or u32(directories_offset + 8 * index) == 0:
            continue
        descriptor_offset = offset(u32(directories_offset + 8 * index))
        while u32(descriptor_offset + name_offset):
            start = offset(u32(descriptor_offset + name_offset))
            names.append(data[start : data.index(b"\0", start)].decode("ascii"))
            descriptor_offset += descriptor_size
    return names


def unresolved_imports(root: Path, system_dlls: set[str]) -> tuple[int, list[str]]:
    """How many PE files were checked, and each import a fresh Windows would not find."""
    checked, problems = 0, []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        # NSIS's own plugins, which 7-Zip extracts beside the payload, are not installed.
        if any(part.startswith("$") for part in relative.parts):
            continue
        if not path.is_file() or path.suffix.lower() not in (".exe", ".dll"):
            continue
        checked += 1
        beside = {sibling.name.lower() for sibling in path.parent.iterdir()}
        for name in imported_dlls(path):
            lower = name.lower()
            if (
                lower in beside
                or lower in HOST_PROVIDED
                or lower.startswith(("api-ms-", "ext-ms-"))
                or (lower in system_dlls and not VISUAL_CPP_RUNTIME.match(lower))
            ):
                continue
            problems.append(f"{relative.as_posix()} imports {name}, which is not beside it")
    return checked, problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="the installed (or extracted) payload")
    parser.add_argument(
        "--system-dir",
        type=Path,
        default=Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32",
    )
    args = parser.parse_args()
    try:
        system_dlls = {path.name.lower() for path in args.system_dir.iterdir()}
        checked, problems = unresolved_imports(args.root, system_dlls)
    except (OSError, ValueError, struct.error) as error:
        parser.error(str(error))
    if checked == 0:
        parser.error(f"no .exe or .dll under {args.root}")
    if problems:
        parser.error("\n".join(problems))
    print(f"every import of {checked} PE files resolves on a fresh Windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
