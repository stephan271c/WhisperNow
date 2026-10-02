"""GGUF export with llama.cpp, and the `llama-server` command that serves it.

llama.cpp lives in `.cache/llama.cpp/` (gitignored): the release source for `convert_hf_to_gguf.py`
in `src/`, and the release binaries in `cuda/` and `cpu/`. The README says how to fetch them.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ..datagen.paths import CACHE
from . import prompt

LLAMA_CPP = CACHE / "llama.cpp"
LLAMA_CPP_BUILD = "b11259"
# Tokens per request slot: a 1,000-word dictation takes about 1,400 prompt tokens with the
# instructions, and its reply may take 2,500 (`prompt.max_tokens`).
CTX_PER_SLOT = 8192


def binary(name: str, cpu: bool = False) -> Path:
    path = LLAMA_CPP / ("cpu" if cpu else "cuda") / f"llama-{LLAMA_CPP_BUILD}" / name
    if not path.exists():
        raise SystemExit(f"{path} is missing; fetch llama.cpp {LLAMA_CPP_BUILD} as the README describes")
    return path


def gguf_path(run_dir: Path, quant: str) -> Path:
    return run_dir / "gguf" / f"model-{quant}.gguf"


def export(run_dir: Path, quants: list[str]) -> list[Path]:
    """Converts `run_dir/model` to a BF16 GGUF, then quantizes it to each of `quants`."""
    full = gguf_path(run_dir, "BF16")
    full.parent.mkdir(parents=True, exist_ok=True)
    if not full.exists():
        subprocess.run(
            [
                sys.executable,
                str(LLAMA_CPP / "src" / "convert_hf_to_gguf.py"),
                str(run_dir / "model"),
                "--outfile",
                str(full),
                "--outtype",
                "bf16",
            ],
            check=True,
            env={**os.environ, "PYTHONPATH": str(LLAMA_CPP / "src" / "gguf-py")},
        )
    made = [full]
    for quant in quants:
        target = gguf_path(run_dir, quant)
        if not target.exists():
            subprocess.run([str(binary("llama-quantize")), str(full), str(target), quant], check=True)
        made.append(target)
    return made


def server_command(model: Path, port: int = 8080, cpu: bool = False, parallel: int = 1) -> list[str]:
    """`llama-server` with the committed chat template, so the GGUF's own template is never used."""
    return [
        str(binary("llama-server", cpu)),
        "--model",
        str(model),
        "--jinja",
        "--chat-template-file",
        str(prompt.TEMPLATE_FILE),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(CTX_PER_SLOT * parallel),
        "--parallel",
        str(parallel),
        "--n-gpu-layers",
        "0" if cpu else "99",
    ]
