"""GGUF export with llama.cpp, and the `llama-server` command that serves it.

llama.cpp lives in `.cache/llama.cpp/` (gitignored): the release source for `convert_hf_to_gguf.py`
in `src/`, and the release binaries in `cuda/` and `cpu/`. The README says how to fetch them.
`build_server` builds `llama-server` from that source with `serving/llama-server-draft-text.patch`
into `draft-text/`, for speculative decoding drafted from the transcript (model/draft.py).
"""

from __future__ import annotations

import hashlib
import os
import shutil
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
DRAFT_PATCH = prompt.SERVING / "llama-server-draft-text.patch"
DRAFT_BUILD = LLAMA_CPP / "draft-text"
# The patch's drafter rides on ngram-simple. 4 draft tokens a pass was fastest on GPU and CPU;
# ngram-simple's own lookup drafts when the text has nothing.
DRAFT_ARGS = ["--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "4", "--spec-ngram-simple-size-m", "4"]


def binary(name: str, cpu: bool = False) -> Path:
    path = LLAMA_CPP / ("cpu" if cpu else "cuda") / f"llama-{LLAMA_CPP_BUILD}" / name
    if not path.exists():
        raise SystemExit(f"{path} is missing; fetch llama.cpp {LLAMA_CPP_BUILD} as the README describes")
    return path


def draft_binary(name: str, cpu: bool = False) -> Path:
    path = DRAFT_BUILD / ("build-cpu" if cpu else "build-cuda") / "bin" / name
    if not path.exists():
        raise SystemExit(f"{path} is missing; build it with `cleanup build-server{' --cpu' if cpu else ''}`, or serve the release build with --no-draft")
    return path


def build_server(cpu: bool = False) -> Path:
    """Patches the release source and builds `llama-server`, with CUDA unless `cpu`.

    Bounded like the repo's Cargo builds (AGENTS.md): four jobs, four CPUs, low I/O priority.
    """
    patch = DRAFT_PATCH.read_bytes()
    source = DRAFT_BUILD / "src"
    marker = source / ".draft-text-patch"
    if not marker.exists() or marker.read_text() != hashlib.sha256(patch).hexdigest():
        shutil.rmtree(source, ignore_errors=True)
        shutil.copytree(LLAMA_CPP / "src", source, ignore=shutil.ignore_patterns("__pycache__", ".deja-dup-ignore"))
        subprocess.run(["patch", "-p1", "--quiet"], cwd=source, input=patch, check=True)
        marker.write_text(hashlib.sha256(patch).hexdigest())
    build = DRAFT_BUILD / ("build-cpu" if cpu else "build-cuda")
    cmake = ["cmake"] if shutil.which("cmake") else ["uvx", "--from", "cmake", "cmake"]
    env = dict(os.environ)
    if not cpu and not shutil.which("nvcc"):
        env["PATH"] = f"/usr/local/cuda/bin:{env['PATH']}"
    configure = [*cmake, "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF",
                 "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", f"-DGGML_CUDA={'OFF' if cpu else 'ON'}",
                 # no web UI: it would be downloaded, unpinned, from Hugging Face
                 "-DLLAMA_USE_PREBUILT_UI=OFF"]
    if not cpu:
        configure.append("-DCMAKE_CUDA_ARCHITECTURES=native")
    if shutil.which("ninja") and not (build / "CMakeCache.txt").exists():
        configure += ["-G", "Ninja"]
    subprocess.run(configure, check=True, env=env)
    bounded = ["systemd-run", "--user", "--scope", "-p", "CPUQuota=400%", "--", "ionice", "-c2", "-n7"] if shutil.which("systemd-run") else []
    subprocess.run([*bounded, *cmake, "--build", str(build), "--target", "llama-server", "-j", "4"], check=True, env=env)
    return draft_binary("llama-server", cpu)


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


def server_command(model: Path, port: int = 8080, cpu: bool = False, parallel: int = 1, draft: bool = False) -> list[str]:
    """`llama-server` with the committed chat template, so the GGUF's own template is never used.

    `draft` serves the patched build, which drafts from each request's `draft_text`.
    """
    return [
        str(draft_binary("llama-server", cpu) if draft else binary("llama-server", cpu)),
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
        *(DRAFT_ARGS if draft else []),
    ]
