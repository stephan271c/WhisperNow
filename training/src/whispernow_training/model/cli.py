"""`uv run --extra train cleanup <command>`: train, sweep, evaluate, export and benchmark the local cleanup model."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tomllib
from pathlib import Path

from ..datagen.paths import TRAINING_ROOT

RUNS = TRAINING_ROOT / "runs"
CONFIGS = TRAINING_ROOT / "configs"
SWEEP_VAL_ROWS = 500


def cmd_train(args: argparse.Namespace) -> None:
    import os

    # Batches vary from a few tokens to 2,800 per row; expandable segments keep the long ones from
    # failing on fragmented memory. Set before torch first touches the GPU.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from . import config, train

    settings = config.load(args.config, args.set or [])
    summary = train.train(settings, RUNS / args.name)
    print(f"{args.name}: {summary['steps']} steps in {summary['seconds']} s, eval loss {summary['final_eval_loss']:.4f}")


def _sweep_runs(path: Path) -> tuple[Path, list[dict]]:
    sweep = tomllib.loads(path.read_text(encoding="utf-8"))
    return path.parent / sweep["config"], sweep["run"]


def cmd_sweep(args: argparse.Namespace) -> None:
    """Trains and scores each run in its own process, so GPU memory is released between runs."""
    base, runs = _sweep_runs(args.sweep)
    for run in runs:
        if args.only and run["name"] not in args.only:
            continue
        if not (RUNS / run["name"] / "train.json").exists():
            overrides = [item for value in run["set"] for item in ("--set", value)]
            subprocess.run([sys.argv[0], "train", str(base), "--name", run["name"], *overrides], check=True)
        if not any((RUNS / run["name"] / "eval").glob(f"val{SWEEP_VAL_ROWS}.*scores.json")):
            subprocess.run([sys.argv[0], "eval", run["name"], "--split", "val", "--limit", str(SWEEP_VAL_ROWS)], check=True)
        for checkpoint in sorted((RUNS / run["name"]).glob("epoch-*")):
            if not any((RUNS / run["name"] / f"eval-{checkpoint.name}").glob(f"val{SWEEP_VAL_ROWS}.*scores.json")):
                subprocess.run(
                    [sys.argv[0], "eval", run["name"], "--checkpoint", checkpoint.name, "--split", "val", "--limit", str(SWEEP_VAL_ROWS)],
                    check=True,
                )


def cmd_eval(args: argparse.Namespace) -> None:
    from . import evaluate, prompt

    overridden = args.repeat_penalty is not None
    if not overridden:
        args.repeat_penalty = prompt.REPEAT_PENALTY
    if args.reference:
        # Its outputs are the references, so new references get a directory of their own.
        tag = evaluate.reference_tag(evaluate.load_rows(args.split, args.limit))
        out_dir = RUNS / "reference" / (f"eval-{tag}" if tag else "eval")

        def produce(rows):
            return [{"id": row.id, "output": row.output, "hit_limit": False, "tokens": None} for row in rows]
    elif args.teacher:
        # `--label` keeps a new teacher prompt's outputs apart, as in `eval-112`.
        out_dir = RUNS / "teacher" / ("eval" if args.label == "server" else f"eval-{args.label}")

        def produce(rows):
            return asyncio.run(evaluate.generate_teacher(rows))
    elif args.server:
        out_dir = RUNS / args.run / f"eval-{args.label}"

        def produce(rows):
            return asyncio.run(evaluate.generate_server(args.server, rows, args.concurrency, args.repeat_penalty, args.repeat_last_n))
    elif args.checkpoint:
        out_dir = RUNS / args.run / f"eval-{args.checkpoint}"

        def produce(rows):
            return evaluate.generate_hf(RUNS / args.run / args.checkpoint, rows, repeat_penalty=args.repeat_penalty)
    else:
        out_dir = RUNS / args.run / "eval"

        def produce(rows):
            return evaluate.generate_hf(RUNS / args.run / "model", rows, repeat_penalty=args.repeat_penalty)

    if overridden and not (args.reference or args.teacher):
        out_dir = out_dir.with_name(f"{out_dir.name}-rp{args.repeat_penalty:g}")
    if args.repeat_last_n is not None and args.server:
        out_dir = out_dir.with_name(f"{out_dir.name}-last{args.repeat_last_n}")
    else:
        args.repeat_last_n = prompt.REPEAT_LAST_N
    scores = evaluate.evaluate(produce, out_dir, args.split, args.limit, with_judge=args.judge)
    overall = scores["overall"]
    def rate(value):
        if not isinstance(value, dict):
            return value
        if "rate" in value or "per_row" in value:
            # A rate, or a fault kind's count per judged row.
            return value.get("rate", value.get("per_row"))
        return {key: rate(inner) for key, inner in value.items()}

    print(json.dumps({key: rate(value) for key, value in overall.items()}))


def cmd_export(args: argparse.Namespace) -> None:
    from . import export

    for path in export.export(RUNS / args.run, args.quant):
        print(path)


def cmd_serve(args: argparse.Namespace) -> None:
    import os

    from . import export

    command = export.server_command(export.gguf_path(RUNS / args.run, args.quant), args.port, args.cpu, args.parallel, not args.no_draft)
    print(" ".join(command), file=sys.stderr)
    os.execv(command[0], command)


def cmd_build_server(args: argparse.Namespace) -> None:
    from . import export

    print(export.build_server(args.cpu))


def cmd_bench(args: argparse.Namespace) -> None:
    from . import bench

    if not (args.teacher or args.server):
        raise SystemExit("bench needs a server URL or --teacher")
    result = bench.bench_teacher(args.repeats) if args.teacher else bench.bench(args.server, args.repeats)
    print(json.dumps(result, indent=2))
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def cmd_handcheck(args: argparse.Namespace) -> None:
    """Prints a server's (or the teacher's) cleanup of each dictation in `data/real/`, beside its draft
    and reference, for reading by hand. These are personal: nothing is written, judged or scored."""
    from ..datagen.paths import DATA
    from ..datagen.schema import Example, read_jsonl
    from . import evaluate

    rows = [
        Example(id=row["id"], input=row["input"], output=row["reference"], source="real", category="real", license=row["license"])
        for row in read_jsonl(DATA / "real" / "real.jsonl")
    ]
    if args.teacher:
        outputs = asyncio.run(evaluate.generate_teacher(rows))
    elif args.server:
        outputs = asyncio.run(evaluate.generate_server(args.server, rows))
    else:
        raise SystemExit("handcheck needs a server URL or --teacher")
    for row, output in zip(rows, outputs, strict=True):
        flags = " [hit the reply limit]" if output["hit_limit"] else ""
        print(f"## {row.id} ({len(row.input.split())} words){flags}\n\nINPUT\n{row.input}\n\nREFERENCE\n{row.output}\n\nOUTPUT\n{output['output']}\n")


def cmd_prefer(args: argparse.Namespace) -> None:
    from . import prefer

    print(json.dumps(prefer.run(RUNS / args.run / "prefer", args.server, args.concurrency), indent=2))


def cmd_code_pairs(args: argparse.Namespace) -> None:
    from . import code_pairs

    print(json.dumps(code_pairs.run(RUNS / args.name, args.count, args.long_share), indent=2))


def cmd_sweep_table(args: argparse.Namespace) -> None:
    from . import report

    _, runs = _sweep_runs(args.sweep)
    print(report.sweep_table([run["name"] for run in runs], f"val{SWEEP_VAL_ROWS}"))


def cmd_judge_check(args: argparse.Namespace) -> None:
    from . import judge_check

    result = judge_check.report() if args.step == "report" else judge_check.run(args.step)
    print(json.dumps(result, indent=1))
    print(f"read every disagreement in {judge_check.OUT / 'disagreements.md'}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Scores side by side: `LABEL=RUN/EVAL_DIR`, e.g. `teacher=teacher/eval q4=full-inst-3e-5/eval-q4`."""
    from . import report

    candidates = {}
    for item in args.candidates:
        label, _, where = item.partition("=")
        candidates[label] = RUNS / where / f"{args.stem}.scores.json"
    groups = [None] if not args.group else args.group
    for group in groups:
        print(report.comparison_table(candidates, group))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(prog="cleanup", description=__doc__)
    commands = parser.add_subparsers(required=True)

    train_parser = commands.add_parser("train", help="fine-tune a model; writes runs/<name>/model and train.json")
    train_parser.add_argument("config", type=Path, help="a TOML file in configs/")
    train_parser.add_argument("--name", required=True, help="run name, the directory under runs/")
    train_parser.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value (repeatable)")
    train_parser.set_defaults(handler=cmd_train)

    sweep = commands.add_parser("sweep", help="train and score every run of a sweep file that is not done yet")
    sweep.add_argument("sweep", type=Path, help="a sweep file in configs/")
    sweep.add_argument("--only", action="append", help="only this run (repeatable)")
    sweep.set_defaults(handler=cmd_sweep)

    table = commands.add_parser("sweep-table", help="print a sweep's val scores as a markdown table")
    table.add_argument("sweep", type=Path)
    table.set_defaults(handler=cmd_sweep_table)

    eval_parser = commands.add_parser("eval", help="generate, check and (with --judge) judge a split")
    eval_parser.add_argument("run", nargs="?", default="teacher", help="run name under runs/")
    eval_parser.add_argument(
        "--split", default="test", choices=["train", "val", "test", "bench-long", "bench-noise"],
        help="a built split, or one of todo 110's benchmarks in data/bench/",
    )
    eval_parser.add_argument("--limit", type=int, help="score a fixed random sample of this many rows")
    eval_parser.add_argument("--judge", action="store_true", help="also judge with LLM_JUDGE_MODEL")
    eval_parser.add_argument("--server", help="OpenAI-compatible base URL of llama-server, e.g. http://127.0.0.1:8080/v1")
    eval_parser.add_argument("--label", default="server", help="names the eval-<label> directory for --server, or for --teacher")
    eval_parser.add_argument("--concurrency", type=int, default=4)
    eval_parser.add_argument("--checkpoint", help="an epoch checkpoint of the run, e.g. epoch-1; scores go to eval-<checkpoint>")
    eval_parser.add_argument(
        "--repeat-penalty", type=float, default=None, help="decode with this repeat penalty instead of the served one; scores go to <eval dir>-rp<value>"
    )
    eval_parser.add_argument(
        "--repeat-last-n", type=int, default=None, help="with --server: the penalty's window instead of the served one; scores go to <eval dir>-last<value>"
    )
    eval_parser.add_argument("--reference", action="store_true", help="score the split's own references, as a baseline")
    eval_parser.add_argument("--teacher", action="store_true", help="evaluate LLM_MODEL with the app's hosted prompt")
    eval_parser.set_defaults(handler=cmd_eval)

    export_parser = commands.add_parser("export", help="convert runs/<run>/model to GGUF and quantize")
    export_parser.add_argument("run")
    export_parser.add_argument("--quant", nargs="+", default=["Q8_0", "Q4_K_M"])
    export_parser.set_defaults(handler=cmd_export)

    serve = commands.add_parser("serve", help="run llama-server on an exported GGUF with the committed chat template")
    serve.add_argument("run")
    serve.add_argument("--quant", default="Q8_0", help="Q8_0 (chosen), Q4_K_M (faster on a CPU) or BF16")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--cpu", action="store_true", help="the CPU build, with no layers on the GPU")
    serve.add_argument("--parallel", type=int, default=1, help="request slots; evaluation uses 4")
    serve.add_argument("--no-draft", action="store_true", help="the release build, without drafting from the transcript")
    serve.set_defaults(handler=cmd_serve)

    build_server = commands.add_parser("build-server", help="build llama-server with the transcript-draft patch (serving/)")
    build_server.add_argument("--cpu", action="store_true", help="the CPU build, for `serve --cpu`")
    build_server.set_defaults(handler=cmd_build_server)

    bench_parser = commands.add_parser("bench", help="time short and long dictations against llama-server")
    bench_parser.add_argument("server", nargs="?", help="OpenAI-compatible base URL")
    bench_parser.add_argument("--teacher", action="store_true", help="time LLM_MODEL with the app's hosted prompt instead")
    bench_parser.add_argument("--repeats", type=int, default=5)
    bench_parser.add_argument("--save", type=Path, help="also write the result as JSON")
    bench_parser.set_defaults(handler=cmd_bench)

    handcheck = commands.add_parser("handcheck", help="print outputs for the dictations in data/real/, to read by hand")
    handcheck.add_argument("server", nargs="?", help="OpenAI-compatible base URL")
    handcheck.add_argument("--teacher", action="store_true", help="LLM_MODEL with the app's hosted prompt instead")
    handcheck.set_defaults(handler=cmd_handcheck)

    prefer_parser = commands.add_parser(
        "prefer", help="sample, judge and pair an SFT model's outputs for the preference stage; writes runs/<run>/prefer/"
    )
    prefer_parser.add_argument("run", help="the SFT run")
    prefer_parser.add_argument("--server", help="llama-server serving the run's Q8_0, for the sampling step")
    prefer_parser.add_argument("--concurrency", type=int, default=4)
    prefer_parser.set_defaults(handler=cmd_prefer)

    pairs_parser = commands.add_parser(
        "code-pairs",
        help="todo 114: DPO pairs made in code from train rows (the reference against one fact broken); "
        "writes runs/<name>/pairs.jsonl for --set train_file",
    )
    pairs_parser.add_argument("name", help="output directory under runs/")
    pairs_parser.add_argument("--count", type=int, default=8000)
    pairs_parser.add_argument("--long-share", type=float, default=0.7, help="share of pairs from rows of 150+ words")
    pairs_parser.set_defaults(handler=cmd_code_pairs)

    check = commands.add_parser(
        "judge-check",
        help="todo 112: the new eval judge on the 33 hand-reviewed rows (reviewed) and on stratified rows the old "
        "judge scored (sample), against the old references; needs LLM_JUDGE_* on codex/gpt-6.1-sol/high",
    )
    check.add_argument("step", choices=["reviewed", "sample", "all", "report"])
    check.set_defaults(handler=cmd_judge_check)

    compare = commands.add_parser("compare", help="print scores of several evaluations side by side")
    compare.add_argument("candidates", nargs="+", metavar="LABEL=RUN/EVAL_DIR")
    compare.add_argument("--stem", default="test", help="scores file stem, e.g. test or val500")
    compare.add_argument(
        "--group",
        action="append",
        choices=["commands", "category", "style", "source", "tag", "length", "pattern", "test_group"],
        help="per group (repeatable)",
    )
    compare.set_defaults(handler=cmd_compare)

    args = parser.parse_args()
    args.handler(args)
