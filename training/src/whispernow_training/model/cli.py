"""`uv run --extra train cleanup <command>`: train, evaluate, export and benchmark the local cleanup models."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from ..datagen.paths import TRAINING_ROOT

RUNS = TRAINING_ROOT / "runs"


def cmd_train(args: argparse.Namespace) -> None:
    import os

    # Batches vary from a few tokens to 2,800 per row; expandable segments keep the long ones from
    # failing on fragmented memory. Set before torch first touches the GPU.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from . import config, train

    settings = config.load(args.config, args.set or [])
    summary = train.train(settings, RUNS / args.name)
    print(f"{args.name}: {summary['steps']} steps in {summary['seconds']} s, eval loss {summary['final_eval_loss']:.4f}")


def cmd_eval(args: argparse.Namespace) -> None:
    from . import evaluate, prompt

    overridden = args.repeat_penalty is not None
    if not overridden:
        args.repeat_penalty = prompt.REPEAT_PENALTY
    if args.reference:
        # Its outputs are the references, so new references get a directory of their own.
        tag = evaluate.reference_tag(evaluate.load_rows(args.split, args.limit))
        out_dir = RUNS / "reference" / (f"eval-{tag}" if tag else "eval")
        generation = {"source": "reference"}

        def produce(rows):
            return [{"id": row.id, "output": row.output, "hit_limit": False, "tokens": None} for row in rows]
    elif args.teacher:
        # `--label` keeps a new teacher prompt's outputs apart.
        out_dir = RUNS / "teacher" / ("eval" if args.label == "server" else f"eval-{args.label}")
        generation = evaluate.teacher_generation()

        def produce(rows):
            return asyncio.run(evaluate.generate_teacher(rows))
    elif args.server:
        out_dir = RUNS / args.run / f"eval-{args.label}"
        repeat_last_n = prompt.REPEAT_LAST_N if args.repeat_last_n is None else args.repeat_last_n
        generation = evaluate.server_generation(args.server, args.repeat_penalty, repeat_last_n)

        def produce(rows):
            return asyncio.run(evaluate.generate_server(args.server, rows, args.concurrency, args.repeat_penalty, repeat_last_n))
    elif args.checkpoint:
        out_dir = RUNS / args.run / f"eval-{args.checkpoint}"
        generation = evaluate.hf_generation(RUNS / args.run / args.checkpoint, args.repeat_penalty)

        def produce(rows):
            return evaluate.generate_hf(RUNS / args.run / args.checkpoint, rows, repeat_penalty=args.repeat_penalty)
    else:
        out_dir = RUNS / args.run / "eval"
        generation = evaluate.hf_generation(RUNS / args.run / "model", args.repeat_penalty)

        def produce(rows):
            return evaluate.generate_hf(RUNS / args.run / "model", rows, repeat_penalty=args.repeat_penalty)

    if overridden and not (args.reference or args.teacher):
        out_dir = out_dir.with_name(f"{out_dir.name}-rp{args.repeat_penalty:g}")
    if args.repeat_last_n is not None and args.server:
        out_dir = out_dir.with_name(f"{out_dir.name}-last{args.repeat_last_n}")
    else:
        args.repeat_last_n = prompt.REPEAT_LAST_N
    scores = evaluate.evaluate(produce, out_dir, args.split, args.limit, with_judge=args.judge, generation=generation)
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


def cmd_rescore(args: argparse.Namespace) -> None:
    from . import evaluate

    for where in args.eval_dirs:
        print(evaluate.rescore(RUNS / where, args.split, args.limit))


def cmd_rejudge(args: argparse.Namespace) -> None:
    from . import evaluate

    ids = args.ids.read_text(encoding="utf-8").split()
    for where in args.eval_dirs:
        print(evaluate.rejudge(RUNS / where, args.split, ids))


def cmd_compare(args: argparse.Namespace) -> None:
    """Scores side by side: `LABEL=RUN/EVAL_DIR`, e.g. `teacher=teacher/eval q4=full-inst-3e-5/eval-q4`.
    Reads an evaluation's rescored file when it has one, and refuses scores of different checks."""
    from . import evaluate, report

    loaded = {}
    for item in args.candidates:
        label, _, where = item.partition("=")
        rescored = evaluate.rescored_path(RUNS / where, args.stem)
        path = rescored if rescored.exists() else RUNS / where / f"{args.stem}.scores.json"
        loaded[label] = json.loads(path.read_text(encoding="utf-8"))
    versions = {label: scores.get("metrics_version", 1) for label, scores in loaded.items()}
    if len(set(versions.values())) > 1:
        listed = ", ".join(f"{label} {version}" for label, version in versions.items())
        raise SystemExit(
            f"the scores use different metric versions ({listed}); "
            f"`cleanup rescore` the older evaluations to version {evaluate.METRICS_VERSION}"
        )
    groups = [None] if not args.group else args.group
    for group in groups:
        print(report.comparison_table(loaded, group))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(prog="cleanup", description=__doc__)
    commands = parser.add_subparsers(required=True)

    train_parser = commands.add_parser("train", help="fine-tune a model; writes runs/<name>/model and train.json")
    train_parser.add_argument("config", type=Path, help="a TOML file in configs/")
    train_parser.add_argument("--name", required=True, help="run name, the directory under runs/")
    train_parser.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value (repeatable)")
    train_parser.set_defaults(handler=cmd_train)

    from . import evaluate

    eval_parser = commands.add_parser("eval", help="generate, check and (with --judge) judge a split")
    eval_parser.add_argument("run", nargs="?", default="teacher", help="run name under runs/")
    eval_parser.add_argument(
        "--split", default="test", choices=evaluate.SPLITS,
        help="a split of the dataset, a fixed sample of one, or a benchmark in data/bench/",
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

    rescore_parser = commands.add_parser(
        "rescore", help="score saved outputs and judge verdicts again with the current checks, without generating or judging"
    )
    rescore_parser.add_argument("eval_dirs", nargs="+", metavar="RUN/EVAL_DIR", help="under runs/, e.g. teacher/eval-112")
    rescore_parser.add_argument("--split", default="test", choices=evaluate.SPLITS)
    rescore_parser.add_argument("--limit", type=int, help="the sample size `eval --limit` scored")
    rescore_parser.set_defaults(handler=cmd_rescore)

    rejudge_parser = commands.add_parser(
        "rejudge", help="judge chosen rows' saved outputs again with the current judge, then rescore"
    )
    rejudge_parser.add_argument("eval_dirs", nargs="+", metavar="RUN/EVAL_DIR", help="under runs/, e.g. teacher/eval-112")
    rejudge_parser.add_argument("--split", default="test", choices=evaluate.SPLITS)
    rejudge_parser.add_argument("--ids", type=Path, required=True, help="a file of row ids, one per line")
    rejudge_parser.set_defaults(handler=cmd_rejudge)

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

    compare = commands.add_parser("compare", help="print scores of several evaluations side by side")
    compare.add_argument("candidates", nargs="+", metavar="LABEL=RUN/EVAL_DIR")
    compare.add_argument("--stem", default="test", help="scores file stem, e.g. test or test.r112.faults")
    compare.add_argument(
        "--group",
        action="append",
        choices=["commands", "category", "style", "source", "tag", "length", "pattern", "test_group"],
        help="per group (repeatable)",
    )
    compare.set_defaults(handler=cmd_compare)

    args = parser.parse_args()
    args.handler(args)
