"""`uv run datagen <command>`: one command per stage of the dataset build."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from . import aawaaz, build, compose, epochs, generate, paths, real, stage1, stage2, teacher
from .llm import LlmClient, LlmSettings, require_r112
from .schema import read_examples, write_examples


def _print_counts(title: str, counts: Counter) -> None:
    print(title)
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")


def cmd_import_aawaaz(_args: argparse.Namespace) -> None:
    examples, dropped = aawaaz.import_rows(aawaaz.load_rows())
    count = write_examples(paths.STAGE1 / "aawaaz.imported.jsonl", examples)
    print(f"wrote {count} rows to {paths.STAGE1 / 'aawaaz.imported.jsonl'}")
    _print_counts("dropped:", dropped)
    _print_counts("held out as test:", Counter(e.category for e in examples if e.split == "test"))


def cmd_sync_teacher(_args: argparse.Namespace) -> None:
    teacher.sync_from_rust()
    print(f"copied the system contract and cleanup prompt into {teacher.TEACHER_DIR}")


async def _judge_aawaaz(args: argparse.Namespace) -> None:
    examples = read_examples(paths.STAGE1 / "aawaaz.imported.jsonl")
    if args.limit:
        examples = examples[: args.limit]
    async with LlmClient(LlmSettings.from_env()) as llm:
        verdicts = await stage1.judge_aawaaz(llm, examples, batch_size=args.batch_size)
    flagged = Counter(v.failed[0] for v in verdicts.values() if not v.passed)
    print(f"{len(verdicts)} judged, {sum(not v.passed for v in verdicts.values())} flagged")
    _print_counts("first failed criterion:", flagged)


async def _judge_register(_args: argparse.Namespace) -> None:
    examples = read_examples(paths.STAGE1 / "aawaaz.imported.jsonl")
    async with LlmClient(LlmSettings.from_env()) as llm:
        verdicts = await stage1.judge_register(llm, examples)
    flagged = Counter(e.category for e in examples if e.id in verdicts and not verdicts[e.id].passed)
    print(f"{len(verdicts)} judged, {sum(flagged.values())} flagged")
    _print_counts("flagged per category:", flagged)


async def _teacher_check(args: argparse.Namespace) -> None:
    async with LlmClient(LlmSettings.from_env()) as llm:
        report = await stage1.teacher_agreement(llm, args.n, args.label)
    print(json.dumps(report["summary"], indent=2))


async def _fix_aawaaz(_args: argparse.Namespace) -> None:
    async with LlmClient(LlmSettings.from_env()) as llm:
        report = await stage1.fix_flagged(llm)
    print(json.dumps(report, indent=2))


async def _restyle(args: argparse.Namespace) -> None:
    from . import restyle

    if args.recheck:
        print(json.dumps(restyle.recheck(), indent=2))
        return
    async with LlmClient(LlmSettings.from_env()) as llm:
        report = await restyle.restyle_all(llm, args.limit)
    print(json.dumps(report, indent=2))


async def _add(args: argparse.Namespace) -> None:
    async with LlmClient(LlmSettings.from_env()) as llm:
        report = await stage2.add_sources(llm, args.with_sotto)
    print(json.dumps(report, indent=2))


async def _generate(args: argparse.Namespace) -> None:
    categories = args.category or list(generate.TARGETS)
    async with LlmClient(LlmSettings.from_env()) as llm:
        counts = await generate.generate(llm, categories, args.scale, args.batch_size, args.test)
    _print_counts("rows per category:", Counter(counts))


async def _compose(args: argparse.Namespace) -> None:
    run = compose.RUNS[args.run_name]
    categories = args.category or list(compose.CATEGORIES)
    async with LlmClient(LlmSettings.from_env()) as llm:
        counts = await compose.generate_all(llm, run, args.total, categories)
    _print_counts(f"{run.name} rows per category and bucket:", Counter(counts))
    print(json.dumps({"rejected": compose.rejections(run)}, indent=1))


async def _bench(args: argparse.Namespace) -> None:
    from . import bench, noise

    if args.step == "probe":
        print(f"{noise.probe_audio()} rows spoken and transcribed into {noise.PROBED}")
        return
    if args.step == "assemble":
        print(json.dumps(bench.assemble(), indent=1))
        return
    async with LlmClient(LlmSettings.from_env()) as llm:
        if args.step == "ami":
            report = await bench.label_ami(llm)
        elif args.step == "noise":
            report = await noise.generate_all(llm, args.kind or noise.KINDS)
        else:
            report = await noise.gate_probed(llm)
    print(json.dumps(report, indent=1))


async def _relabel(args: argparse.Namespace) -> None:
    from . import relabel

    scopes = relabel.SCOPES if args.step == "all" else (args.step,)
    if not args.report:
        settings = LlmSettings.from_env()
        require_r112(settings, "teacher", "judge", "generator")
        async with LlmClient(settings) as llm:
            for scope in scopes:
                if args.retry:
                    report = await relabel.retry(
                        llm, scope, relabel.inputs(scope), round_=args.retry, settled_hedges=args.settled_hedges
                    )
                    print(json.dumps(report, indent=1))
                else:
                    print(json.dumps(await relabel.run(llm, scope, relabel.inputs(scope)), indent=1))
    for scope in scopes:
        if relabel.active(scope):
            report = relabel.scope_report(scope)
            (relabel.DIR / f"{scope}.report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
            print(relabel.compare_markdown(report, f"{scope}, old against new"))


def cmd_profile(args: argparse.Namespace) -> None:
    from . import inject, profile
    from .schema import read_jsonl

    groups: dict[str, list[str]] = {}
    if real.OUT.exists():
        # Only the real dictations' rates are printed, never their text.
        groups["real, parakeet, over 30 words"] = [
            row["input"] for row in read_jsonl(real.OUT)
            if "parakeet" in row.get("asr_model_id", "") and len(row["input"].split()) > 30
        ]
    train = read_examples(paths.BUILD / "train.jsonl") if (paths.BUILD / "train.jsonl").exists() else []
    for label, keep in (
        ("aawaaz", lambda e: e.source == aawaaz.REPO),
        ("teacher-generated", lambda e: e.source == generate.SOURCE),
        ("teacher-generated, recognizer", lambda e: e.source == generate.SOURCE and "style_recognizer" in e.tags),
        ("injected, recognizer", lambda e: e.source == inject.SOURCE and "style_recognizer" in e.tags),
        ("injected, raw", lambda e: e.source == inject.SOURCE and "style_raw" in e.tags),
    ):
        groups[f"train: {label}"] = [e.input for e in train if keep(e)]
    if args.injection:
        rows = inject.injected_rows(read_examples(stage1.CLEAN), build.INJECTED, build.SEED)
        for style in inject.STYLES:
            groups[f"injection now, {style}"] = [e.input for e in rows if f"style_{style}" in e.tags]
    for path in args.file or []:
        groups[str(path)] = [row["input"] for row in read_jsonl(path)]
    print(profile.table({label: profile.profile(texts) for label, texts in groups.items() if texts}))


def cmd_build(args: argparse.Namespace) -> None:
    report = build.build(with_sotto=args.with_sotto)
    print(build.stats_markdown(report))
    print(f"wrote splits and stats to {paths.BUILD}")


def cmd_epochs(args: argparse.Namespace) -> None:
    for path in epochs.materialize(args.epochs):
        print(path)
    print((epochs.EPOCHS / "stats.json").read_text(encoding="utf-8"))


def cmd_sample(args: argparse.Namespace) -> None:
    print(build.review_sample(args.layout, args.n, args.seed))


def cmd_asr_probe(args: argparse.Namespace) -> None:
    from . import asr_probe

    if not args.report and args.model == "parakeet":
        written = asr_probe.run(args.n, args.seed, keep_audio=not args.no_audio, shard=args.shard, shards=args.shards)
        print(f"{written} new rows in {asr_probe.OUT_DIR}")
    elif not args.report:
        written = asr_probe.redecode(args.model, shard=args.shard, shards=args.shards)
        print(f"{written} new {args.model} rows in {asr_probe.OUT_DIR}")
    report = asr_probe.report(asr_probe.load(args.model))
    (asr_probe.OUT_DIR / f"report.{args.model}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))


def cmd_asr_speed(args: argparse.Namespace) -> None:
    from . import asr_speed

    print(json.dumps(asr_speed.run(args.model or None), indent=1))


def cmd_publish(args: argparse.Namespace) -> None:
    build.publish(args.repo)
    print(f"pushed {paths.BUILD} splits to the private dataset {args.repo}")


async def _export_real(args: argparse.Namespace) -> None:
    history = args.history or real.default_history()
    if args.no_draft:
        count = await real.export_real(None, history, args.limit)
    else:
        async with LlmClient(LlmSettings.from_env()) as llm:
            count = await real.export_real(llm, history, args.limit)
    print(f"{count} rows in {real.OUT}; write each reference by hand, starting from the draft")


def main() -> None:
    parser = argparse.ArgumentParser(prog="datagen", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("import-aawaaz", help="stage 1: import and normalize aawaaz").set_defaults(
        run=cmd_import_aawaaz
    )
    commands.add_parser("sync-teacher", help="copy the teacher prompt from the Rust sources").set_defaults(
        run=cmd_sync_teacher
    )
    judge = commands.add_parser("judge-aawaaz", help="stage 1: flag rows below the standard")
    judge.add_argument("--limit", type=int, default=0, help="judge only the first N rows")
    judge.add_argument("--batch-size", type=int, default=15)
    judge.set_defaults(run=lambda args: asyncio.run(_judge_aawaaz(args)))
    commands.add_parser(
        "judge-register", help=f"stage 1: flag {', '.join(stage1.REGISTER_CATEGORIES)} rows that change register"
    ).set_defaults(run=lambda args: asyncio.run(_judge_register(args)))
    check = commands.add_parser("teacher-check", help="stage 1: compare the teacher with aawaaz references")
    check.add_argument("--n", type=int, default=50)
    check.add_argument("--label", default="current", help="name of this prompt revision in the report")
    check.set_defaults(run=lambda args: asyncio.run(_teacher_check(args)))
    commands.add_parser("fix-aawaaz", help="stage 1: relabel flagged rows, re-judge, drop failures").set_defaults(
        run=lambda args: asyncio.run(_fix_aawaaz(args))
    )

    add = commands.add_parser("add", help="stage 2: add handy and disfl_qa rows, audited by the judge")
    add.add_argument(
        "--with-sotto",
        action="store_true",
        help="also add sotto rows; part of sotto is Grok output, which xAI's terms bar from training",
    )
    add.set_defaults(run=lambda args: asyncio.run(_add(args)))

    gen = commands.add_parser("generate", help="stage 3: generate gap and hard-case rows; resumes where it stopped")
    gen.add_argument("--category", action="append", choices=generate.TARGETS, help="only this category (repeatable)")
    gen.add_argument("--scale", type=float, default=1.0, help="multiply every category's target, e.g. 0.05 for a pilot")
    gen.add_argument("--batch-size", type=int, help="transcripts per request (default: 50, 25 for long_ramble)")
    gen.add_argument("--test", action="store_true", help="generate the extra rows pinned to the test split")
    gen.set_defaults(run=lambda args: asyncio.run(_generate(args)))

    restyle_parser = commands.add_parser(
        "restyle", help="step 7: rewrite half the aawaaz inputs in recognizer style, words unchanged"
    )
    restyle_parser.add_argument("--limit", type=int, default=0, help="restyle only the first N chosen rows")
    restyle_parser.add_argument("--recheck", action="store_true", help="check the rejected rewrites again; no LLM calls")
    restyle_parser.set_defaults(run=lambda args: asyncio.run(_restyle(args)))

    compose_parser = commands.add_parser(
        "compose", help="long composed-task rows (train) or the long benchmark's generated rows (bench); resumes"
    )
    compose_parser.add_argument("--run", dest="run_name", choices=["train", "bench"], default="train")
    compose_parser.add_argument("--total", type=int, required=True, help="rows over all categories, split 40/35/25 by bucket")
    compose_parser.add_argument("--category", action="append", choices=compose.CATEGORIES, help="only this category (repeatable)")
    compose_parser.set_defaults(run=lambda args: asyncio.run(_compose(args)))

    bench_parser = commands.add_parser(
        "bench",
        help="the held-out benchmarks: ami (label AMI turns), noise (generate), probe (Kokoro and Parakeet, "
        "needs --extra audio), gate-probed, assemble (write data/bench/*.jsonl); the long run's generated rows "
        "come from `compose --run bench`",
    )
    bench_parser.add_argument("step", choices=["ami", "noise", "probe", "gate-probed", "assemble"])
    bench_parser.add_argument("--kind", action="append", help="noise: only this pattern (repeatable)")
    bench_parser.set_defaults(run=lambda args: asyncio.run(_bench(args)))

    relabel_parser = commands.add_parser(
        "relabel",
        help="rewrite every reference with gpt-6.1-sol at high reasoning (every role; the command refuses "
        "any other setting): build, bench-long, bench-noise or all; resumable",
    )
    relabel_parser.add_argument("step", choices=["build", "bench-long", "bench-noise", "all"])
    relabel_parser.add_argument("--report", action="store_true", help="only report old against new from the records")
    relabel_parser.add_argument(
        "--retry", nargs="?", const=1, type=int, metavar="ROUND",
        help="retry round ROUND (default 1): check the dropped rows again and give each that still fails one "
        "more rewrite with the judge's note. ROUND must exceed every earlier round",
    )
    relabel_parser.add_argument(
        "--settled-hedges", action="store_true",
        help="with --retry: also check the passed rows whose reference keeps a hedge the speaker may have settled "
        "soon after (\"maybe some jasmine rice, actually yeah get it\"), and rewrite the ones that fail",
    )
    relabel_parser.set_defaults(run=lambda args: asyncio.run(_relabel(args)))

    profile_parser = commands.add_parser(
        "profile", help="filler formatting and correction cues of the real dictations, the build and given files"
    )
    profile_parser.add_argument("file", nargs="*", type=Path, help="more JSONL files with an input field")
    profile_parser.add_argument("--injection", action="store_true", help="also profile the current injection code")
    profile_parser.set_defaults(run=cmd_profile)

    build_parser = commands.add_parser("build", help="filter, deduplicate, split by seed group, and report stats")
    build_parser.add_argument("--with-sotto", action="store_true", help="include sotto rows if stage 2 added them")
    build_parser.set_defaults(run=cmd_build)

    epochs_parser = commands.add_parser(
        "epochs", help="todo 111: one freshly noised variant of the train split per epoch, for noise_per_epoch training"
    )
    epochs_parser.add_argument("--epochs", type=int, default=5)
    epochs_parser.set_defaults(run=cmd_epochs)

    sample = commands.add_parser("sample", help="print random built rows of one layout for a hand review")
    sample.add_argument("layout", choices=["plain", "paragraphs", "list", "mixed"])
    sample.add_argument("--n", type=int, default=50)
    sample.add_argument("--seed", type=int, default=110)
    sample.set_defaults(run=cmd_sample)

    probe = commands.add_parser(
        "asr-probe", help="speak sampled inputs with Kokoro and transcribe them as the app does (needs --extra audio)"
    )
    probe.add_argument(
        "--model", choices=("parakeet", "whisper", "qwen3"), default="parakeet",
        help="parakeet speaks and transcribes; another model transcribes the audio parakeet's run kept",
    )
    probe.add_argument("--n", type=int, default=300)
    probe.add_argument("--seed", type=int, default=108)
    probe.add_argument("--no-audio", action="store_true", help="do not keep the synthesized wav files")
    probe.add_argument("--shard", type=int, default=0, help="run the rows at positions shard, shard+shards, ...")
    probe.add_argument("--shards", type=int, default=1, help="how many probe processes share the rows")
    probe.add_argument("--report", action="store_true", help="only rewrite report.<model>.json from the rows already probed")
    probe.set_defaults(run=cmd_asr_probe)
    speed = commands.add_parser(
        "asr-speed", help="time each catalog speech model on the probe audio and score its WER (needs --extra audio)"
    )
    speed.add_argument("--model", action="append", help="measure only this model; repeatable")
    speed.set_defaults(run=cmd_asr_speed)

    publish = commands.add_parser("publish", help="push the built splits to a private Hugging Face dataset")
    publish.add_argument("repo", help="dataset repo id, such as your-name/whispernow-cleanup")
    publish.set_defaults(run=cmd_publish)

    export = commands.add_parser("export-real", help="real-dictation test set from the local history (never uploaded)")
    export.add_argument("--history", type=Path, help="history.sqlite3 path; defaults to the app's data dir")
    export.add_argument("--limit", type=int, default=300)
    export.add_argument("--no-draft", action="store_true", help="skip the teacher drafts")
    export.set_defaults(run=lambda args: asyncio.run(_export_real(args)))

    args = parser.parse_args()
    if args.command == "relabel" and args.settled_hedges and not args.retry:
        relabel_parser.error("--settled-hedges requires --retry with a round above every earlier round")
    args.run(args)
