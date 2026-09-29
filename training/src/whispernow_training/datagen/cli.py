"""`uv run datagen <command>`: one command per stage of the dataset build."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from . import aawaaz, build, generate, paths, real, stage1, stage2, teacher
from .judge import judge_all, pair_item
from .llm import LlmClient, LlmSettings
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
    print(f"copied the system contract and fix_grammar prompt into {teacher.TEACHER_DIR}")


async def _judge_aawaaz(args: argparse.Namespace) -> None:
    examples = read_examples(paths.STAGE1 / "aawaaz.imported.jsonl")
    if args.limit:
        examples = examples[: args.limit]
    async with LlmClient(LlmSettings.from_env()) as llm:
        verdicts = await judge_all(
            llm, [pair_item(e) for e in examples], stage1.JUDGMENTS, batch_size=args.batch_size
        )
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
    unknown = sorted(set(categories) - set(generate.TARGETS))
    if unknown:
        raise SystemExit(f"unknown categories {unknown}; choose from {sorted(generate.TARGETS)}")
    async with LlmClient(LlmSettings.from_env()) as llm:
        counts = await generate.generate(llm, categories, args.scale, args.batch_size)
    _print_counts("rows per category:", Counter(counts))


def cmd_build(args: argparse.Namespace) -> None:
    report = build.build(with_sotto=args.with_sotto)
    print(build.stats_markdown(report))
    print(f"wrote splits and stats to {paths.BUILD}")


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
    gen.add_argument("--category", action="append", help="only this category (repeatable)")
    gen.add_argument("--scale", type=float, default=1.0, help="multiply every category's target, e.g. 0.05 for a pilot")
    gen.add_argument("--batch-size", type=int, help="transcripts per request (default: 50, 25 for long_ramble)")
    gen.set_defaults(run=lambda args: asyncio.run(_generate(args)))

    restyle_parser = commands.add_parser(
        "restyle", help="step 7: rewrite half the aawaaz inputs in recognizer style, words unchanged"
    )
    restyle_parser.add_argument("--limit", type=int, default=0, help="restyle only the first N chosen rows")
    restyle_parser.add_argument("--recheck", action="store_true", help="check the rejected rewrites again; no LLM calls")
    restyle_parser.set_defaults(run=lambda args: asyncio.run(_restyle(args)))

    build_parser = commands.add_parser("build", help="filter, deduplicate, split by seed group, and report stats")
    build_parser.add_argument("--with-sotto", action="store_true", help="include sotto rows if stage 2 added them")
    build_parser.set_defaults(run=cmd_build)

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

    publish = commands.add_parser("publish", help="push the built splits to a private Hugging Face dataset")
    publish.add_argument("repo", help="dataset repo id, such as your-name/whispernow-cleanup")
    publish.set_defaults(run=cmd_publish)

    export = commands.add_parser("export-real", help="real-dictation test set from the local history (never uploaded)")
    export.add_argument("--history", type=Path, help="history.sqlite3 path; defaults to the app's data dir")
    export.add_argument("--limit", type=int, default=300)
    export.add_argument("--no-draft", action="store_true", help="skip the teacher drafts")
    export.set_defaults(run=lambda args: asyncio.run(_export_real(args)))

    args = parser.parse_args()
    args.run(args)
