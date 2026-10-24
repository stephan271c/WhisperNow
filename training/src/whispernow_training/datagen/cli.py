"""`uv run datagen <command>`: one command per step of the dataset, in the order the README gives them."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from . import paths, real, teacher
from .dataset import build, round1_base, round3_weak_spots, round4_cutoffs_cues, stats
from .llm import LlmClient, LlmSettings, require_reference_target
from .references import review
from .schema import read_examples, write_examples
from .sources import aawaaz, aawaaz_audit, public_audit
from .synthetic import compose, generate


def _print_counts(title: str, counts: Counter) -> None:
    print(title)
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")


def _reference_settings(*roles: str) -> LlmSettings:
    settings = LlmSettings.from_env()
    require_reference_target(settings, *roles)
    return settings


def cmd_sync_teacher(_args: argparse.Namespace) -> None:
    teacher.sync_from_rust()
    print(f"copied the system contract and cleanup prompt into {teacher.TEACHER_DIR}")


def cmd_import_aawaaz(_args: argparse.Namespace) -> None:
    examples, dropped = aawaaz.import_rows(aawaaz.load_rows())
    count = write_examples(aawaaz_audit.IMPORTED, examples)
    print(f"wrote {count} rows to {aawaaz_audit.IMPORTED}")
    _print_counts("dropped:", dropped)
    _print_counts("held out as test:", Counter(e.category for e in examples if e.split == "test"))


async def _judge_aawaaz(args: argparse.Namespace) -> None:
    examples = read_examples(aawaaz_audit.IMPORTED)
    if args.limit:
        examples = examples[: args.limit]
    async with LlmClient(LlmSettings.from_env()) as llm:
        verdicts = await aawaaz_audit.judge_aawaaz(llm, examples, batch_size=args.batch_size)
    flagged = Counter(v.failed[0] for v in verdicts.values() if not v.passed)
    print(f"{len(verdicts)} judged, {sum(not v.passed for v in verdicts.values())} flagged")
    _print_counts("first failed criterion:", flagged)


async def _judge_register(_args: argparse.Namespace) -> None:
    examples = read_examples(aawaaz_audit.IMPORTED)
    async with LlmClient(LlmSettings.from_env()) as llm:
        verdicts = await aawaaz_audit.judge_register(llm, examples)
    flagged = Counter(e.category for e in examples if e.id in verdicts and not verdicts[e.id].passed)
    print(f"{len(verdicts)} judged, {sum(flagged.values())} flagged")
    _print_counts("flagged per category:", flagged)


async def _teacher_check(args: argparse.Namespace) -> None:
    async with LlmClient(LlmSettings.from_env()) as llm:
        report = await aawaaz_audit.teacher_agreement(llm, args.n, args.label)
    print(json.dumps(report["summary"], indent=2))


async def _fix_aawaaz(_args: argparse.Namespace) -> None:
    async with LlmClient(LlmSettings.from_env()) as llm:
        report = await aawaaz_audit.fix_flagged(llm)
    print(json.dumps(report, indent=2))


async def _add_public(args: argparse.Namespace) -> None:
    async with LlmClient(LlmSettings.from_env()) as llm:
        report = await public_audit.add_sources(llm, args.with_sotto)
    print(json.dumps(report, indent=2))


async def _generate(args: argparse.Namespace) -> None:
    categories = args.category or list(generate.RUN_TARGETS[args.run_name] if args.run_name else generate.TARGETS)
    async with LlmClient(LlmSettings.from_env()) as llm:
        counts = await generate.generate(llm, categories, args.scale, args.batch_size, args.test, args.run_name or "")
    _print_counts("rows per category:", Counter(counts))


async def _restyle(args: argparse.Namespace) -> None:
    from .synthetic import restyle

    if args.recheck:
        print(json.dumps(restyle.recheck(), indent=2))
        return
    async with LlmClient(LlmSettings.from_env()) as llm:
        report = await restyle.restyle_all(llm, args.limit)
    print(json.dumps(report, indent=2))


async def _compose(args: argparse.Namespace) -> None:
    run = compose.RUNS[args.run_name]
    categories = args.category or list(compose.CATEGORIES)
    async with LlmClient(LlmSettings.from_env()) as llm:
        counts = await compose.generate_all(llm, run, args.total, categories)
    _print_counts(f"{run.name} rows per category and bucket:", Counter(counts))
    print(json.dumps({"rejected": compose.rejections(run)}, indent=1))


async def _relabel(args: argparse.Namespace) -> None:
    from .references import relabel

    scopes = relabel.SCOPES if args.step == "all" else (args.step,)
    if not args.report:
        async with LlmClient(_reference_settings("teacher", "judge", "generator")) as llm:
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


async def _review_refs(args: argparse.Namespace) -> None:
    async with LlmClient(_reference_settings("teacher", "judge")) as llm:
        report = await review.review_build(llm, args.limit)
        print(f"{llm.calls} requests, {llm.cache_hits} from the cache")
    print(json.dumps(report, indent=1))
    print(f"wrote {review.FIXES}")


def cmd_probe(args: argparse.Namespace) -> None:
    from .speech import probe

    if not args.report and args.model == "parakeet":
        written = probe.run(args.n, args.seed, keep_audio=not args.no_audio, shard=args.shard, shards=args.shards)
        print(f"{written} new rows in {probe.OUT_DIR}")
    elif not args.report:
        written = probe.redecode(args.model, shard=args.shard, shards=args.shards)
        print(f"{written} new {args.model} rows in {probe.OUT_DIR}")
    report = probe.report(probe.load(args.model))
    (probe.OUT_DIR / f"report.{args.model}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))


def cmd_heard(args: argparse.Namespace) -> None:
    from .speech import heard

    if args.step == "run":
        if not args.n:
            raise SystemExit("heard run needs --n, the number of train rows every process samples from")
        counts = heard.run(
            args.n, args.recognizer, args.seed, args.shard, args.shards, args.tts_provider, args.asr_provider, args.tts
        )
        _print_counts(f"{args.recognizer} shard {args.shard}:", counts)
    elif args.step == "restore":
        _print_counts("restored again:", heard.rerestore())
    print(json.dumps(heard.report(), indent=1))


def cmd_cutoff(args: argparse.Namespace) -> None:
    from .speech import cutoff

    if args.step == "run":
        _print_counts(f"shard {args.shard}:", cutoff.run(shard=args.shard, shards=args.shards, tts_provider=args.tts_provider))
    elif args.step == "check":
        _print_counts("Mercury Decide:", asyncio.run(cutoff.check()))
    elif args.step == "top-up":
        print(json.dumps(cutoff.top_up(args.shards, args.tts_provider), indent=1))
    elif args.step == "bench":
        print(f"wrote {cutoff.bench(tts_provider=args.tts_provider)} rows to {cutoff.BENCH}")
    elif args.step == "report":
        print(json.dumps(cutoff.report(), indent=1))
    else:
        print(cutoff.score(args.eval_dirs))


async def _weak_spots(args: argparse.Namespace) -> None:
    from .speech import repair

    if args.step == "report":
        print(json.dumps(round3_weak_spots.report(), indent=1))
        return
    async with LlmClient(_reference_settings("teacher", "judge")) as llm:
        if args.step == "check":
            result = await round3_weak_spots.check(llm)
        elif args.step == "top-up":
            result = await round3_weak_spots.top_up(llm)
        elif args.step == "repair":
            result = await repair.label(llm, args.limit)
        else:
            result = await round3_weak_spots.label(llm, args.step.replace("-", "_"), args.limit)
        print(f"{llm.calls} requests, {llm.cache_hits} from the cache")
    print(json.dumps(result, indent=1))


async def _literal_cue(args: argparse.Namespace) -> None:
    if args.step == "report":
        print(json.dumps(round4_cutoffs_cues.report(), indent=1))
        return
    async with LlmClient(_reference_settings("teacher", "judge")) as llm:
        step = round4_cutoffs_cues.check if args.step == "check" else round4_cutoffs_cues.top_up
        result = await step(llm)
        print(f"{llm.calls} requests, {llm.cache_hits} from the cache")
    print(json.dumps(result, indent=1))


def cmd_build(args: argparse.Namespace) -> None:
    if args.with_sotto and "base" not in args.rounds:
        raise SystemExit("--with-sotto changes round 1; pass --rounds base too")
    for name in build.ROUNDS:
        if name not in args.rounds:
            continue
        run, out_dir = build.ROUNDS[name]
        report = round1_base.build(args.with_sotto) if name == "base" else run()
        print(stats.stats_markdown(report))
        print(f"wrote round {name} to {out_dir}")


async def _bench(args: argparse.Namespace) -> None:
    from .bench import assemble, noise

    if args.step == "probe":
        print(f"{noise.probe_audio()} rows spoken and transcribed into {noise.PROBED}")
        return
    if args.step == "assemble":
        print(json.dumps(assemble.assemble(), indent=1))
        return
    async with LlmClient(LlmSettings.from_env()) as llm:
        if args.step == "ami":
            report = await assemble.label_ami(llm)
        elif args.step == "noise":
            report = await noise.generate_all(llm, args.kind or noise.KINDS)
        else:
            report = await noise.gate_probed(llm)
    print(json.dumps(report, indent=1))


def cmd_asr_speed(args: argparse.Namespace) -> None:
    from .speech import speed

    print(json.dumps(speed.run(args.model or None), indent=1))


def cmd_profile(args: argparse.Namespace) -> None:
    from . import profile
    from .schema import read_jsonl
    from .synthetic import inject

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
        rows = inject.injected_rows(read_examples(aawaaz_audit.CLEAN), round1_base.INJECTED, round1_base.SEED)
        for style in inject.STYLES:
            groups[f"injection now, {style}"] = [e.input for e in rows if f"style_{style}" in e.tags]
    for path in args.file or []:
        groups[str(path)] = [row["input"] for row in read_jsonl(path)]
    print(profile.table({label: profile.profile(texts) for label, texts in groups.items() if texts}))


def cmd_sample(args: argparse.Namespace) -> None:
    print(build.review_sample(args.layout, args.n, args.seed))


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


def _speech_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shard", type=int, default=0, help="the rows at positions shard, shard+shards, ...")
    parser.add_argument("--shards", type=int, default=1, help="how many processes share the rows")
    parser.add_argument(
        "--tts-provider", choices=["cuda", "cpu"], default="cuda",
        help="where Kokoro runs; cuda needs the audio-gpu extra and uses the float model",
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="datagen", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("sync-teacher", help="copy the teacher prompt from the Rust sources").set_defaults(
        run=cmd_sync_teacher
    )

    # Sources
    commands.add_parser("import-aawaaz", help="import and normalize aawaaz").set_defaults(run=cmd_import_aawaaz)
    judge = commands.add_parser("judge-aawaaz", help="flag aawaaz rows below its own standard")
    judge.add_argument("--limit", type=int, default=0, help="judge only the first N rows")
    judge.add_argument("--batch-size", type=int, default=15)
    judge.set_defaults(run=lambda args: asyncio.run(_judge_aawaaz(args)))
    commands.add_parser(
        "judge-register", help=f"flag {', '.join(aawaaz_audit.REGISTER_CATEGORIES)} rows that change register"
    ).set_defaults(run=lambda args: asyncio.run(_judge_register(args)))
    check = commands.add_parser("teacher-check", help="compare the teacher with aawaaz references")
    check.add_argument("--n", type=int, default=50)
    check.add_argument("--label", default="current", help="name of this prompt revision in the report")
    check.set_defaults(run=lambda args: asyncio.run(_teacher_check(args)))
    commands.add_parser("fix-aawaaz", help="relabel flagged aawaaz rows, judge again, drop failures").set_defaults(
        run=lambda args: asyncio.run(_fix_aawaaz(args))
    )
    add = commands.add_parser("add-public", help="add handy and disfl_qa rows, audited by the judge")
    add.add_argument(
        "--with-sotto",
        action="store_true",
        help="also add sotto rows; part of sotto is Grok output, which xAI's terms bar from training",
    )
    add.set_defaults(run=lambda args: asyncio.run(_add_public(args)))

    # Generated rows
    gen = commands.add_parser("generate", help="generate gap and hard-case rows; resumes where it stopped")
    gen.add_argument(
        "--category", action="append", choices=sorted({*generate.TARGETS, *(c for run in generate.RUN_TARGETS.values() for c in run)}),
        help="only this category (repeatable)",
    )
    gen.add_argument("--scale", type=float, default=1.0, help="multiply every category's target, e.g. 0.05 for a pilot")
    gen.add_argument("--batch-size", type=int, help="transcripts per request (default: 50, 25 for long_ramble)")
    gen.add_argument("--test", action="store_true", help="generate the extra rows pinned to the test split")
    gen.add_argument("--run", dest="run_name", choices=list(generate.RUN_TARGETS), help="a later round's extra train rows")
    gen.set_defaults(run=lambda args: asyncio.run(_generate(args)))
    restyle_parser = commands.add_parser("restyle", help="rewrite half the aawaaz inputs in recognizer style, words unchanged")
    restyle_parser.add_argument("--limit", type=int, default=0, help="restyle only the first N chosen rows")
    restyle_parser.add_argument("--recheck", action="store_true", help="check the rejected rewrites again; no LLM calls")
    restyle_parser.set_defaults(run=lambda args: asyncio.run(_restyle(args)))
    compose_parser = commands.add_parser(
        "compose",
        help="long composed-task rows (train), more of them weighted toward the weakest categories and lengths "
        "(targeted), or the long benchmark's generated rows (bench); resumes",
    )
    compose_parser.add_argument("--run", dest="run_name", choices=list(compose.RUNS), default="train")
    compose_parser.add_argument(
        "--total", type=int, required=True, help="rows over all categories, split by the run's bucket shares and category weights"
    )
    compose_parser.add_argument("--category", action="append", choices=compose.CATEGORIES, help="only this category (repeatable)")
    compose_parser.set_defaults(run=lambda args: asyncio.run(_compose(args)))

    # References
    relabel_parser = commands.add_parser(
        "relabel",
        help="rewrite every reference with gpt-6.1-sol at high reasoning (every role; the command refuses "
        "any other setting): build (round 1), bench-long, bench-noise or all; resumable",
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
    review_parser = commands.add_parser(
        "review-refs",
        help="have the teacher review suspect round 2 references (Disfl-QA against the original question, "
        "line-initial command case) and the judge verify each fix blind, into data/fixes/refs.jsonl",
    )
    review_parser.add_argument("--limit", type=int, default=0, help="review only the first N candidates of each kind")
    review_parser.set_defaults(run=lambda args: asyncio.run(_review_refs(args)))

    # Speech (needs --extra audio)
    probe = commands.add_parser("probe", help="speak sampled inputs with Kokoro and transcribe them as the app does")
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
    probe.set_defaults(run=cmd_probe)
    heard_parser = commands.add_parser(
        "heard", help="round 1 train rows spoken by Kokoro or Phonon and transcribed by Parakeet or Whisper, "
        "the pool of the recognizer repair rows"
    )
    heard_parser.add_argument(
        "step", choices=["run", "restore", "report"],
        help="restore: apply the current restore rules to every record's heard text, without speaking",
    )
    heard_parser.add_argument("--n", type=int, default=0, help="run: train rows sampled, the same for every process")
    heard_parser.add_argument("--recognizer", choices=["parakeet", "whisper"], default="parakeet")
    heard_parser.add_argument("--seed", type=int, default=120)
    heard_parser.add_argument(
        "--tts", choices=["kokoro", "phonon"], default="kokoro",
        help="who speaks; rows already heard are skipped whoever spoke them, so a Phonon run wants a new --seed",
    )
    heard_parser.add_argument(
        "--asr-provider", choices=["cpu", "cuda"], default="cpu",
        help="where the recognizer runs; the app runs it on the CPU, and only Parakeet fp16 gains much on the GPU",
    )
    _speech_options(heard_parser)
    heard_parser.set_defaults(run=cmd_heard)
    cutoff_parser = commands.add_parser(
        "cutoff", help="cut-off word starts ('interf interface'): run (cut round 3 train rows, Kokoro/Phonon and "
        "Parakeet), check (Mercury Decide, one call per question per cut), top-up (check, then replace failed rows "
        "until every length bucket is full), bench (data/bench/cutoff.jsonl), report, score (eval dirs' bench-cutoff outputs)"
    )
    cutoff_parser.add_argument("step", choices=["run", "check", "top-up", "bench", "report", "score"])
    cutoff_parser.add_argument("eval_dirs", nargs="*", type=Path, help="score: eval dirs with bench-cutoff and test outputs")
    _speech_options(cutoff_parser)
    cutoff_parser.set_defaults(run=cmd_cutoff)

    # Rounds
    weak_spots = commands.add_parser(
        "weak-spots", help="round 3's new train rows: disfl-qa or ami (label and gate), repair (label the heard "
        "records' raw recognizer text), check (the round's generated rows), top-up (replace the rows check drops), report"
    )
    weak_spots.add_argument("step", choices=["disfl-qa", "ami", "repair", "check", "top-up", "report"])
    weak_spots.add_argument("--limit", type=int, default=0, help="disfl-qa, ami, repair: label only the first N rows not done yet")
    weak_spots.set_defaults(run=lambda args: asyncio.run(_weak_spots(args)))
    literal_cue = commands.add_parser(
        "literal-cue", help="round 4's literal correction-word rows (generated with `generate --run round4`): check "
        "(a second teacher sample for every row), top-up (check, then generate replacements until each category has "
        "its target), report"
    )
    literal_cue.add_argument("step", choices=["check", "top-up", "report"])
    literal_cue.set_defaults(run=lambda args: asyncio.run(_literal_cue(args)))
    build_parser = commands.add_parser(
        "build", help="build rounds of the dataset, each from the one before; the last writes data/build"
    )
    build_parser.add_argument(
        "--rounds", nargs="+", choices=list(build.ROUNDS), default=list(build.LATER_ROUNDS),
        help="the rounds to build (default: every round after base, which is kept as built)",
    )
    build_parser.add_argument("--with-sotto", action="store_true", help="base: include sotto rows if add-public added them")
    build_parser.set_defaults(run=cmd_build)

    # Benchmarks and checks
    bench_parser = commands.add_parser(
        "bench",
        help="the held-out benchmarks: ami (label AMI turns), noise (generate), probe (Kokoro and Parakeet, "
        "needs --extra audio), gate-probed, assemble (write data/bench/*.jsonl); the long run's generated rows "
        "come from `compose --run bench`",
    )
    bench_parser.add_argument("step", choices=["ami", "noise", "probe", "gate-probed", "assemble"])
    bench_parser.add_argument("--kind", action="append", help="noise: only this pattern (repeatable)")
    bench_parser.set_defaults(run=lambda args: asyncio.run(_bench(args)))
    speed = commands.add_parser(
        "asr-speed", help="time each catalog speech model on the probe audio and score its WER (needs --extra audio)"
    )
    speed.add_argument("--model", action="append", help="measure only this model; repeatable")
    speed.set_defaults(run=cmd_asr_speed)
    profile_parser = commands.add_parser(
        "profile", help="filler formatting and correction cues of the real dictations, the dataset and given files"
    )
    profile_parser.add_argument("file", nargs="*", type=Path, help="more JSONL files with an input field")
    profile_parser.add_argument("--injection", action="store_true", help="also profile the current injection code")
    profile_parser.set_defaults(run=cmd_profile)
    sample = commands.add_parser("sample", help="print random rows of one layout for a hand review")
    sample.add_argument("layout", choices=["plain", "paragraphs", "list", "mixed"])
    sample.add_argument("--n", type=int, default=50)
    sample.add_argument("--seed", type=int, default=110)
    sample.set_defaults(run=cmd_sample)
    export = commands.add_parser("export-real", help="real-dictation test set from the local history (never uploaded)")
    export.add_argument("--history", type=Path, help="history.sqlite3 path; defaults to the app's data dir")
    export.add_argument("--limit", type=int, default=300)
    export.add_argument("--no-draft", action="store_true", help="skip the teacher drafts")
    export.set_defaults(run=lambda args: asyncio.run(_export_real(args)))
    publish = commands.add_parser("publish", help="push the dataset's splits to a private Hugging Face dataset")
    publish.add_argument("repo", help="dataset repo id, such as your-name/whispernow-cleanup")
    publish.set_defaults(run=cmd_publish)

    args = parser.parse_args()
    if args.command == "relabel" and args.settled_hedges and not args.retry:
        relabel_parser.error("--settled-hedges requires --retry with a round above every earlier round")
    args.run(args)
