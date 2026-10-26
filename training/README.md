# WhisperNow cleanup training

This directory builds the **WhisperNow Cleanup** dataset, then trains, evaluates, exports and serves the cleanup models the app runs locally: LFM2.5 1.2B, LFM2.5 2.6B and Qwen3.5 4B. A cleanup model takes a noisy speech transcript and returns the cleaned text, or `EMPTY` when nothing was said.

[making-the-dataset.md](making-the-dataset.md) tells how the dataset was built in four rounds and why. `hub/` holds the Hugging Face cards for the dataset and the three models.

## Layout

| Path | What it holds |
|---|---|
| `src/whispernow_training/datagen/` | The dataset code: the LLM client, the reference checks (`gates.py`) and the judge |
| `src/whispernow_training/datagen/sources/` | Imported datasets and their audits |
| `src/whispernow_training/datagen/synthetic/` | Generated, composed, restyled and injected rows |
| `src/whispernow_training/datagen/references/` | Reference rewriting (`relabel`) and review |
| `src/whispernow_training/datagen/speech/` | TTS and recognizer runs |
| `src/whispernow_training/datagen/dataset/` | The four rounds, and the filters, deduplication and input shaping they share |
| `src/whispernow_training/datagen/bench/` | The benchmarks |
| `src/whispernow_training/model/` | Training, evaluation, metrics, GGUF export, serving and latency |
| `configs/` | One TOML file per model, and the fixed test samples |
| `prompts/` | Every prompt ([prompts/README.md](prompts/README.md)) |
| `serving/` | The chat template, the prompt fixture shared with the app, and the llama-server patch |
| `modal_train.py` | Training on a Modal GPU |

`data/`, `runs/` and `.cache/` are gitignored.

## Setup

- `uv sync` installs the data tools. Run commands with `uv run`, and tests with `uv run pytest`.
- **Speech steps** need `uv sync --extra audio` (sherpa-onnx 1.13.6) and these models:
  - the app's Parakeet TDT 0.6B v2 fp16 and Silero VAD, in `~/.local/share/WhisperNow/models/`;
  - Kokoro `kokoro-int8-multi-lang-v1_0` from the k2-fsa/sherpa-onnx `tts-models` release, in `.cache/tts/`;
  - for Whisper and Qwen3-ASR runs, `sherpa-onnx-whisper-distil-large-v3.5` and `sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25`, in `.cache/asr/`.

  `heard` and `cutoff` run Kokoro on the GPU by default. That needs the CUDA 13 build (`uv sync --inexact --extra audio-gpu`) and the float `kokoro-multi-lang-v1_0` in `.cache/tts/`. Pass `--tts-provider cpu` to use the int8 model on the CPU.

  The app uses sherpa-onnx 2.0.0, but 2.0 ignores the eSpeak data Kokoro reads here and shifts Parakeet's transcripts slightly, so the speech steps stay on 1.13.6.
- **Training** needs `uv sync --extra train` (torch with CUDA 13.0 wheels, transformers and TRL). Run model commands as `uv run --extra train cleanup …`. Syncing one extra removes the others, so name each one you need: `uv sync --extra audio --extra train`.
- **Export and serving** use llama.cpp release **b11259** in `.cache/llama.cpp/`: the release source in `src/` (for `convert_hf_to_gguf.py`) and the release binaries in `cuda/llama-b11259/` and `cpu/llama-b11259/`. `cleanup build-server [--cpu]` builds the patched llama-server into `.cache/llama.cpp/draft-text/build-{cuda,cpu}/`. The build takes about 10 minutes; run it under the CPU and I/O limits AGENTS.md sets for Cargo.
- **LLM settings** come from the repo-root `.env` ([below](#llm-roles)). `datagen publish` needs `hf auth login`.
- Run GPU- and CPU-heavy jobs under `nice -n 10 ionice -c2 -n7`, and not while a Cargo build is running.

## Building the dataset

The steps below run in order, grouped by round.

- **Resuming.** Rerun a stopped LLM step and it picks up where it left off. Every reply is cached in `.cache/llm/` under its exact request, and judges save their verdicts as each batch finishes.
- **Pilots.** `generate --scale 0.05` generates 5% of each category's target, and `--category NAME` generates one category.
- **Reference writers.** The steps that write references (`relabel`, `review-refs`, `weak-spots`, `literal-cue`) run only with `gpt-6.1-sol` at high reasoning through codex as both teacher and judge.

### Round 1

Later rounds build from the stored `data/rounds/1-base/`, so don't rebuild it unless you mean to replace it. A rebuild won't match the stored files. The filler injection and filters changed after round 1 was built, which moves about 100 rows. No build applies the 16 test fixes from round 3's review either: they were written into the stored test file directly, and `data/fixes/test-before.jsonl` keeps the old lines.

These are the steps that made round 1:

| Step | Command | Writes |
|---|---|---|
| Copy the app's cleanup prompt | `datagen sync-teacher` | `prompts/teacher/` |
| Import aawaaz, flag rows below its standard and rows that change register, compare the teacher with aawaaz's references, and relabel or drop the flagged rows | `datagen import-aawaaz`, `judge-aawaaz`, `judge-register`, `teacher-check --label NAME`, `fix-aawaaz` | `data/aawaaz/` |
| Add handy and Disfl-QA, with their audits | `datagen add-public` | `data/public/` |
| Short generated rows; `--test` writes the rows pinned to test | `datagen generate [--test]` | `data/generated/` |
| Long generated dictations | `datagen compose --total 8300` | `data/composed/` |
| A first build of round 1 | `datagen build --rounds base` | `data/rounds/1-base/` |
| Speak 300 inputs from that build with Kokoro and transcribe them as the app does (Parakeet unless `--model` is given) | `datagen probe [--model whisper\|qwen3]` | `data/probe/` |
| Restyle half the aawaaz inputs as a recognizer writes them | `datagen restyle` | `data/restyle/` |
| Build again with the restyled inputs, rewrite every reference, then build with the new references | `datagen build --rounds base`, `relabel build`, `build --rounds base` | `data/relabel/`, `data/rounds/1-base/` |

### Benchmarks

`relabel` reads the assembled benchmark files, so assemble each benchmark, relabel it, and assemble it again. `relabel all` relabels the round 1 build and both benchmarks.

| Step | Command | Writes |
|---|---|---|
| The long benchmark's generated rows and AMI turns | `datagen compose --run bench --total 360`, `bench ami` | `data/bench/long/` |
| The noise benchmark, a third of it spoken and transcribed | `datagen bench noise`, `bench probe`, `bench gate-probed` | `data/bench/noise/` |
| Write both benchmark files and check them for overlap with the dataset | `datagen bench assemble` | `data/bench/{long,noise}.jsonl`, `report.json` |
| Rewrite their references, then write them again | `datagen relabel bench-long`, `relabel bench-noise`, `bench assemble` | `data/relabel/`, `data/bench/` |

The overlap check is skipped until `data/build/` exists, so run `bench assemble` again after round 4.

### Rounds 2 to 4

Build each round before running the next round's steps: round 3's steps read round 2, and round 4's read round 3. Once every step has run, `datagen build` rebuilds rounds 2 to 4 together.

Each round adds train rows only and copies val and test byte for byte. Its new rows are filtered (`dataset/filters.py`), deduplicated so that an earlier row always wins (`dataset/dedup.py`), and given the row fixes and input variants every round applies (`dataset/shape.py`). Each round writes `stats.{json,md}` and the rows it dropped (`dropped.jsonl`) beside its splits.

| Step | Command | Writes |
|---|---|---|
| **Round 2:** long generated dictations, weighted toward the weakest categories | `datagen compose --run targeted --total N` | `data/composed-targeted/` |
| Build round 2 | `datagen build --rounds targeted` | `data/rounds/2-targeted/` |
| **Round 3:** speak and transcribe round 1 train rows, the pool of the recognizer repair rows | `datagen heard run --n N --recognizer parakeet` | `data/heard/` |
| Review suspect references; the judge checks each fix blind | `datagen review-refs` | `data/fixes/refs.jsonl` |
| Disfl-QA questions and AMI turns, with references written and checked | `datagen weak-spots disfl-qa`, `weak-spots ami` | `data/weak_spots/` |
| Short rows in the weak categories; `top-up` replaces the rows the check drops | `datagen generate --run round3`, `weak-spots top-up` | `data/generated/*.round3.jsonl`, `round3.checks.jsonl` |
| Recognizer repair rows from the heard records | `datagen weak-spots repair` | `data/repair/` |
| Build round 3 | `datagen build --rounds weak-spots` | `data/rounds/3-weak-spots/` |
| **Round 4:** bench-cutoff, test rows with cut-off word starts | `datagen cutoff bench` | `data/bench/cutoff.jsonl` |
| Cut-off word starts in round 3 train inputs; `top-up` replaces the rows Mercury fails until every length bucket is full | `datagen cutoff run`, `cutoff top-up` | `data/cutoff/` |
| Literal-cue rows (correction words used literally), each checked against a second teacher sample | `datagen generate --run round4`, `literal-cue top-up` | `data/generated/literal_cue.round4.jsonl`, `round4.checks.jsonl` |
| Build round 4, the published dataset | `datagen build --rounds cutoffs-cues` | `data/build/` |
| Push the dataset to a private Hugging Face repo, with the card's columns and broader categories (`build.PUBLISHED_CATEGORY`) | `datagen publish category271/whispernow-cleanup` | Hugging Face |

### Checks

| Check | Command | Writes |
|---|---|---|
| Print 50 random rows of one layout (`plain`, `paragraphs`, `list` or `mixed`) to read by hand | `datagen sample list --n 50` | stdout |
| Filler and correction-cue rates of the dataset, the filler injection (`--injection`) or any file | `datagen profile [--injection] [FILE …]` | stdout |
| Speed and WER of the app's speech models on the probe audio | `datagen asr-speed` | `data/probe/asr_speed.json` |
| Real dictations from the local history, to read by hand (never uploaded) | `datagen export-real` | `data/real/real.jsonl` |
| Summaries of a step's records | `datagen weak-spots report`, `literal-cue report`, `heard report`, `cutoff report` | stdout |

## Training and evaluating

Each model is a full fine-tune of a base checkpoint on `data/build/`. It takes one user message holding only the transcript, with no system message, and replies with the cleaned text or `EMPTY`.

| Step | Command | Writes |
|---|---|---|
| Train on a Modal GPU | `modal run --detach -e main modal_train.py --config configs/lfm25-2.6b.toml --name NAME --gpu H100`, then `modal volume get -e main whispernow-runs NAME runs/` | `runs/NAME/` |
| Train locally (`--set key=value` overrides any config value) | `cleanup train configs/lfm25-1.2b.toml --name NAME` | `runs/NAME/` |
| Evaluate the checkpoint with transformers | `cleanup eval NAME --split test --judge` | `runs/NAME/eval/` |
| Export to GGUF | `cleanup export NAME [--quant Q8_0 Q4_K_M]` | `runs/NAME/gguf/` |
| Serve a GGUF with the committed template (`--cpu`, `--parallel 4`, `--no-draft`) | `cleanup serve NAME --quant Q8_0` | llama-server on `127.0.0.1:8080` |
| Evaluate what the server returns, on a split or a benchmark | `cleanup eval NAME --server http://127.0.0.1:8080/v1 --label gpu-q8 --split bench-long --judge` | `runs/NAME/eval-gpu-q8/` |
| Evaluate a cloud model with the app's prompt (`LLM_MODEL`) | `cleanup eval --teacher --label NAME --split test --judge` | `runs/teacher/eval-NAME/` |
| Latency on a short dictation and on test rows of 100 to 1,000 words | `cleanup bench http://127.0.0.1:8080/v1 --save runs/NAME/bench.json` | the JSON |
| Compare evaluations | `cleanup compare a=NAME/eval-gpu-q8 b=teacher/eval-x --stem test.r112.faults [--group length]` | stdout |
| Score saved outputs again with the current checks, or judge chosen rows again | `cleanup rescore RUN/EVAL …`, `cleanup rejudge RUN/EVAL … --ids IDS.txt` | `<stem>.metrics2.scores.json` |
| Print a server's output on the real dictations, unscored | `cleanup handcheck http://127.0.0.1:8080/v1` | stdout |

**Splits.** `--split` takes:

- `train`, `val` and `test`;
- the benchmarks `bench-long`, `bench-noise` and `bench-cutoff`;
- two fixed test samples for slow or costly models: `test-sample` (800 rows drawn per category and length bucket) and `test-long` (the 536 test rows over 200 words).

`--limit N` scores a fixed random sample. Evaluation reuses the outputs it already wrote, so adding `--judge` later only runs the judge. `datagen cutoff score` scores `bench-cutoff` outputs against the same rows' test outputs.

**Configs.** `configs/lfm25-1.2b.toml`, `lfm25-2.6b.toml` and `qwen35-4b.toml` share one recipe: SFT on prompt-completion pairs rendered by the committed template, with loss on the completion only, at learning rate 1e-5 for 3 epochs. Each file's comments give its GPU memory needs. `checkpoint_minutes` saves a resumable checkpoint to the runs Volume, since Modal can preempt a GPU. `pause_at_step` stops a run at a step, so it can resume in another Modal workspace.

**Scores.** `<split>.scores.json` holds each rate overall and per category, input style, source, tag, length bucket and noise pattern. The code checks live in `model/metrics.py`. `--judge` adds the judge's faults (`prompts/judge/eval.md`) and two rates per 1,000 reference words:

- `factual_errors`: dropped, changed, wrong-side and invented facts. The model cards report this one.
- `facts_lost`: the same without invented facts.

When every reference of a split was written by `gpt-6.1-sol` at high reasoning, the judge and score files are named `<split>.r112.faults.*`, and the judge must be the same model. A cloud model scores high on chrF just by writing like the teacher that wrote the references, so compare cloud models on the judge's counts.

### The prompt contract

Training and the app must build the same prompt.

- Training and serving both render prompts with `serving/chat_template.jinja`, which rejects a system message.
- `serving/prompt_golden.json` pins the messages, the rendered prompt, the token limit, the decoding settings and the draft for a few transcripts. `tests/test_model_prompt.py` and the app's `local_cleanup_prompt_matches_the_training_golden_fixture` both check against it. Change it, the template, `model/prompt.py` and the app's `build_local_cleanup_prompt` together.
- Each request carries a draft: the transcript pre-cleaned by code (`model/draft.py`; in the app, `llm/draft.rs`). The app's engine uses it for speculative decoding, and requests from this directory send it to the patched llama-server as `draft_text`.
- Decoding is greedy, with `repeat_penalty` 1.05 over `repeat_last_n` 4096 tokens, and `max_tokens` = max(64, ⌈2.5 × input words⌉). A reply that hits the limit counts as a loop, and the app keeps the pre-cleaned text.

### In the app

The app runs the models in process on llama.cpp b11259's C API. It uses the GPU when it can (Vulkan on Linux and Windows, Metal on Apple silicon) and the CPU otherwise. To use a model, choose the provider **Local cleanup model**, then pick and download one in Settings → Models. Each model is pinned in the app's registry by URL, size and SHA-256. The patched server is for evaluation and latency only.

## LLM roles

`datagen/llm.py` reads `LLM_*` from the repo-root `.env`:

- `LLM_MODEL` is the teacher.
- `LLM_JUDGE_MODEL` is the judge.
- `LLM_GENERATOR_MODEL` writes transcripts and defaults to the teacher.

Each role has a backend (`LLM_BACKEND`, `LLM_JUDGE_BACKEND`, `LLM_GENERATOR_BACKEND`) and a reasoning effort (`LLM_REASONING`, …). Every row records the generator, teacher and judge that made it.

- `codex` runs one `codex exec` per request from an empty directory, with a read-only sandbox and no session saved. Our system message replaces codex's instructions.
- `claude` runs one `claude -p` per request in safe mode, with no tools or MCP servers.
- `openai` calls any OpenAI-compatible endpoint (`LLM_BASE_URL`, `LLM_API_KEY`). An OpenRouter model ending in `:batch` goes through OpenRouter's batch API at half price.

`LLM_CONCURRENCY` and `LLM_CODEX_CONCURRENCY` cap requests in flight. The client retries rate limits, 5xx errors and timeouts with backoff.

## Licenses

A row's `license` is its source's license: aawaaz is MIT, and handy, Disfl-QA and AMI are CC-BY-4.0. Disfl-QA's questions derive from SQuAD 2.0, which is CC-BY-SA-4.0. Text made for this dataset (generated transcripts, injected fillers, `EMPTY` rows) is under the repository's MIT license.
