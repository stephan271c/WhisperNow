# WhisperNow cleanup training

This directory builds the **WhisperNow Cleanup** dataset and trains, evaluates, exports and serves the local cleanup models the app runs: LFM2.5 1.2B, LFM2.5 2.6B and Qwen3.5 4B. A cleanup model takes a dictation transcript, as a speech recognizer writes it, and returns the text the speaker meant, or `EMPTY` when nothing was said.

- [making-the-dataset.md](making-the-dataset.md) explains how the dataset was built and why each round was added.
- `hub/` holds the Hugging Face cards: the dataset (`whispernow-cleanup`) and the three models.

## Layout

| Path | What it holds |
|---|---|
| `src/whispernow_training/datagen/sources/` | The imported datasets and their audits: aawaaz, handy and Disfl-QA (`public`), AMI meeting turns |
| `datagen/synthetic/` | Rows the teacher writes: generated gap and hard-case rows, long composed rows, filler injection, recognizer restyling, and the second-sample check |
| `datagen/references/` | Rewriting every reference with one teacher (`relabel`), and reviewing suspect ones (`review`) |
| `datagen/speech/` | Kokoro or Phonon speech through the app's recognizers: the probe, heard text, recognizer repair rows, cut-off word starts, and speech model speed |
| `datagen/dataset/` | The dataset in rounds (`round1_base` … `round4_cutoffs_cues`), with the filters, deduplication, input shaping and stats they share |
| `datagen/bench/` | The held-out benchmarks: long dictations and noisy transcripts |
| `datagen/llm.py`, `gates.py`, `judge.py`, `teacher.py` | The LLM client, the reference gates, the batch judge and the app's teacher request |
| `src/whispernow_training/model/` | Training, evaluation, metrics, GGUF export, serving and latency |
| `configs/` | One TOML file per model, and the fixed test samples |
| `prompts/` | Every prompt ([prompts/README.md](prompts/README.md)) |
| `serving/` | The chat template, the prompt fixture shared with the app, and the llama-server patch |
| `modal_train.py` | Training on a Modal GPU |

## Setup

- `uv sync` installs the data tools. Run commands with `uv run`, and tests with `uv run pytest`.
- Speech steps need `uv sync --extra audio` (sherpa-onnx 1.13.6, the version the app vendors), the app's Parakeet TDT 0.6B v2 fp16 and Silero VAD models in `~/.local/share/WhisperNow/models/`, and Kokoro `kokoro-int8-multi-lang-v1_0` (the k2-fsa/sherpa-onnx `tts-models` release) in `.cache/tts/`. Whisper and Qwen3-ASR runs read `sherpa-onnx-whisper-distil-large-v3.5` and `sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25` from `.cache/asr/`. `uv sync --inexact --extra audio-gpu` swaps in the CUDA 13 build for Kokoro on an NVIDIA GPU, with the float `kokoro-multi-lang-v1_0`; `--tts-provider cpu` keeps the int8 model.
- Training needs `uv sync --extra train` (torch with CUDA 13.0 wheels, transformers and TRL). `uv sync` with one extra removes the others; `uv sync --all-extras` keeps them all. Model commands run as `uv run --extra train cleanup …`.
- Export and serving use llama.cpp release **b11259** under `.cache/llama.cpp/`: `src/` (the release source, for `convert_hf_to_gguf.py`), `cuda/llama-b11259/` and `cpu/llama-b11259/` (the release binaries). `cleanup build-server [--cpu]` builds the patched llama-server into `draft-text/build-{cuda,cpu}/`; it takes about 10 minutes, bounded like a Cargo build (AGENTS.md).
- LLM settings come from the repo-root `.env`: the `LLM_*` variables that `LlmSettings.from_env` in `datagen/llm.py` reads. `publish` needs `hf auth login`.
- `data/`, `runs/` and `.cache/` are gitignored. Run GPU and CPU-heavy jobs under `nice -n 10 ionice -c2 -n7`, never while Cargo builds.

## The dataset

The dataset is built in four rounds. Each round starts from the files the one before wrote and adds train rows only: val and test are copied byte for byte from round 1, so every round's models are scored on the same held-out rows. A round's own rows carry its tag (`round2`, `round3`, `round4`).

| Round | Module | Writes | Adds | Train rows |
|---|---|---|---|---|
| 1, base | `round1_base` | `data/rounds/1-base/` | The audited sources, generated and composed rows, filler injection and `EMPTY` rows, every reference rewritten by `gpt-6.1-sol` and gated | 46,389 |
| 2, targeted | `round2_targeted` | `data/rounds/2-targeted/` | Long composed rows weighted toward the categories and lengths where round 1's models lost the most facts | 56,573 |
| 3, weak spots | `round3_weak_spots` | `data/rounds/3-weak-spots/` | Disfl-QA questions, AMI meeting turns, generated rows in four weak categories, verified reference fixes, recognizer repair rows | 60,236 |
| 4, cut-offs and literal cues | `round4_cutoffs_cues` | `data/build/` | Cut-off word starts in round 3's inputs, short dictations that use a correction word literally | 60,836 |

`data/build/` is the dataset: train 60,836, val 2,424 and test 3,297 rows. Round 1 is kept as built, with the test references round 3's review fixed written into it. Today's code rebuilds it with small differences, so `datagen build` rebuilds rounds 2 to 4 from it and round 1 only with `--rounds base`.

### Commands

Run them in this order. The model each step uses comes from `.env`; the steps that write references (`relabel`, `review-refs`, `weak-spots`, `literal-cue`) refuse any teacher or judge but `gpt-6.1-sol` at high reasoning through codex.

| Step | Command | Writes |
|---|---|---|
| Copy the app's cleanup prompt | `datagen sync-teacher` | `prompts/teacher/` |
| **Sources** | | |
| Import aawaaz | `datagen import-aawaaz` | `data/aawaaz/aawaaz.imported.jsonl` |
| Judge every aawaaz row, then its register-sensitive categories | `datagen judge-aawaaz`, `datagen judge-register` | `data/aawaaz/judgments.jsonl`, `register_judgments.jsonl` |
| Compare the teacher with aawaaz's references | `datagen teacher-check --label NAME` | `data/aawaaz/teacher_check/` |
| Relabel or drop the flagged rows | `datagen fix-aawaaz` | `data/aawaaz/aawaaz.clean.jsonl` |
| Add handy and Disfl-QA, audited (`--with-sotto` also adds sotto; see the terms below) | `datagen add-public` | `data/public/*.jsonl` |
| **Generated rows** | | |
| Gap and hard-case rows; `--test` for the rows pinned to test | `datagen generate [--test]` | `data/generated/<category>[.test].jsonl` |
| Restyle half the aawaaz inputs as a recognizer writes them | `datagen restyle` | `data/restyle/` |
| Long composed rows | `datagen compose --total 8300` | `data/composed/` |
| **References** | | |
| Rewrite every reference of round 1 and both benchmarks | `datagen relabel all` | `data/relabel/` |
| Review suspect references, verified blind by the judge | `datagen review-refs` | `data/fixes/refs.jsonl` |
| **Speech** (`--extra audio`) | | |
| Hear how the recognizers write 300 sampled inputs | `datagen probe [--model whisper\|qwen3]` | `data/probe/` |
| Round 1 train rows spoken and transcribed, the pool of the repair rows | `datagen heard run --n N --recognizer parakeet` | `data/heard/` |
| Cut-off word starts, checked by Mercury Decide, until every length bucket is full | `datagen cutoff run`, then `datagen cutoff top-up` | `data/cutoff/` |
| **Round 2** | | |
| Targeted long composed rows | `datagen compose --run targeted --total N` | `data/composed-targeted/` |
| **Round 3** | | |
| Disfl-QA questions and AMI turns, labelled and gated | `datagen weak-spots disfl-qa`, `datagen weak-spots ami` | `data/weak_spots/` |
| Recognizer repair rows from the heard records | `datagen weak-spots repair` | `data/repair/` |
| Rows in the weak categories, checked until each has its target | `datagen generate --run round3`, then `datagen weak-spots top-up` | `data/generated/<category>.round3.jsonl`, `round3.checks.jsonl` |
| **Round 4** | | |
| Literal correction-word rows, each with a second teacher sample | `datagen generate --run round4`, then `datagen literal-cue top-up` | `data/generated/literal_cue.round4.jsonl`, `round4.checks.jsonl` |
| **Build** | | |
| Rounds 2 to 4 (`--rounds base targeted …` picks rounds) | `datagen build` | `data/rounds/*/`, `data/build/` |
| **Benchmarks** | | |
| The long benchmark's generated rows and AMI turns | `datagen compose --run bench --total 360`, `datagen bench ami` | `data/bench/long/` |
| The noise benchmark, a third of it spoken and transcribed | `datagen bench noise`, `datagen bench probe`, `datagen bench gate-probed` | `data/bench/noise/` |
| Write both benchmark files and check them against the dataset | `datagen bench assemble` | `data/bench/{long,noise}.jsonl` |
| Test rows with cut-off word starts | `datagen cutoff bench` | `data/bench/cutoff.jsonl` |
| **Checks** | | |
| Rows of one layout for a hand review | `datagen sample list --n 50` | stdout |
| Filler and correction-cue rates of the dataset, the injection or any file | `datagen profile [--injection] [FILE …]` | stdout |
| Speed and WER of the app's speech models on the probe audio | `datagen asr-speed` | `data/probe/asr_speed.json` |
| Real dictations from the local history, to read by hand (never uploaded) | `datagen export-real` | `data/real/real.jsonl` |
| Push the dataset to a private Hugging Face repo, with the card's 7 fields and 10 categories (`build.PUBLISHED_CATEGORY`) | `datagen publish category271/whispernow-cleanup` | Hugging Face |

Every LLM step resumes. Judges append verdicts as batches finish, generators log each batch, and every reply is cached by its exact request in `.cache/llm/`, so a rerun repeats no finished call. `generate --scale 0.05` runs a pilot at 5% of each target, and `--category NAME` runs one category. `weak-spots report`, `literal-cue report`, `heard report` and `cutoff report` summarize their records.

### What a round does with its rows

- **Filters** (`dataset/filters.py`) drop references with markdown, malformed lists, answers instead of edits, a protected token (path, URL, email, identifier, flag) written another way, an invented subject line, legal formality or signature, or an implausible length.
- **Deduplication** (`dataset/dedup.py`) drops exact and near-duplicate inputs (MinHash over word 3-grams without fillers, Jaccard 0.85 or more) and outputs repeated more than 3 times. Test rows, earlier sources and earlier rounds win.
- **Splits** go by `seed_group`, so a row made from another shares its split.
- **Input shaping** (`dataset/shape.py`) gives composed rows a recognizer's punctuation (Parakeet, Whisper, Qwen3-ASR or none), densifies fillers in half of them and in 30% of the short generated, handy and Disfl-QA train and val rows, and leaves fillers bare in two thirds of the generated recognizer-style rows. Every change is one the reference already drops, so references stay right.
- **Stats** go to `stats.{json,md}` beside the splits, with the dropped rows in `dropped.jsonl`.

## The models

Each model is a full fine-tune of a base checkpoint on `data/build/`. It takes one user message holding only the transcript, with no system message, and replies with the edited text or `EMPTY`.

### The prompt contract

- `serving/chat_template.jinja` is the one chat template: training renders every row with it and serving uses it, and it refuses a system turn.
- `serving/prompt_golden.json` pins the messages, the rendered prompt, the output budget, the decoding settings and the draft for a few transcripts. `tests/test_model_prompt.py` and the app's `local_cleanup_prompt_matches_the_training_golden_fixture` both check against it, so change it, the template, `model/prompt.py` and the app's `build_local_cleanup_prompt` together.
- Each request carries a draft, the transcript pre-cleaned by code (`model/draft.py`; in the app, `llm/draft.rs`). The app's engine drafts from it for speculative decoding; training-side requests send it as `draft_text` to the patched llama-server.
- Decoding is greedy, with `repeat_penalty` 1.05 over `repeat_last_n` 4096 tokens, and `max_tokens` = max(64, ⌈2.5 × input words⌉). A reply that hits the limit counts as a loop, and the app keeps the pre-cleaned text.

### Commands

| Step | Command | Writes |
|---|---|---|
| Train on a Modal GPU | `modal run --detach -e main modal_train.py --config configs/lfm25-2.6b.toml --name NAME --gpu H100`, then `modal volume get -e main whispernow-runs NAME runs/` | `runs/NAME/` |
| Train locally (`--set key=value` overrides any config value) | `cleanup train configs/lfm25-1.2b.toml --name NAME` | `runs/NAME/` |
| Evaluate the checkpoint with transformers | `cleanup eval NAME --split test --judge` | `runs/NAME/eval/` |
| Export to GGUF | `cleanup export NAME [--quant Q8_0 Q4_K_M]` | `runs/NAME/gguf/` |
| Serve a GGUF with the committed template (`--cpu`, `--parallel 4`, `--no-draft`) | `cleanup serve NAME --quant Q8_0` | llama-server on `127.0.0.1:8080` |
| Evaluate what the server returns, on a split or a benchmark | `cleanup eval NAME --server http://127.0.0.1:8080/v1 --label gpu-q8 --split bench-long --judge` | `runs/NAME/eval-gpu-q8/` |
| Evaluate a cloud model with the app's prompt (`LLM_MODEL`) | `cleanup eval --teacher --label NAME --split test --judge` | `runs/teacher/eval-NAME/` |
| Latency on a short dictation and test rows of 100 to 1,000 words | `cleanup bench http://127.0.0.1:8080/v1 --save runs/NAME/bench.json` | the JSON |
| Compare evaluations | `cleanup compare a=NAME/eval-gpu-q8 b=teacher/eval-x --stem test.r112.faults [--group length]` | stdout |
| Score saved outputs again with the current checks, or judge chosen rows again | `cleanup rescore RUN/EVAL …`, `cleanup rejudge RUN/EVAL … --ids IDS.txt` | `<stem>.metrics2.scores.json` |
| Read a server's output on the real dictations, nothing scored | `cleanup handcheck http://127.0.0.1:8080/v1` | stdout |

`--split` takes `train`, `val`, `test`, the benchmarks (`bench-long`, `bench-noise`, `bench-cutoff`), and two fixed test samples for slow or costly models: `test-sample` (800 rows drawn per category and length bucket) and `test-long` (the 536 test rows over 200 words). `--limit N` scores a fixed random sample. Evaluation reuses the outputs it already wrote, so adding `--judge` later only runs the judge. `datagen cutoff score` scores `bench-cutoff` outputs against the same rows' test outputs.

### Configs

`configs/lfm25-1.2b.toml`, `lfm25-2.6b.toml` and `qwen35-4b.toml` share one recipe: learning rate 1e-5 with a cosine schedule and 3% warmup, 3 epochs, 32 rows per optimizer step, fp32 master weights under bf16 autocast, and TRL's chunked loss. Each file's comments give its GPU memory needs. Training is SFT on prompt-completion pairs rendered by the committed template, with loss on the completion only. `checkpoint_minutes` saves a resumable checkpoint to the runs Volume, since Modal's GPUs can be preempted, and `pause_at_step` stops a run to finish it in another workspace.

### What evaluation reports

`<split>.scores.json` holds each rate overall and per category, input style, source, tag, length bucket and noise pattern:

- **Mechanical checks** (`model/metrics.py`): chrF and exact match, layout, formatting added or missed, markdown and list markers, protected tokens kept, fillers left, `EMPTY` handling, loops, assistant openers, large deletions, em dashes, spelled-out counts and correction cues kept.
- **Judge** (`--judge`, `prompts/judge/eval.md`): whether the output answered the transcript instead of editing it, and every fault, one per fact: `dropped`, `changed`, `wrong_side` (an abandoned value kept), `unresolved` (both sides of a correction kept) and `invented`. `facts_lost` counts the first three per judged row and per 1,000 reference words; the model cards report the latter. Rows over 300 words get a judge request each.

When every reference of a split was written by `gpt-6.1-sol` at high reasoning, the judge and score files are named `<split>.r112.faults.*`, and the judge must be the same model. Cloud teachers score high on chrF against their own kind of reference by construction, so compare on the judge's counts.

### Using it in the app

The app runs the models in process on llama.cpp b11259's C API, on the GPU when one works (Vulkan on Linux and Windows, Metal on Apple silicon) and otherwise on the CPU. Choose the provider **Local cleanup model**, pick a model in Settings → Models, and download it. Each model is pinned in the app's registry by URL, size and SHA-256. The patched server (`cleanup build-server`, `cleanup serve`) is for evaluation and latency only.

## The LLM client

`datagen/llm.py` reads `LLM_*` from `.env`. `LLM_MODEL` is the teacher, `LLM_JUDGE_MODEL` the judge, and `LLM_GENERATOR_MODEL` writes transcripts (it defaults to the teacher). Each role has a backend (`LLM_BACKEND`, `LLM_JUDGE_BACKEND`, `LLM_GENERATOR_BACKEND`) and a reasoning effort (`LLM_REASONING`, …):

- `codex` runs one `codex exec` per request, from an empty directory with a read-only sandbox and no session saved; our system message replaces codex's instructions. Each call runs in its own process group, so a timeout kills the native binary too.
- `claude` runs one `claude -p` per request in safe mode, with no tools or MCP servers.
- `openai` calls any OpenAI-compatible endpoint (`LLM_BASE_URL`, `LLM_API_KEY`). An OpenRouter model ending in `:batch` goes through OpenRouter's batch API at half price; `cleanup eval --teacher` waits for the batch, whose id is kept in `.cache/llm-batches/`.

A semaphore per backend caps requests in flight (`LLM_CONCURRENCY`, `LLM_CODEX_CONCURRENCY`). Rate limits, 5xx and timeouts retry up to 7 times with exponential backoff. Judges take rows in batches, 15 by default. Only a reply's final text is kept. `LLM_GENERATORS` can spread the generator role over several routes (`llm.PROVIDERS`); every row records its generator, teacher and judge.

## Terms and licenses

This is a reading of the published terms, not a legal review.

- **A row's `license` is its text's license** (`schema.OWN_LICENSE`). An imported row keeps its dataset's license, though `gpt-6.1-sol` rewrote its reference. Text made for this dataset (generated transcripts, injected fillers, `EMPTY` rows) is under the repository's MIT license.
- **Model outputs.** `gpt-6.1-sol` (through the Codex CLI) wrote every reference and most generated transcripts. OpenAI's terms forbid using output to develop models that compete with OpenAI; whether a local dictation-cleanup model competes is the project owner's call. Z.ai's GLM models wrote the earlier generated transcripts and restyled half the aawaaz inputs, and Z.ai's terms carry the same clause. Every row names the models that wrote it.
- **aawaaz** (`shantanugoel/aawaaz-transcript-cleanup-dataset`): MIT. Its outputs came from Gemini 2.5 Flash and coding agents. Its generator repo has no license, so our prompts are written from scratch.
- **handy** (`MagicNoThief/handy-dictation-editing`): CC-BY-4.0. Rows that share a line, or a quarter of their word 6-grams, with the DisfluencySpeech transcripts (Switchboard re-recordings) are dropped.
- **Disfl-QA** (`google-research-datasets/disfl_qa`): CC-BY-4.0; its questions derive from SQuAD 2.0 (CC-BY-SA-4.0).
- **AMI** meeting transcripts: CC-BY-4.0, text only.
- **Left out:** sotto (`juanquivilla/sotto-transcript-cleanup`, MIT but partly written by Grok, whose terms bar training competing models) unless `--with-sotto` is passed; any non-commercial or LDC data.
