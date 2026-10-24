# WhisperNow training data

Builds the dataset for a local Clean Up & Format model (todo 108), then trains, evaluates, exports and serves that model (todo 109; see [The cleanup model](#the-cleanup-model-todo-109)).

Every row has one schema (`datagen/schema.py`): `id`, `input`, `output`, `source`, `category`, `tags`, `layout` (`plain`, `list`, `paragraphs` or `mixed`, computed from the output), `seed_group`, `split`, `license`, `teacher_model` and `teacher_reasoning` (the teacher's reasoning effort, set from the todo 112 relabel on).

## Setup

- `uv sync` installs everything. Run commands with `uv run`, and tests with `uv run pytest`.
- LLM settings come from the repo-root `.env`: the `LLM_*` variables that `LlmSettings.from_env` in `datagen/llm.py` reads. The key is never printed or logged.
- `publish` needs a Hugging Face login (`hf auth login`).
- `asr-probe` needs `uv sync --extra audio` (sherpa-onnx 1.13.6, the version the app vendors), the app's Parakeet TDT 0.6B v2 fp16 and Silero VAD models in `~/.local/share/WhisperNow/models/`, and Kokoro `kokoro-int8-multi-lang-v1_0` (Apache-2.0, from the k2-fsa/sherpa-onnx `tts-models` release) unpacked in `.cache/tts/`. `--model whisper` and `--model qwen3` need `sherpa-onnx-whisper-distil-large-v3.5` (the app's catalog build, same SHA-256) and `sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25` from the k2-fsa/sherpa-onnx `asr-models` release, unpacked in `.cache/asr/`.
- `heard run` speaks with Kokoro on an NVIDIA GPU: `uv sync --inexact --extra audio-gpu` swaps in sherpa-onnx 1.13.6's CUDA 13 build (it conflicts with `audio`, and `--inexact` keeps the `train` extra), which needs the CUDA 13 runtime, cuBLAS and cuDNN 9 on the system, and the float Kokoro `kokoro-multi-lang-v1_0` (same release, same voices) in `.cache/tts/`. `--tts-provider cpu` uses the int8 model instead. On an RTX 5070 the float Kokoro runs at about 27x real time against 1.1x for the int8 on the CPU, and Parakeet fp16 at 88x against 20x with transcripts within 0.2% of words of the CPU's (`--asr-provider cuda`); Whisper int8 gains little on the GPU and stays on the CPU. On a 12 GB GPU only Parakeet and two Whisper shards fit: each process's onnxruntime arena only grows (Parakeet's to about 6 GB, each Kokoro's to about 3 GB), a third shard ran out of memory, and even two run out now and then over hours, so the 10,000-row run restarted crashed processes (a restart resumes past the rows written). It took about 10 hours: Parakeet about 700 rows an hour, each Whisper shard about 150.
- Data lands in `data/` and the response cache in `.cache/`. Both are gitignored.

## Build it, one command per stage

| Step | Command | Writes |
|---|---|---|
| 1. Import aawaaz | `uv run datagen import-aawaaz` | `data/stage1/aawaaz.imported.jsonl` |
| Copy the teacher prompt from Rust | `uv run datagen sync-teacher` | `prompts/teacher/` |
| Judge every aawaaz row | `uv run datagen judge-aawaaz` | `data/stage1/judgments.jsonl` |
| Judge register (casual, creative, legal, email, medical, meeting notes) | `uv run datagen judge-register` | `data/stage1/register_judgments.jsonl` |
| Teacher agreement check | `uv run datagen teacher-check --label v5-low` | `data/stage1/teacher_check/` |
| Relabel or drop flagged rows | `uv run datagen fix-aawaaz` | `data/stage1/aawaaz.clean.jsonl`, `report.json` |
| 2. Add handy and disfl_qa | `uv run datagen add` | `data/stage2/*.jsonl`, `report.json` |
| 3. Generate gaps and hard cases | `uv run datagen generate` | `data/stage3/<category>.jsonl` |
| 3b. Extra test rows for the small categories | `uv run datagen generate --test` | `data/stage3/<category>.test.jsonl` |
| Hear what the recognizers write (speaks with Kokoro, then transcribes as the app does; `--model whisper` or `qwen3` re-transcribes the kept audio; `--report` only rewrites the report) | `uv run --extra audio datagen asr-probe --n 300` | `data/probe/<model>*.jsonl`, `report.<model>.json` |
| Time the app's speech models on 40 probe dictations and score their WER (`--model <name>` for one) | `uv run --extra audio datagen asr-speed` | `data/probe/asr_speed.json` |
| 7. Restyle half the aawaaz inputs (`--recheck` re-checks rejects, no calls) | `uv run datagen restyle` | `data/restyle/aawaaz.jsonl`, `rejected.jsonl` |
| 110.3 Filler formatting and correction cues: real dictations, the build, the injection, any files | `uv run datagen profile [--injection] [FILE …]` | stdout |
| 110.4 Long benchmark, generated rows (equal thirds of 150–300, 300–600 and 600–1,000 words) | `uv run datagen compose --run bench --total 360` | `data/bench/long/<category>.<bucket>.jsonl` |
| 110.4 Long benchmark, real AMI meeting turns (text only, CC BY 4.0) | `uv run datagen bench ami` | `data/bench/long/ami.jsonl` |
| 110.4 Noise benchmark, nine patterns and a control set | `uv run datagen bench noise` | `data/bench/noise/<pattern>.jsonl` |
| 110.4 Speak a third of the noise rows with Kokoro and transcribe them with Parakeet | `nice ionice -c3 uv run --extra audio datagen bench probe` | `data/bench/noise/probed.jsonl` |
| 110.4 New references for the probed rows, through the gates | `uv run datagen bench gate-probed` | `data/bench/noise/probed.gated.jsonl` |
| 110.4 Write both benchmark files, and check them against the build | `uv run datagen bench assemble` | `data/bench/{long,noise}.jsonl`, `report.json` |
| 110.5 Long composed training rows (40/35/25% by bucket). `gpt-6.1-sol` through codex writes the transcripts (low) and references (high; xhigh over 600 words). The judge checks one row per request (low up to 600 words, high above), and a failure is relabelled once by the teacher. Each cell runs `compose.LANES` batches at once (2, 6 and 12 by bucket), so the slow long cells keep the codex slots busy. The first 2,558 rows had `glm-5.3` as generator, teacher and judge | `LLM_BACKEND=codex LLM_MODEL=gpt-6.1-sol LLM_REASONING=high LLM_GENERATOR_MODEL=gpt-6.1-sol LLM_GENERATOR_REASONING=low LLM_JUDGE_MODEL=gpt-6.1-sol LLM_JUDGE_REASONING=high LLM_CODEX_CONCURRENCY=64 uv run datagen compose --total 8300` | `data/stage4/<category>.<bucket>.jsonl` |
| v2: targeted long rows, more composed training rows weighted toward the categories and lengths where the fine-tuned models lose the most facts (`compose.TARGETED_WEIGHTS`: revised_choice 32%, recipe_quantities 22%; `TARGETED_SHARES`: 20/40/40% by bucket), from name and setting pools shared with neither the training run nor the benchmark. Rows are tagged `targeted` and `v2` | `LLM_BACKEND=codex LLM_MODEL=gpt-6.1-sol LLM_REASONING=high LLM_GENERATOR_MODEL=gpt-6.1-sol LLM_GENERATOR_REASONING=low LLM_JUDGE_MODEL=gpt-6.1-sol LLM_JUDGE_REASONING=high LLM_CODEX_CONCURRENCY=64 uv run datagen compose --run targeted --total N` | `data/stage4-targeted/<category>.<bucket>.jsonl` |
| v2: heard rows. Kokoro speaks a sample of v1 train transcripts and Parakeet (two thirds) or Whisper transcribes them as the app does; the row keeps its origin's reference after `restore` puts back what the recognizer lost, and `check` (Mercury Decide, free) drops rows whose heard text reads differently from the spoken (see [Heard rows](#heard-rows-v2)). Every process needs the same `--n`; `--shard`/`--shards` split one recognizer's rows; `report` summarizes the records, `restore` reapplies changed rules to the saved heard text (run `check` again after it), and `audit` has the judge check kept rows against their origins | `nice -n 10 ionice -c3 uv run --extra audio-gpu datagen heard run --n N --recognizer parakeet --asr-provider cuda`, and two of `… --recognizer whisper --shard K --shards 2`; then `uv run datagen heard check` | `data/heard/heard.<recognizer>.<shard>.jsonl`, `meaning.jsonl`, `audit.jsonl` |
| 112. Rewrite every reference with `gpt-6.1-sol` at high reasoning (teacher and judge; the command refuses any other setting for any role). `build`, `bench-long`, `bench-noise` or `all` is the full run; `--retry N` checks the dropped rows again and rewrites the ones that still fail, with the judge's notes from this round and the one before (rows the user ruled dropped stay dropped); `--settled-hedges` with `--retry` also rechecks passed rows that keep a hedge the speaker may have settled; `--report` only compares old and new references | `LLM_BACKEND=codex LLM_MODEL=gpt-6.1-sol LLM_REASONING=high LLM_JUDGE_BACKEND=codex LLM_JUDGE_MODEL=gpt-6.1-sol LLM_JUDGE_REASONING=high LLM_GENERATOR_BACKEND=codex LLM_GENERATOR_MODEL=gpt-6.1-sol LLM_GENERATOR_REASONING=high LLM_CODEX_CONCURRENCY=64 uv run datagen relabel all` | `data/relabel/<scope>.jsonl`, `data/relabel/inputs/` |
| 4. Filter, deduplicate, split, report | `uv run datagen build` | `data/build/{train,val,test}.jsonl`, `stats.md` |
| v2 build: dataset v1 (`data/build` and both benchmarks as they stood on 2026-10-18, frozen) plus the v2 rows in train. v1's rows are copied unchanged, val and test byte for byte, except that a heard row replaces its origin's train row; v2 rows skip the relabel (the same teacher and gates wrote them), get injected long versions from a pool of their own, and are dropped when they fail a filter or duplicate a v1 row. `--no-heard` keeps every v1 input, to measure what the heard swaps add | `uv run datagen build --v2 [--no-heard]` | `data/build-v2/{train,val,test}.jsonl`, `stats.md`; `data/build-v2-no-heard/` |
| Draw rows for a hand review | `uv run datagen sample list --n 50 --seed 110` | stdout |
| Publish (private) | `uv run datagen publish <user>/<repo>` | Hugging Face dataset |
| Real-dictation sanity check (not an evaluation set) | `uv run datagen export-real` | `data/real/real.jsonl` (never uploaded) |

**Every LLM step resumes.** The judge commands append verdicts as batches finish and skip rows already judged. `generate` logs each batch in `<category>.batches.jsonl` and continues from there. Each response is cached by its exact request in `.cache/llm/`, so a rerun repeats no finished call. `generate --scale 0.05` runs a pilot at 5% of each target. `generate --category list_edits` runs one category.

**The relabel (todo 112)** sends each transcript in the app's own request (`teacher.teacher_messages`), never showing the old reference. The gates check the new reference in code (protected tokens kept, no em dash, no correction cue kept from the transcript) and then with the completeness judge, which also gets code's flags: spelled-out numbers next to a word, and hedges or "because" the reference says less often than the transcript. Rows over 300 words get a judge request each; benchmark rows always do. A failure is written once more with the judge's note added to the request. `--retry N` requires N to exceed every earlier retry round and checks the rows still dropped: it checks their last reference again, since the gates may have changed, and gives each that still fails one more rewrite with the new note. Rows that fail every round stay dropped. The first run copies the rows it relabels to `data/relabel/inputs/` (the build's injected rows apart, for the stats only). Once a scope has records, `build` and `bench assemble` take every reference from them, drop the rows that failed, refuse to run while a scope is partial; the injected rows are made again from the relabelled parents and inherit `teacher_model` and `teacher_reasoning`. `stats.md` then compares old and new references by source. Its hedge and reason keep rates leave out the occurrences a reference may drop: a hedge said again straight away, a closing aside ("I think that's everything") and one followed within 10 words by a correction cue; `report.json` keeps the raw rates as `kept_raw`.

`build` also makes the programmatic rows: filler injection and `EMPTY`. Last, after the splits, `spoken_variant` varies the spoken inputs in place, with no LLM:

- **Recognizer styles:** each composed recognizer-style row gets Parakeet's, Whisper's or Qwen3's punctuation, or keeps the generator's text (`plain`). The shares are the restyled aawaaz rows' (`restyle.ASR_STYLES`).
- **Densify (`inject.densify`):** 50% of composed rows, and 30% of the train and val rows of the stage 3, handy and disfl_qa sources, get more fillers, clusters, "the uh the" repeats and repeated words, up to 3–6 fillers per 100 words. Only the Parakeet rows also get pause punctuation. These rows are tagged `densified`.
- **References:** every change is one the reference drops or rewrites anyway, so references stay right without a judge.
- **Left as built:** todo 109's test rows, aawaaz and the injected rows.
- **Per epoch:** the step is a pure function of the row and a salt, so training can pass `salt=f"epoch{n}"` to draw fresh variants of the train rows each epoch.

## The LLM client

`datagen/llm.py` reads `LLM_*` from `.env`:

- **Concurrency.** A semaphore per backend caps requests in flight (`LLM_CONCURRENCY=3`, and `LLM_CODEX_CONCURRENCY=16` for codex).
- **Retries.** Rate limits, 5xx and timeouts retry up to 7 times, with exponential backoff (4 s doubling, capped at 180 s) and jitter.
- **Batched judging.** Judges see 10–30 rows per request.
- **Reasoning.** Only the final message text is kept. `<think>` blocks and reasoning parts are dropped.
- **Models.** `LLM_MODEL` (`glm-5.3-flash`) is the teacher. `LLM_JUDGE_MODEL` (`glm-5.3`) is the judge and validator. `LLM_GENERATOR_MODEL` writes transcripts and defaults to `LLM_MODEL`.
- **Backend per role.** `LLM_BACKEND` serves the teacher. `LLM_JUDGE_BACKEND` and `LLM_GENERATOR_BACKEND` default to it. `LLM_GENERATOR_REASONING` defaults to `LLM_REASONING`.

**Generator pool.** `LLM_GENERATORS` spreads the generator role over several routes, comma-separated, each `provider:model#reasoning` (`llm.PROVIDERS`):

| Provider | Route | Key | In flight |
|---|---|---|---|
| `nvidia` | NVIDIA NIM, `https://integrate.api.nvidia.com/v1` | `NVIDIA_API_KEY` | 8 |
| `orca` | Orca Router, `https://api.orcarouter.ai/v1` | `ORCAROUTER_API_KEY` | 4 |
| `openrouter` | OpenRouter directly | `OPENROUTER_API_KEY` | 4 |
| `codex` | `codex exec`, sharing the codex semaphore | codex's own login | 8 |

- The OpenAI-compatible routes send the reasoning as `reasoning_effort`. On NVIDIA's glm-5.3-flash it changes how long the model thinks. On OpenRouter's Nemotron and Orca's DeepSeek only `none` (no reasoning) made a measurable difference in our probes: low, high and max each thought for 2,000–7,000 tokens with no order between them.
- **The pool is not used.** A pilot on the targeted run (2026-10-18, `gpt-6.1-sol` as teacher and judge) found no route worth its Codex cost:
  - Writing transcripts is about 17% of the Codex spend per transcript (measured with `codex exec --json`, weighting cached input 0.1× and output 8×). The teacher at xhigh on long rows costs about 4 times a generator request.
  - So another generator saves Codex only if it keeps at least about 70% of its transcripts (`gpt-6.1-sol` keeps 84% of its own).
  - Kept: Orca DeepSeek 68% with reasoning off and 55% with it on; NVIDIA glm-5.3 and glm-5.3-flash 7–33%; OpenRouter Nemotron 6–13%.
  - Most of the losses were the layout check: the GLM models label 62–78% of these list-heavy dictations `prose` (`gpt-6.1-sol` 12%), and the teacher writes a list. Judged without the check, about a third of those rejects would pass, for `gpt-6.1-sol`'s own rejects as for the others'.
  - The 93 rows the pilot kept are in `data/stage4-targeted/`, each naming its generator.

- `LLM_<PROVIDER>_CONCURRENCY` overrides a provider's slots, which its routes share.
- Each request goes to the least-used route that is not resting and whose provider has a free slot.
- A route rests after a rate limit (`Retry-After`, else 30 s doubling to 10 minutes), after an account or quota refusal (30 minutes, so a free tier the account qualifies for later joins by itself), or after three failures in a row (2 minutes). A configuration error (HTTP 400 or 404, such as a bad model name) takes it out of the run.
- The request then moves to another route. Rate limits and refusals don't count against it; other failures count toward the usual 7.
- `LLM_GENERATOR_FALLBACK` names a route that takes requests only while every pool route rests or is out, not while they are merely busy.
- A cached reply keeps the route that wrote it, so a rerun reuses it whatever the pool holds now.
- Every generated row records its generator (`generator_provider`, `generator_model`, `generator_reasoning`), teacher (`teacher_model`, `teacher_reasoning`) and judge (`judge_model`, `judge_reasoning`). Rejected rows and batch records carry the route's label as `generator`, so pass rates split by route. An injected row keeps its origin's fields.

**Backend.** `LLM_BACKEND=openai` calls any OpenAI-compatible endpoint directly with `LLM_BASE_URL` and `LLM_API_KEY`.

- An OpenRouter model ending in `:batch` (for example `anthropic/claude-haiku-5.5:batch`) is served only through OpenRouter's batch API, at half the per-token price. `cleanup eval --teacher` sends every row it needs as one batch and waits for it, which can take up to 24 hours. The batch's id is kept under `.cache/llm-batches/`, so a rerun waits for the same batch instead of paying for another. A batch that expires after 24 hours stops the run with an error, keeping whatever it answered; send the rest through the model without `:batch`.
- `cleanup bench --teacher` times single requests, so it needs the model without `:batch`.

`LLM_BACKEND=codex` runs one `codex exec` per request.

- Each request runs from an empty temporary directory, with a read-only sandbox, no web search and no session saved.
- The user's codex config, rules and any `AGENTS.md` are ignored.
- Our system message replaces codex's own instructions (`model_instructions_file`). The user message goes on stdin, and only the last message is kept.
- **Caveat:** codex still declares its tools and adds its environment note, about 4,000 tokens per request.
- Each call runs in its own process group (`llm.run_process_group`). `codex` is a node wrapper around a native binary, and a binary left behind holds the pipes open, which once hung a run for 70 minutes. On a timeout or cancellation, or 10 s after the wrapper exits with its pipes still open, the whole group is killed.
- The reasoning setting is codex's `model_reasoning_effort`: low, medium, high, xhigh or max.

`LLM_BACKEND=claude` runs one `claude -p` per request, under `LLM_CONCURRENCY`.

- Each request runs from an empty temporary directory with `--safe-mode` (no CLAUDE.md, skills, plugins or hooks), no MCP servers, no tools and no session saved.
- Our system message replaces Claude Code's (`--system-prompt`). The user message goes on stdin, and only the final result is kept.
- **Caveat:** Claude Code still adds a line naming it a Claude agent to the system message, a reminder with the account's email before the user message, and an environment note (working directory, platform) after it. A clean environment doesn't remove them.
- The reasoning setting is `--effort`: low, medium, high, xhigh or max.

**Thinking.** GLM 5.3 models always reason. `LLM_REASONING` and `LLM_JUDGE_REASONING` set the effort. We run both at `low`; see the teacher agreement section.

### Timing

Measured with 3 requests in flight:

| Call | Median | Mean | p90 |
|---|---|---|---|
| Teacher, one transcript, low reasoning | 3.5 s | 6.9 s | 15.7 s |
| Teacher, one transcript, high reasoning | 13.5 s | 17.5 s | |
| Judge, 15 rows | 8.7 s | 11.1 s | 16.2 s |
| Generator, 15 transcripts | 35 s | | |

Projected wall-clock times:

- **Stage 1 judge:** 22,772 rows in 1,519 batches of 15, about 16 batches a minute, so about 95 minutes. One row per call would have taken over 20 hours.
- **Stage 2 audit:** about 15,000 rows in 506 batches of 30, about 40 minutes at full concurrency.
- **Stage 3 generation:** about 3,550 kept rows. The pilot pass rate of 50–70% means about 6,500 transcripts. That is about 130 generator calls at about 110 s, 6,500 teacher calls at about 5 s, and 650 validator calls at about 11 s: roughly 15 hours of calls, or about 5 hours at 3 in flight.

Actual wall-clock times, on one account:

- **Stage 1 judge:** about 2.5 hours, sharing the account with the stage 3 pilot. The register pass and the relabel pass took about 1 hour 50 minutes more (23:16 to 01:04).
- **Stage 2 audit:** about 3.5 hours at 1 in flight, run alongside stage 3.
- **Stage 3 generation:** about 4 hours, from 22:27 to 02:33. It ran at 2 in flight while stages 1 and 2 ran, and at 4 afterwards. The pass rate was 71–95%, better than the pilot, so it needed 4,214 transcripts in 95 batches. A batch took about 20 minutes at 2 in flight and 3–7 minutes at 4.
- **Provider stall:** once, every request hung for about 15 minutes. The 900 s request timeout and retry recovered it.

## Terms and licenses

This is a reading of the published terms, not a legal review.

- **A row's `license` is its text's license** (`schema.OWN_LICENSE`). An imported row keeps its dataset's license, though `gpt-6.1-sol` rewrote its reference. Text made for this dataset (generated transcripts, injected fillers, `EMPTY` rows) is ours, under the repository's MIT license. The license does not name the models that wrote a row; the providers' terms below apply all the same.
- **Z.ai (teacher, judge and generator outputs).**
  - Z.ai's terms of use forbid using outputs "to develop, train, or enhance algorithms or models that compete with us", and forbid automated access by bots or spiders.
  - Whether a local dictation-cleanup model competes with Z.ai is a judgment call for the project owner. Rows whose output the teacher wrote carry `teacher_model`.
- **OpenAI (`gpt-6.1-sol` through the Codex CLI, todo 110's composed rows).**
  - OpenAI's terms of use also forbid using output to develop models that compete with OpenAI. Whether this model competes is the same judgment call as for Z.ai.
  - `gpt-6.1-sol` wrote every reference in the build and both benchmarks. Z.ai's models wrote the stage 3 transcripts and the first composed ones, and `glm-5.3-flash` restyled half the aawaaz inputs.
- **v2 transcript generators.** Each row names its generator, so the terms of the model that wrote its transcript can be checked per row.
  - NVIDIA NIM serves `z-ai/glm-5.3` and `z-ai/glm-5.3-flash` with an API key; Z.ai's terms on GLM outputs (above) apply whichever host serves the model.
  - OpenRouter's `nvidia/nemotron-3.5-super-vl:free` takes direct API calls.
  - Orca Router's `deepseek/deepseek-v4.1-flash-free` needs the workspace owner to have linked an established GitHub account; before that it answers HTTP 429 `free_rate_limited`, and its route rests and retries every 30 minutes.
- **aawaaz** (`shantanugoel/aawaaz-transcript-cleanup-dataset`): MIT. Its outputs came from Gemini 2.5 Flash and coding agents, so check Google's output terms before any commercial release.
  - Its generator repo has no license file, so our generator, validator and category prompts are written from scratch. They follow its loop but copy none of its text.
- **handy** (`MagicNoThief/handy-dictation-editing`): CC-BY-4.0.
  - Its real-speech rows re-record Switchboard-1 (LDC) transcripts. We drop every row that shares a whole line, or at least a quarter of its word 6-grams, with the DisfluencySpeech transcripts (`amaai-lab/DisfluencySpeech`, `nyralabs/disfluency_speech_english`). That removed 7,077 rows.
  - Only the transcript columns of those parquet files are read, never the audio.
- **disfl_qa** (`google-research-datasets/disfl_qa`): CC-BY-4.0. Its questions derive from SQuAD 2.0 (CC-BY-SA-4.0), so attribution and share-alike may carry over.
- **sotto** (`juanquivilla/sotto-transcript-cleanup`): MIT, but partly written by Grok 4.20. xAI's terms bar using outputs to train competing models, so `add` leaves it out unless you pass `--with-sotto`.
- **Excluded entirely:** no non-commercial data (such as JFLEG) and no LDC data (Switchboard mirrors).
- **Programmatic rows:**
  - Filler injection inherits its origin row's license.
  - `EMPTY` rows are ours (MIT).

## Real transcripts: what the inputs look like

The local history (`~/.local/share/WhisperNow/history.sqlite3`, still on the older recordings/attempts schema) has only 20 transcripts:

- 7 are from Parakeet TDT 0.6B v2 and 13 have no model recorded.
- They average 6 words.
- All 7 Parakeet transcripts are punctuated and cased.
- Numbers come out as words ("three") in 6 of 20, and none contain digits.
- Only one has a filler, without a comma.

That is too few to measure filler style per model. The inputs we generate cover both styles: `recognizer` (punctuated, cased, mostly digits, bare fillers) and `raw` (aawaaz-like: lowercase, little punctuation, numbers as words).

**Second look (todo 110).** The current history (`entries` schema) held 19 real Parakeet dictations of 9 or more words, about 540 words, all from one speaker. Only aggregate counts were taken; no text left the machine. Compared with the composed long rows:

| per 100 words | real | composed, as generated | composed, densified |
|---|---|---|---|
| um/uh | 3.6 | 0.8 | 4.3 |
| "the uh the" | 0.36 | 0.05 | 0.50 |
| repeated words | 0.71 | 0.71 | 1.07 |
| fillers with a comma after | 5% | 14% | 1% |
| words per sentence | 11.4 | 16.5 | 15.8 |

The generator writes bare fillers in the right places but about a fifth as many as real speech, so `build` densifies a share of the inputs (`spoken_variant`, above). Real fillers sit where the speaker plans what comes next: at a sentence start, after a conjunction, discourse word or comma, and after a preposition, verb or determiner before the phrase it introduces. The injector weights its slots that way and never puts a filler inside a name, a number or a spoken symbol.

## Stage 1: aawaaz

`import-aawaaz` reads `shantanugoel/aawaaz-transcript-cleanup-dataset` (22,772 rows after normalization) and holds out 3% per category as test (736 rows kept after cleaning). Normalization strips markdown bold only in word-bounded pairs, so content asterisks such as `6-31G**` and `x**2` survive; `tests/test_aawaaz.py` checks this on real rows.

Two judges on `glm-5.3` read every row:

- **`judge-aawaaz`**, all 22,772 rows, on four criteria (`prompts/audit/aawaaz.md`): content preserved (including hedges such as "maybe"), no hallucination, lead-in kept, and not answered.
- **`judge-register`**, the 11,866 rows of `casual_conversation`, `creative_writing`, `legal_contract`, `email_professional`, `medical_clinical` and `meeting_notes`, on register kept (`prompts/audit/aawaaz_register.md`): no shortening into notes or a chart summary, and no added formality. The first three categories came out of the 100-row hand review below; the last three were added after the first 50-per-layout review of the build (see [Build review](#hand-review-of-the-build)). The judge flagged casual 919, creative 6, legal 39, email 12, medical 76 and meeting notes 35.

`fix-aawaaz` relabels every flagged row with the teacher and re-judges it on the same criteria. A relabel that passes replaces the row (tagged `relabeled`, with `teacher_model` set); one that fails is dropped.

| Category | Rows | Flagged | Fixed | Dropped |
|---|---|---|---|---|
| academic_research | 2,000 | 10 | 10 | 0 |
| casual_conversation | 2,000 | 945 | 937 | 8 |
| creative_writing | 1,941 | 27 | 26 | 1 |
| email_professional | 1,914 | 67 | 63 | 4 |
| financial_business | 2,000 | 10 | 10 | 0 |
| legal_contract | 2,008 | 69 | 67 | 2 |
| medical_clinical | 2,007 | 95 | 94 | 1 |
| meeting_notes | 1,994 | 91 | 79 | 12 |
| recipe_cooking | 959 | 61 | 57 | 4 |
| self_corrections_heavy | 1,996 | 44 | 39 | 5 |
| shopping_lists | 1,954 | 348 | 301 | 47 |
| technical_code | 1,999 | 8 | 8 | 0 |
| **Total** | **22,772** | **1,775** | **1,691** | **84** |

Failed criteria (a row can fail several): register_kept 1,087, content_preserved 573, lead_in_kept 151, no_hallucination 125, not_answered 6.

- **Shopping lists** fail most on content (17.8%), almost always a dropped hedge: "maybe ten large boxes", "I think we have baking soda, get just in case".
- **Kept:** 22,688 rows (736 of them test): plain 14,410, paragraphs 3,295, mixed 2,631, list 2,352.

The counts are in `data/stage1/report.json`.

**Hand review of 100 cleaned rows**, 25 per layout (seed 100), read after the first fix pass:

- **plain.**
  - **casual_conversation stories were shortened into notes** in 3 of the 25 ("Went to a sake and sushi pairing dinner. Chef prepared six courses…"), dropping subjects, articles and some details. That breaks the prompt's own rule ("do not shorten sentences into notes"), and the four stage 1 criteria never checked it. Counting rows that lose most of the speaker's "I"s found it in 30% of casual_conversation (591 of 2,000) and rarely anywhere else. The register judge flagged 919, and after relabeling the count is 24.
  - **legal_contract outputs raise the register**: "do a mutual release" becomes "execute a mutual release" and "pays their own legal fees" becomes "shall bear its own legal fees". The register judge flagged 39 such rows.
  - One self-correction row lost a real detail ("My wife says our free night is Wednesday").
- **list.** Good overall. aawaaz invents group labels ("Basics:", "Pantry:") and normalizes item names ("a little potty seat the one that goes on top of the regular toilet" becomes "Toilet seat adapter (toddler)"). The teacher prompt accepts both styles. One relabeled meeting note has an empty "Attendees:" heading; only 13 rows across stage 1 have a label with nothing under it, and some of those are legitimate "Ingredients:" headings over groups.
- **paragraphs.** Good: emails, letters and clinical notes with sensible breaks. Mild register lifts: expanded contractions ("She is"), "gonna" as "Going to", "and all that" as "and all related procedures".
- **mixed.** Mostly meeting notes in aawaaz's house style: a title, an attendees line and topic headings over the speaker's sentences, with action items as bullets. One row nests bullets, and 1,479 rows have indented sub-items; the prompt does not forbid them, so they stay. One relabeled recipe leaves a dangling fragment ("…but curry laksa specifically.").
- **Not found in the 100:** answered transcripts, markdown, or changed protected tokens.

After the register fix, the relabeled casual_conversation stories read as the speaker's own sentences. A few still open with "So", which the prompt asks to drop.

## Teacher prompt and agreement

The teacher is `LLM_MODEL` sent exactly what the app sends a cleanup provider. That is `CLEANUP_SYSTEM_CONTRACT` as the system message, and `CLEANUP_PROMPT` wrapped as `build_cleanup_prompt` does. `prompts/teacher/` holds copies, and `tests/test_teacher.py` fails if they drift from the Rust source.

The check runs the teacher on 50 held-out aawaaz rows, spread across categories. `glm-5.3` then compares each output with the aawaaz reference on layout, lead-ins and kept facts (`prompts/judge/agreement.md`).

| Prompt | Reasoning | n | All agree | Layout | Lead-ins | Facts | Same layout class |
|---|---|---|---|---|---|---|---|
| v1: first rewrite to the aawaaz style | low | 50 | 0.84 | 0.90 | 0.94 | 0.96 | 0.86 |
| v2: revised after v1's disagreements | low | 50 | 0.80 | 0.92 | 0.94 | 0.90 | 0.88 |
| v3: recipes keep side remarks as sentences, meeting notes list the speaker, reports and explanations stay prose | low | 50 | 0.84 | 0.92 | 0.98 | 0.92 | 0.88 |
| v3 | high | 50 | 0.88 | 0.90 | 0.94 | 1.00 | 0.84 |
| v4: lists inside messages ("get, bring, or do") | low | 50 | 0.76 | 0.92 | 0.86 | 0.88 | 0.92 |
| **v5: lists inside messages ("get or bring")** | low | 100 | 0.82 | 0.90 | 0.91 | 0.93 | 0.83 |

On the same first 50 rows, v5 agrees on 0.82 against v3's 0.84. With n = 50, one row is 0.02 and the standard error is about 0.05, so these prompts are within noise of each other on aawaaz.

**Why v5.** Its added sentence is about messages, which aawaaz barely has. The stage 3 pilot showed that v3 kept a message's shopping or bring-list as prose ("while you're at the store can you get toothpaste, the good coffee, bananas and dish soap"). The todo wants that part laid out as a list. On the pilot's 15 mixed-message inputs:

- v5 laid out 9 as lists, and kept the status recaps and blocker summaries as prose.
- v4's broader "or do several things" also turned a clinical plan into bullets, so v5 narrows it.

**What the judge still flags**, in all versions:

- missing group labels in long notes ("Toppings:", topic labels in meeting notes);
- a long single paragraph split in two, or the reverse;
- a few dropped details, such as an opening sentence or cooking oil;
- once, "$11,800" written as "$11.80".

The stage 1 judge and the stage 3 validator catch these in the data.

**Reasoning: low.** High reasoning raised agreement on v3 by 2 rows of 50 (facts 1.00 against 0.92) but took 5 times as long per call: a mean of 17.5 s against 3.6 s. Every teacher output is then judged, and flagged rows are dropped or relabeled. So we keep `LLM_REASONING=low` and let the judge filter. The v3-high run predates the separate judge setting, so its judge reasoning was not recorded.

## Stage 2: handy and disfl_qa

`add` loads handy's alpaca config and disfl_qa, drops what conflicts with the target rules, samples by category with a fixed seed, and has `glm-5.3` audit every sampled row (`prompts/audit/handy_disfl_qa.md`). Rows whose only failure is a list kept as prose are relabeled by the teacher and re-judged; other failures are dropped. sotto is off by default (see Terms).

**Hand audit of 50 random rows per set** (seed 50), before the fixes below:

- **handy.**
  - Most rows are fine: short requests, no-edit rows, self-corrections resolved to the right side.
  - **Wikipedia sentences spliced into lowercase dictation** (for example "i think Thus, each such firm may offer itself as…") in 7 of 50. No recognizer writes that, and one output had wrongly capitalized "may" as the month "May". Dropped: 3,503 rows across the dataset.
  - **British spelling.** Converted to American with a word list (`americanize`), tagged `americanized`.
  - **Numbers left as words** ("seven chairs" is fine, but "the fifteenth" and "twenty percent" conflict with our written forms). Day ordinals after "the" or "on" are written as digits ("on the 15th", tagged `normalized`). Rows still holding larger number words, times or percents in words are dropped: 13,029.
  - **Questions ending in a period** ("What time does the Paris office open on weekdays."). A one-sentence wh-question gets its question mark.
  - **The verb "may" capitalized as the month** in kept rows too ("factors May contribute"). `build` lowercases it when the words around it are not a month's ("in May", "May and"), tagged `normalized`: 21 rows.
  - Switchboard-derived text: see Terms (7,077 dropped).
  - handy's own dictated questions are copies of disfl_qa questions, so questions come from disfl_qa only.
- **disfl_qa.**
  - Most rows are clean self-corrected questions.
  - **Rewording beyond removing the disfluency** in 4 of 50: "what number of students" became "How many students", "sea life" became "animals", and one kept the wrong side of the correction. Rows whose output uses a word the speaker never said are dropped: 1,330.
  - **SQuAD's own typos**: no question mark, lowercase start, a stray `\t`. We capitalize, add the question mark, and drop rows with a backslash.

Kept per category, before the audit: handy self_correction 3,500, short_dictation 3,000, no_edit 3,000, dictation 1,500 (plus 878 of its own test rows), and disfl_qa 3,000 (plus 300 test).

**Audit results** (`data/stage2/report.json`, judge `glm-5.3`):

| Source | Dropped before audit | Audited | Flagged | Relabeled | Kept |
|---|---|---|---|---|---|
| handy | Switchboard 7,077; number words 13,029; spliced written text 3,503 | 11,878 | 186 | 9 | 11,701 |
| disfl_qa | reworded 1,330; number words 52 | 3,300 | 64 | 0 | 3,236 |

- **handy's flags:** content_preserved 124, no_hallucination 58, fillers_removed 15, layout_ok 9. All 9 layout flags were relabeled and kept. Five of them turned "we need to book the room order lunch for eight and print the agenda" into a `We need to:` list, and four became questions with question marks.
- **disfl_qa's flags:** content_preserved 40, fillers_removed 17, no_hallucination 13. All of these were dropped.
- **Kept per category:**
  - handy: self_correction 3,786, short_dictation 3,204, no_edit 3,078, dictation 1,633. These include handy's own test rows.
  - disfl_qa: dictated_question 3,236.
- **Near-duplicates.** handy has templated near-duplicates, such as "order lunch for four" and "order lunch for five". The build's near-duplicate pass removes them.

## Stage 3: generated gaps and hard cases

Each batch runs the aawaaz loop as a plain script:

1. **Generate.** A fresh request to `LLM_MODEL` for about 50 transcripts as JSON (25 for `long_ramble`, whose batch of 50 took 12 minutes). It gets:
   - `prompts/generate/base.md`;
   - one input style (batches alternate `recognizer` and `raw`);
   - the category's guidance in `prompts/categories/`;
   - a seeded pick of names and settings;
   - the previous batch's rejection notes.

   Each transcript comes back with the layout the generator intends.
2. **Label.** The teacher writes each output, exactly as the app would.
3. **Check layout.** Automatically: a list where the generator meant prose, or prose where it meant a list, is rejected. `ambiguous_intent` leaves this to the validator.
4. **Validate.** `LLM_JUDGE_MODEL` checks the pairs in tens on six criteria (`prompts/judge/generate.md`): realism in the stated style, content preserved, no hallucination, not answered, layout, and clean.
5. **Keep or reject.** Passing rows are appended. Rejected rows and their notes go to `<category>.rejected.jsonl`, and the notes feed the next batch.

| Category | Target | What it covers |
|---|---|---|
| `dictated_command` | 500 | Commands and requests dictated as text, including ones buried mid-dictation |
| `prose_mentions` | 750 | Formatting negatives: stories, comparisons and "first… then…" narration that name things but stay prose |
| `spoken_symbols` | 500 | Spelled-out symbols, spoken punctuation and layout commands, spoken emails and URLs |
| `mixed_message` | 300 | A message with a list inside it |
| `list_edits` | 300 | "Scratch the eggs", "make that two bags", reordering |
| `long_ramble` | 300 | Long dictations switching between prose and lists |
| `command_question_list` | 300 | Lists of questions and instructions that must stay text |
| `technical_list` | 300 | List items with code, paths and flags |
| `ambiguous_intent` | 300 | Either layout could fit; the validator judges the choice |
| `filler_injection` (programmatic) | 2,000 | Fillers, repeats and openers injected into clean one-paragraph aawaaz outputs, in both styles; the clean text is the target |
| `empty` (programmatic) | 150 | Filler-only inputs whose target is `EMPTY` |

**Stage 3 results.** All nine categories reached their targets, 3,550 rows. Totals come from `<category>.batches.jsonl` and `<category>.rejected.jsonl`:

| Category | Batches | Returned | Passed | Pass rate | Top rejection reasons |
|---|---|---|---|---|---|
| `ambiguous_intent` | 7 | 332 | 305 | 92% | no_hallucination 17, layout_ok 8, content_preserved 6 |
| `command_question_list` | 8 | 377 | 306 | 81% | layout_intent 47, layout_ok 15, clean 4 |
| `dictated_command` | 11 | 549 | 508 | 93% | layout_intent 27, content_preserved 10, clean 5 |
| `list_edits` | 7 | 343 | 302 | 88% | content_preserved 24, clean 16, no_hallucination 8 |
| `long_ramble` | 16 | 363 | 305 | 84% | no_hallucination 24, layout_intent 19, clean 10 |
| `mixed_message` | 9 | 425 | 303 | 71% | layout_intent 79, content_preserved 24, no_hallucination 22 |
| `prose_mentions` | 16 | 800 | 758 | 95% | content_preserved 26, clean 11, layout_intent 5 |
| `spoken_symbols` | 13 | 636 | 505 | 79% | clean 57, layout_intent 32, content_preserved 26 |
| `technical_list` | 8 | 389 | 302 | 78% | layout_intent 78, content_preserved 4, clean 2 |

Passing rows past the target, or repeating an earlier input, are not kept.

- **The teacher leans toward prose.** 289 rejections were the automatic layout check, and almost all of them were the teacher keeping a list as sentences: 79 in `mixed_message` and 77 in `technical_list`. The other direction, a list where the speaker was not making one, happened mostly in `command_question_list` (20) and `dictated_command` (16).
- **`spoken_symbols` fails on `clean`.** The teacher leaves spoken symbols as words ("hash" not "#") or wraps code in backticks.
- **`content_preserved` notes** are mostly dropped emphasis words ("honestly", "actually") that the prompt does not list as fillers.

Fixes made during the run:

- **Batch size.** A 50-transcript `long_ramble` batch took 12 minutes and came back as 64 KB of JSON missing its closing brace, so `long_ramble` batches are 25.
- **Reply parsing.** `parse_json_reply` falls back to the outermost object or array when the reply has prose around it or a missing closing brace. `_items` salvages the well-formed `{"input", "layout"}` pairs when the generator writes something that is not JSON, such as a Python expression, inside the list.
- **Stall.** Once, every request hung for about 15 minutes at the provider. The 900 s request timeout and retry recovered it without losing a batch.

**Extra test rows.** The hashed 5% left the small categories with 8–31 test rows, so one row moved a rate by up to 12 points. `generate --test` wrote 680 more rows pinned to test (`TEST_TARGETS`), bringing each category to about 100 test rows and `long_ramble` to 48. Their batches are numbered after the training batches, so they draw new names and settings, and their inputs are checked against the training rows. `build` adds 90 `EMPTY` rows pinned to test from a second seed; 15 training `EMPTY` rows that repeated one of them were dropped. The run took about 40 minutes at 3 in flight:

| Category | Batches | Returned | Passed | Pass rate |
|---|---|---|---|---|
| `ambiguous_intent` | 2 | 97 | 90 | 93% |
| `command_question_list` | 3 | 116 | 91 | 78% |
| `dictated_command` | 2 | 85 | 75 | 88% |
| `list_edits` | 3 | 112 | 93 | 83% |
| `long_ramble` | 2 | 48 | 44 | 92% |
| `mixed_message` | 3 | 127 | 85 | 67% |
| `prose_mentions` | 2 | 81 | 80 | 99% |
| `spoken_symbols` | 3 | 114 | 81 | 71% |
| `technical_list` | 3 | 116 | 91 | 78% |

The test rows come from the same prompts as the training rows, so they measure the same distribution more precisely; they carry the same teacher habits, such as condensed list items and dropped hedge words.


## Input style (step 7)

Imported aawaaz inputs are lowercase and unpunctuated, but every recognizer the app ships punctuates: Parakeet TDT 0.6B v2 and v3 add punctuation and capitalization (their model cards say so), and so do the Whisper models. So half the aawaaz inputs are restyled to look like what the app's recognizer hands the cleanup model.

### What the recognizers write

`asr-probe` speaks 300 sampled inputs (55% aawaaz, half of those from the longest quarter; 20% teacher, 15% handy, 10% disfl_qa) with a random English Kokoro voice at speed 0.95–1.15, in chunks with 0.15–1.4 s pauses: 2.3 hours of audio. It then runs the app's path in Python, with the same Silero VAD settings and segmenter constants and the same segment join, through three recognizers on the same audio: Parakeet TDT 0.6B v2 fp16 (greedy search), Whisper distil-large-v3.5 int8 with the app's settings (automatic language, no tail padding), and Qwen3-ASR 0.6B int8. `data/probe/report.<model>.json` has the numbers:

| | Parakeet | Whisper distil | Qwen3 0.6B |
|---|---|---|---|
| Periods, commas per 100 words | 5.0, 6.1 | 4.7, 5.7 | 8.4, 7.4 |
| Sentence words (p25, p50, p75) | 8, 16, 26 | 9, 16, 27 | 5, 10, 16 |
| Numbers written as digits | 41% | 63% | 4% |
| Mid-sentence filler followed by a comma | 16% | 32% | 62% |
| "um" kept (of 365) | 354 | 335 | 317 |
| Spoken "period" kept as a word (of 240) | 236 | 236 | 238; 37% as its own sentence ("hear. Period. Our") |
| Segments ending with a period, starting lower case | 81%, 24% | 70%, 38% | 96%, 0% |
| Other | every "$" glued to the word before it ("that's$4,200") | 17% of segments all lowercase and unpunctuated | hyphenated number words ("thirty-one") |
| WER against the spoken text (p50) | 5.3% | 7.1% | 6.2% |

All three keep fillers ("uh" losses are Kokoro's: it says "uh" so that it is heard as "a" or "our"), keep spoken layout commands as words ("new line" 68–71 of 72), and sometimes write a spoken email or URL as an address. The app decodes long dictation in segments, about one per 75 words here, so a mid-sentence pause can leave a period and a new sentence ("I will. personally connect"). Synthetic speech is flatter than a person's, so treat the sentence-break numbers as rough.

### Speech model speed

`asr-speed` decodes every 7th probe dictation (40 of them, 1,032 s of segment audio after the app's segmenter) with each speech model in the app's catalog, using the app's recognizer settings: four CPU threads, greedy search. It needs every model in `.cache/asr/` (the app's FP16 Parakeet is read from the app's model directory). The catalog's speed ratings come from these numbers; accuracy comes from the Open ASR Leaderboard where a model is listed, because synthetic speech only ranks the models.

Measured on a Ryzen 9 8945HS on 2026-10-08. The GPU column was measured on an RTX 5070 through sherpa-onnx 1.13.6's CUDA 13 build (commit bf759d3):

| Model | × real time, CPU | × real time, GPU | WER on the probe (CPU / GPU) |
|---|---|---|---|
| Parakeet TDT v2 0.6B FP16 | 20.9 | 92.4 | 5.8% / 6.0% |
| Parakeet TDT v2 0.6B int8 | 27.5 | 22.0 | 6.0% / 6.1% |
| Parakeet Unified 0.6B int8 | 25.0 | 22.7 | 6.5% / 6.5% |
| Parakeet TDT v3 0.6B int8 | 25.9 | 23.1 | 6.8% / 6.7% |
| Parakeet TDT 110M int8 | 84.4 | 67.0 | 7.8% / 7.7% |
| SenseVoice Small int8 (2024-07-17) | 55.6 | 48.3 | 12.6% / 12.7% |
| Qwen3-ASR 0.6B int8 | 5.9 | 2.9 | 6.9% / 7.0% |
| Whisper distil-large-v3.5 int8 | 4.9 | 7.6 | 7.5% / 7.7% |
| Whisper large-v3-turbo int8 | 4.1 | 6.9 | 7.3% / 7.5% |

The app runs speech models on the CPU only: the int8 builds are slower on the GPU and use more host memory there, and only FP16 Parakeet and the Whisper models gained.

Three builds were tried and left out. Whisper large-v3 int8 ran slower than real time (0.5×; 6.3% WER), while Parakeet v2 is more accurate in English and Turbo covers the same languages. Moonshine Tiny (`sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27`) returns an empty result for any segment over about 9 s, and the app's segments run to 28 s. The 2025-09-09 SenseVoice build is a Cantonese fine-tune (WSYue) that writes English in garbled capitals (34% WER here).

### Restyle

`restyle` sends a hashed 50% of the cleaned aawaaz rows to the teacher with `prompts/restyle/recognizer.md`, eight rows or 5,000 characters a call. The prompt asks for a modern recognizer's style: sentence case, long sentences, spoken commands kept as words, numbers half digits and half words, and nothing else changed. `same_words` checks every rewrite in code: after casefolding, with number runs collapsed, the letters of the words must match the spoken input in order, wherever the spaces fall ("a v" and "AV", "q three" and "Q3", "m and a" and "M&A", "seventies" and "'70s"). A rewrite that fails keeps the original lowercase input. Rejected rewrites keep their text, so `restyle --recheck` can check them again after a change to `same_words` without calling the teacher.

Of 11,441 chosen rows, 10,716 (94%) passed and 725 keep their input. The first run rejected 1,650; loosening `same_words` for acronyms, letters mixed with digits, decades and "&" let 925 of them through on a recheck of the cached replies. What still fails is a real change: a dropped repeat ("the uh the"), a spoken "dash", "comma" or "period" dropped or turned into the mark, or an added word.

### Several recognizers

The cleanup model runs after Parakeet, Whisper and similar-sized recognizers that come later, so what matters is a spread of real formats rather than one recognizer's. `build` gives each restyled row one of four styles, fixed by a hash of its id (`restyle.ASR_STYLES`), tags it `asr_<style>`, and applies it with `recognizer_quirks`:

| Style | Share | Commas kept | Other changes | Result on the rewrites (periods, commas per 100 words; "um," share) |
|---|---|---|---|---|
| `parakeet` | 30% | 40% | fillers mostly bare, "$" glued, segment joins | 5.3, 5.4; 15% |
| `whisper` | 30% | 45% | a comma after 30% of fillers, segment joins, 17% of segments lowercased and unpunctuated | 4.4, 5.3; 23% |
| `qwen3` | 20% | 55% | 65% of commas before a new clause ("and", "so", "I", …) become sentence ends, a comma after 60% of fillers, numbers back to the words spoken (`spoken_numbers`, 58% hyphenated), "Period." as its own sentence 37% of the time, every segment capitalized | 7.9, 6.3; 59% |
| `plain` | 20% | all | none: the teacher's rewrite as written, for recognizers that write clean text | 5.1, 13.9; 53% |

The segment joins break inputs over 110 words into 40–110-word segments, end each with a period at the style's rate, capitalize a segment's first word unless the style starts it in lower case, then join them as `join_segment_texts` does. `spoken_numbers` puts back the exact words from the original transcript ("$230" as "two hundred and thirty dollars") and leaves a row alone when its numbers do not line up one to one. Every style keeps `same_words` true. The teacher's rewrites write 36% of numbers as digits, fewer than Whisper's 63%. The reference outputs are unchanged, since the words are the same.

`build` filters and deduplicates the aawaaz rows on their original spoken input, which their references were checked against, and swaps the rewrites in afterwards. The protected-token filter would otherwise drop rows where the rewrite writes "401k" or "PhD" and the reference "401(k)" or "Ph.D.".

The stage 3 and injected recognizer-style inputs had commas on both sides of nearly every mid-sentence "um" and "uh" (6,012 before and 6,936 after, over 6,951 fillers). The `parakeet` share of them (tagged `asr_parakeet`) gets Parakeet's filler commas (`restyle.filler_commas`); the rest keep theirs.

The build's inputs come out 48% lowercase and unpunctuated (the other aawaaz half, most of handy, and the `raw` stage 3 and injected rows) and 52% punctuated: 24% other punctuated rows (disfl_qa, some handy, stage 3 and injected `recognizer`), and 10% `parakeet`, 7% `whisper`, 5% `qwen3` and 5% `plain`.

### Heard rows (v2)

`heard run` gives v1 train rows a second input: what a recognizer wrote when Kokoro spoke the row's transcript, with the probe's voices, chunks and pauses and the app's segmenter and join (`asr_probe`). The row keeps its origin's reference, which stays right only where the cleanup could recover the spoken words from the heard ones. `heard.restore` makes sure it can:

- It aligns the heard words with the spoken ones, lowercased, fillers dropped and number runs compared by value: "$1,200" matches "twelve hundred dollars", "April 15, 2024" matches "april fifteenth twenty twenty four", "$2 million" matches "two million dollars", "£50" matches "fifty pounds". Number words read in groups join up only digit by digit or as a year ("nine six two oh one", "twenty twenty six"), so "eighteen twenty percent" does not match "1820%". A sign the comparison would drop ("±", "°") only matches the same sign.
- It puts the spoken words back, over whole tokens and with the fillers before them, wherever a difference touches any word other than an article, a discourse word ("so", "well", "yeah", "ok", "just") or a filler, or the recognizer made up a word: "uh the" heard as "other", "uh" as "ear", "U" or "R", "er" as "or", "internet" as "intranet", "ice in the cooler" as "ice and the cooler", a filler glued to a code ("A4721"), a misspelled name or drug, a filler heard as a capital "A" mid-sentence, which reads as the letter ("Garden A type 3"). Restored words take the reference's spelling and the recognizer's trailing punctuation, and a name heard with other capitals takes the reference's ("Grainlab" as "GrainLab"). A capitalized reference word counts as a name only if the reference never writes it in lower case, so "The" or "And" opening a quote is not one.
- What stays as heard is the recognizer's formatting: casing, punctuation, sentence breaks at segment cuts, digits or words for numbers, the glued "$", "a" or "UM" for fillers, article slips and compounds ("front end").
- A row is dropped when a restored stretch is over 6 words, the restored words are over 15% of the row and over 6 (the recognizer or Kokoro lost a phrase), the result fails the judge's code check (`gates.code_check`: a protected token the reference lacks, such as "2nd" from "on second thought"), or the result is the spoken text again. A dropped row is spoken again with a new voice, three tries in all. Transcripts with a backslash or control character are not sampled.

The rules were set on a 300-row pilot (196 Parakeet, 104 Whisper; 4.2 hours of audio) and `heard audit`. With the first rules (numbers, names, rare words, negations and cues only), the judge failed 17 of 100 kept rows that it passed with the origin's input. Fillers heard as words, content-word mishearings, number pairs, "pound fifty" and name capitals made most of them. With the rules above it fails 3 of 150 such rows, and 3 of 150 with the origin's input, and all 3 are a recognizer sentence break or capital changing how a hedge or "So" reads. So the rest is generated without the judge. (One of the 3, a row with "And" and "She" capitalized throughout, came from a name-casing bug fixed since; the fix changed the casing in 65 of the 277 kept rows.) The pilot keeps 277 of 300 rows (Parakeet 180/196, Whisper 97/104; 169/184 up to 100 words, 49/52 to 200, 59/64 above).

What this costs is the word errors: the heard text differs from the spoken words in 4.1% of words, the kept inputs in 0.3%. The rows teach the recognizers' real formatting, and almost none of their mishearings, which the reference could not follow. `heard restore` applies changed rules to the saved heard text without speaking anything again.

A heard row has source `heard` and keeps its origin's category, license, seed group (so it stays in train) and provenance fields. It is tagged `heard`, `style_recognizer`, `recognizer:<name>`, `from:<origin source>` and `heard_from:<origin id>`, plus the origin's tags except the ones about its input (`style_*`, `asr_*`, `densified`, `composed`, injection tags). `build --v2` swaps each kept row in for its origin's train row after the filters, so no reference is trained on twice and the row count stays the same; a heard row that fails a filter leaves its origin in place. Heard rows skip the duplicate check, since each copies its origin's words, and `spoken_variant`. The heard rows are realistic inputs for references the model already sees, not new data: fresh programmatic noise per epoch, the same kind of second view, kept val loss flat but scored no better.

`heard check` asks Mercury Decide (`inception/mercury-decide:free` on OpenRouter's System One endpoint, `prompts/judge/meaning.json`) whether each kept row's heard input says something its spoken text does not, and `build --v2` swaps in only the rows it passed. Verdicts go to `data/heard/meaning.jsonl` with the input they saw, so a row `heard restore` changes is asked again. On the pilot it takes 17 seconds and flags 5 of 277 kept rows (1.8%). By hand about half are real, all from punctuation or a filler heard as "a" rather than a changed word: "…for the fluid overload and a potassium. She doesn't need any supplementation" (spoken: "and uh potassium she doesn't need any supplementation"). The rest are sentence breaks a reader would see through ("how emotional. The finish line is like I crossed it"). On rows restored by the first rules it flagged 9 of the 13 that the judge failed for a changed word ("internet" as "intranet", "hole" as "hall", a dropped "bay" or "other") and missed dropped hedges ("I guess", "sort of"); it flagged 10 of the 83 rows the judge passed. It flags 49% of rows given the raw recognizer text. Its answers vary a little: asked twice, 2 of 217 rows changed verdict.

The full run heard 10,153 rows (133.5 hours of audio; the 10,000 sampled plus 153 pilot rows sampled before the odd-text rule). The restore kept 9,565 (Parakeet 6,451 of 6,818, Whisper 3,114 of 3,335; 94% up to 100 words, 97% to 200, 92% above). The meaning check flagged 209 of them (2.2%), and `build --v2` swaps in 9,296; the other 60 fail a build filter, nearly all a changed protected token, and leave their origin in place.

`heard audit` judges kept rows twice against the same reference, once with the heard input and once with the origin's. A row that passes with the origin's input and fails with the heard one is a miss of the rules; the origin side gives the judge's own failure rate on these references.

## Build: filters, split and stats

`build` drops:

- outputs with markdown: bold pairs, backticks, `#` headings, links or tables (content asterisks such as `x**2` stay);
- malformed lists: bullets other than `- `, empty items, numbering that skips;
- answers: outputs that open like a reply ("Sure", "Here is") where the speaker, past their fillers and openers, did not;
- changed protected tokens: a path, URL, email, identifier or flag that reappears in another form ("x25" as "X.25").
  - A dropped token is not flagged, because a self-correction may remove it.
  - A word-like token may change case ("co2" as "CO2"); paths, files, emails, URLs and flags may not ("readme.md" as "README.md");
- invented subject lines: a `Subject:` line when the speaker never said "subject" (aawaaz writes one for some emails);
- added legal formality: "shall", "hereby", "herein", "hereinafter", "thereof", "pursuant" or "whereas" that the speaker never said ("is going to pay" as "shall pay");
- invented signatures: a name or title under the closing ("Thanks,") that the speaker never said after the same closing (aawaaz fills one in from the body);
- implausible length ratios: an output over 1.3× the input plus 8 words, or a prose output under 30% of a 20-word input.
  - Lists, and inputs with a self-correction ("no wait", "scratch that"), may shrink further;
- exact and near-duplicate inputs: MinHash over word 3-grams without fillers, Jaccard of 0.85 or more. Filler-only `EMPTY` inputs are only checked for exact duplicates;
- outputs repeated more than 3 times, which trims handy's templates: "Forget the password reset I already did it." appeared 57 times.

Test rows and earlier sources win a duplicate.

**Splits** go by `seed_group`, so an injected row always shares its origin's split:

- A group with a preset test row (aawaaz's 3% held out by category, handy's and disfl_qa's own test splits, the `generate --test` rows and the second-seed `EMPTY` rows) is test.
- Other groups hash 5% into val.
- Groups made only of generated rows also hash 5% into test.

**Stats** are written to `data/build/stats.{json,md}`: counts per split, source, category, layout, license and tag, the dropped reasons, and the input and output length distributions.

### The current build

52,520 rows: train 46,745, val 2,457, test 3,318. Of the train rows, 25.4% are over 150 words, 11.0% over 300 and 4.4% over 600 (`stats.md`, "Input length by split").

| Source | Rows | License |
|---|---|---|
| aawaaz | 22,594 (43%) | MIT |
| teacher (stage 3 and composed) | 12,694 | MIT (ours) |
| handy | 10,303 | CC-BY-4.0 |
| injected | 3,717 | MIT (its origin's) |
| disfl_qa | 3,212 | CC-BY-4.0 |

- **Test groups:** todo 109's 2,828 test rows (`configs/test-ids-109.txt`) keep their ids and inputs. The 490 new test rows are composed rows of 150–1,000 words and their injected copies.
- **Layouts:** plain 32,743, mixed 10,824, paragraphs 5,181, list 3,772. That is 38% with layout, and 28% with a list.
- **Every row** has a source and a license. None is non-commercial or LDC.
- **Lengths:** the median input is 72 words and the median output 57; the 95th percentiles are 511 and 445; the longest input is 1,130.

| Dropped | Rows |
|---|---|
| output repeated too often | 1,218 |
| near-duplicate input | 219 |
| invented subject line | 49 |
| changed protected token | 30 |
| exact duplicate input | 28 |
| added legal formality | 19 |
| invented signature | 10 |
| implausible length ratio | 8 |
| answered | 7 |
| failed the completeness audit | 6 |
| markdown | 2 |
| malformed list: numbering skips | 1 |

### Hand review of the build

Two rounds of 50 random built rows per layout (200 rows a round), read against the acceptance list: answered transcripts, changed protected tokens, dropped or invented facts, wrongly added or missed lists, markdown, raised register. `datagen sample` draws such a set.

**First round (seed 108), before the fixes below:**

- **plain:**
  - a legal row turned "is going to pay" into "shall pay";
  - a raw-style injected row lost the "./" of "./data:/app/data".
- **paragraphs:**
  - two emails had a `Subject:` line nobody dictated;
  - one sign-off name was inferred from the body;
  - a dictated medical note became a chart summary ("HbA1c 9.2 (elevated from prior 7.8). Plan: …");
  - one email was reworded more formally;
  - two meeting titles gained words ("Annual Review Calibration Notes").
- **mixed:**
  - two meeting notes were condensed into label fragments;
  - one teacher row kept a correction remark ("We have candles, skip that one.").
- **list:** none.

**Fixes:**

- The raw style keeps leading punctuation (`inject._raw_token`).
- Build filters for invented subject lines and added legal formality.
- The register judge was extended to email, medical and meeting notes, and `fix-aawaaz` relabeled the 123 rows it flagged in those categories.
- After the second round, a filter for invented signatures.

**Second round (seed 109), on the fixed build (before the signature filter):**

- **plain:** none of the listed failures. One borderline creative row turns "he was like…" speech into reported speech ("suggested I try…").
- **list:** none.
- **paragraphs:**
  - One email that ended in a bare "thanks" got a three-line signature (name, title, college) taken from the body. The new invented-signature filter drops it and 7 like it.
  - A creative scene lost its "the scene where" framing.
- **mixed:**
  - **A dropped fact in a teacher row:** "before the shift starts, restock the gauze, the tape, and the saline bags" became "before the shift starts:" over the three items, losing "restock".
  - An email to Rosa that ended "thanks Rosa" was signed "Rosa".
  - A recipe restructured into steps added "until smooth".
  - aawaaz meeting notes still condense some of the speaker's sentences into labels ("Options: scale up (extra $1,200/month) or optimize queries…"), dropping who said what. The register judge passes these as the meeting-note layout.
- **Not found in the 200:** answered transcripts, changed protected tokens, markdown, wrongly added or missed lists, raised register.

So the build is not free of the listed failures. What is left is rare and small: one dropped verb, one misattributed sign-off, and one added phrase in 200 rows, plus the meeting-note condensation that is aawaaz's house style. Teacher rows also drop emphasis and hedge words at times ("honestly" and "maybe" are among the words most often missing from teacher list outputs). A per-row content check against the input would be the next filter to try.

## Real-dictation sanity check (step 9)

This export was a look at what real recognizer output looks like. It is not an evaluation set: todo 109 evaluates on the build's test split only.

`export-real` reads the history database read-only, from either schema. It writes each distinct transcript to `data/real/real.jsonl` with:

- a teacher `draft`;
- an empty `reference` to write by hand.

A rerun keeps written references and adds new transcripts up to `--limit`. `--no-draft` skips sending transcripts to the LLM.

This is personal data. `build` and `publish` never read `data/real/`; never train on it.

**First export (12 rows).** The 20 transcripts hold 12 distinct texts, 7 of them mic tests. References are filled in by the agent that built this dataset: 11 match the teacher draft, and one keeps a grammar fix the draft made.

## The cleanup model (todo 109)

A small model fine-tuned on `data/build/` to do Clean Up & Format alone. It takes one user message holding only the transcript, with no system message and no instructions, and replies with the edited text or `EMPTY`.

### The prompt contract

- `serving/chat_template.jinja` is the one chat template. Training renders every row with it, and `llama-server --chat-template-file` serves with it, so the two cannot drift. It refuses a system turn.
- `serving/prompt_golden.json` holds the messages, the rendered prompt, the output budget and the draft (`draft_text`, below) for a few transcripts. `tests/test_model_prompt.py` and the Rust test `local_cleanup_prompt_matches_the_training_golden_fixture` (whispernow-core) both check against it. Change the fixture, the template, `model/prompt.py` and `build_local_cleanup_prompt` together.
- Each request also has a draft: the transcript pre-cleaned by code (`model/draft.py`; in the app, `llm/draft.rs` in whispernow-core). The app's in-process engine drafts from it for speculative decoding (`llm/local.rs`); training-side requests send it as `draft_text` to the patched llama-server, and the release build ignores it.
- Decoding is greedy, with `repeat_penalty` 1.05 over `repeat_last_n` 4096 tokens (the whole prompt and reply, as transformers' `repetition_penalty` works; llama.cpp's default window of 64 keeps fewer protected tokens), and `max_tokens` = max(64, ⌈2.5 × input words⌉). No build row needs more. A reply that hits the limit is treated as a loop: the app keeps the deterministic text. The fixture's `decoding` pins these fields for both sides.

### Setup

- `uv sync --extra train` adds torch (CUDA 13.0 wheels), transformers, TRL, PEFT and accelerate. `uv sync` with one extra removes the others; use `uv sync --all-extras` to keep `audio` too. Every command below runs as `uv run --extra train cleanup …`.
- Export and serving use llama.cpp release **b11259**, unpacked under `.cache/llama.cpp/` (gitignored):
  - `src/`: the release source (`https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11259.tar.gz`), for `convert_hf_to_gguf.py` and `gguf-py`;
  - `cuda/llama-b11259/`: `llama-b11259-bin-ubuntu-cuda-13.4-x64.tar.gz`, which loads the CUDA 13 runtime from `/usr/local/cuda`;
  - `cpu/llama-b11259/`: `llama-b11259-bin-ubuntu-x64.tar.gz`.
  - `draft-text/`: `cleanup build-server [--cpu]` copies `src/`, applies `serving/llama-server-draft-text.patch` and builds `llama-server` into `build-cuda/` or `build-cpu/`. It takes about 10 minutes, bounded like a Cargo build (four jobs, four CPUs, low I/O priority), and needs CMake (or `uvx` to fetch it) and, for the CUDA build, the CUDA toolkit (`/usr/local/cuda`). Ninja is used when installed.
- Runs land in `runs/<name>/` (gitignored): `config.json`, `train.json` (steps, time, peak GPU memory, loss history), `model/` (the merged bf16 checkpoint), `eval*/` and `gguf/`.
- Before training, check `nvidia-smi` and `df -h .`. Run under `nice -n 10 ionice -c2 -n7`, and never while Cargo builds (AGENTS.md).

### Commands

| Step | Command | Writes |
|---|---|---|
| Train one config | `cleanup train configs/lfm25-350m.toml --name NAME [--set learning_rate=1e-4 --set lora.r=16 …]` | `runs/NAME/` |
| Train and score every run of a sweep that is not done yet | `cleanup sweep configs/sweep-lfm25-350m.toml [--only NAME]` | `runs/<run>/`, `runs/<run>/eval/val500.*` |
| A sweep's val scores as a table | `cleanup sweep-table configs/sweep-lfm25-350m.toml` | stdout |
| Evaluate a checkpoint (HF, batched greedy) | `cleanup eval NAME --split test --judge` | `runs/NAME/eval/test.{outputs,judge}.jsonl`, `test.scores.json` |
| Evaluate the teacher (`LLM_MODEL` with the app's hosted prompt) | `cleanup eval --teacher --split test --judge` | `runs/teacher/eval/` |
| Export to GGUF | `cleanup export NAME [--quant Q8_0 Q4_K_M]` | `runs/NAME/gguf/model-{BF16,Q8_0,Q4_K_M}.gguf` |
| Build llama-server with the transcript drafter | `cleanup build-server [--cpu]` | `.cache/llama.cpp/draft-text/build-{cuda,cpu}/` |
| Serve a GGUF with the committed template, drafting from each request's `draft_text` (`--no-draft`: the release build, plain decoding) | `cleanup serve NAME [--quant Q8_0] [--cpu] [--parallel 4] [--port 8080] [--no-draft]` | llama-server on `127.0.0.1:8080` |
| Evaluate what llama-server returns | `cleanup eval NAME --server http://127.0.0.1:8080/v1 --label q4 --split test --judge` | `runs/NAME/eval-q4/` |
| Latency for a short dictation and test rows of about 100, 200, 300 (`long`), 500 and 1,000 words | `cleanup bench http://127.0.0.1:8080/v1 --save runs/NAME/bench-gpu-q4.json` | the JSON |
| The teacher's latency on the same dictations | `cleanup bench --teacher --save runs/teacher/bench.json` | the JSON |
| Read a server's (or the teacher's) cleanup of the dictations in `data/real/` by hand; nothing is written or scored | `cleanup handcheck http://127.0.0.1:8080/v1` / `cleanup handcheck --teacher` | stdout |
| Fresh noise per epoch: one re-noised copy of the train split per epoch after the first (`noise_per_epoch` runs make it themselves) | `uv run datagen epochs [--epochs 5]` | `data/build/epochs/train.epoch-<n>.jsonl`, `stats.json` |
| Evaluate an epoch checkpoint | `cleanup eval NAME --checkpoint epoch-1 --split val --limit 500` | `runs/NAME/eval-epoch-1/` |
| Score the split's own references, as a baseline for the judge | `cleanup eval --reference --split val --limit 500 --judge` | `runs/reference/eval/` |
| Decode with another repeat penalty, or penalty window (server only) | `cleanup eval NAME … --repeat-penalty 1.0` / `--repeat-last-n 64` | `runs/NAME/eval…-rp1/`, `…-last64/` |
| Score a benchmark (`bench-long` or `bench-noise`, from `data/bench/`) | `cleanup eval NAME --server http://127.0.0.1:8080/v1 --label gpu-q8 --split bench-long --judge` | `runs/NAME/eval-gpu-q8/bench-long.*` |
| Score a hosted model on 800 test rows instead of 3,297 (`configs/test-sample-800.txt`, per category and length bucket in proportion to test). Through OpenRouter: `LLM_BACKEND=openai LLM_BASE_URL=https://openrouter.ai/api/v1 LLM_API_KEY=… LLM_MODEL=google/gemma-4-26b-a4b-it LLM_REASONING=` (empty: the provider's default; `none` turns reasoning off), with the judge settings above | `cleanup eval --teacher --label gemma4-26b-a4b --split test-sample --judge` | `runs/teacher/eval-gemma4-26b-a4b/test-sample.*` |
| Score a run judged on all of test on the same 800 rows, from its saved outputs and verdicts | `cleanup rescore NAME/eval-q4 --split test-sample` | `runs/NAME/eval-q4/test-sample.r112.faults.metrics2.scores.json` |
| Compare evaluations side by side | `cleanup compare teacher=teacher/eval q4=NAME/eval-q4 [--stem test] [--group test_group --group length …]` | stdout |
| Score saved outputs and judge verdicts again with the current checks, generating and judging nothing (`compare` then reads the new file) | `cleanup rescore NAME/eval-q4 [teacher/eval …] --split test` | `runs/NAME/eval-q4/<stem>.metrics2.scores.json` |
| Judge chosen rows' saved outputs again with the current judge (after a reference or judge prompt change), then rescore. The verdicts are appended to the split's verdict file and replace those rows' earlier ones; other rows keep theirs | `LLM_JUDGE_BACKEND=codex LLM_JUDGE_MODEL=gpt-6.1-sol LLM_JUDGE_REASONING=high cleanup rejudge NAME/eval-q4 [teacher/eval …] --split bench-long --ids IDS.txt` | `runs/NAME/eval-q4/<stem>.faults.judge.jsonl`, `<stem>.metrics2.scores.json` |

`--limit N` scores a fixed random sample (seed 109) of a split; the sweep scores 500 val rows, for the final model and for each `epoch-N` checkpoint that training saves before the last epoch. Evaluation reuses the outputs it already wrote, so adding `--judge` later only runs the judge.

### Configs

`configs/lfm25-350m.toml` holds every setting, and `--set` overrides any of them as TOML (`--set base_model='LiquidAI/LFM2.5-350M-Base'`, `--set method='lora'`). `build` names the directory whose train and val splits a run reads (`data/build`, or `--set build=data/build-v2`). On build-v2 the 350M needs `--set chunked_loss=true` on a 12 GB GPU: its first, longest batch (2,803 tokens) runs out of memory upcasting the whole vocabulary's logits to fp32 (5.5 GiB); with it the batch peaks at 9.1 GB. A sweep file names that config and lists runs, each with its overrides. Training is SFT with TRL: a conversational prompt-completion dataset rendered by the committed template, loss on the completion only, length-grouped batches of 8 with 4 accumulation steps (32 rows per optimizer step), rows up to `max_length` 2,816 tokens, cosine schedule with 3% warmup. The batch fits the longest batches of todo 110's build, which come first (`cleanup train` turns on PyTorch's expandable segments for them). A full fine-tune keeps fp32 master weights under bf16 autocast; LoRA (all linear layers) trains on bf16 weights and is merged before saving, so every run exports the same way.

`configs/minicpm5-1b.toml` fits MiniCPM5-1B's full fine-tune in 12 GB of GPU memory: bitsandbytes' 8-bit AdamW (`optim`), frozen bf16 input embeddings (`freeze_embeddings`), Liger's fused cross-entropy (`liger`), and batches of 2 with 16 accumulation steps (4 × 8 runs out of memory on the 2,800-token rows). Its sweeps are `configs/sweep-minicpm5-1b.toml` and `configs/sweep-minicpm5-1b-stage2.toml`.

Todo 111's runs on todo 110's build are in `configs/sweep-lfm25-350m-long.toml` and `configs/sweep-minicpm5-1b-long.toml`. `--set noise_per_epoch=true` trains on a freshly noised copy of the train split in each epoch (`datagen/epochs.py`): injected rows get new noise in their own target, and composed and short-source rows get their recognizer style and densified share drawn again. The other rows are the same in every epoch, and val, test and the benchmarks are never re-noised. Such a run makes one pass over the epoch files, and saves its `epoch-N` checkpoints at the same steps as a plain run.

The preference stage and fact-weighted SFT were tried and removed.

### What evaluation reports

`<split>.scores.json` holds each rate overall, for the command categories together, and per category, input style (punctuated or not), source, tag, input length bucket and noise pattern (`pattern:<name>` tags). For the test split, `test_group` separates todo 109's test rows (`configs/test-ids-109.txt`) from rows added since. The judge takes up to 15 rows per request, fewer when long rows would pass 16,000 characters. `cleanup serve` gives each request slot 8,192 tokens of context, enough for a 1,000-word dictation and its 2,500-token reply limit. Every rate is a percentage of the rows it applies to:

- **Mechanical checks** (`model/metrics.py`): chrF and exact match against the reference; layout matches; formatting added where the reference has none, and missed where it has some; bad list markers and markdown; protected tokens (URLs, emails, paths, flags, identifiers, numbers) kept; fillers left beyond the reference's count; passthrough rows kept unchanged; `EMPTY` answered right and given wrongly; loops (a repeated 5-gram, or hitting the output limit); assistant openers; more than 30% of the words deleted.
- **Contract checks** (todo 112): em dashes, spelled-out counts next to a word (an upper bound: names and idioms match too), correction cues kept from the transcript, and `kept_words`, the share of the transcript's "maybe", "probably", "I think", "I guess", "I believe" and "because" that the output keeps.
- **Judge** (`--judge`, `prompts/judge/eval.md`, `LLM_JUDGE_MODEL`): answered or executed the transcript instead of editing it, and every fault, one per fact, by kind: `dropped`, `changed` (another value, or attached to the wrong thing), `wrong_side` (an abandoned value kept as if it stands), `unresolved` (both sides of a correction kept with the speaker taking one back, so nothing is lost) and `invented`. `facts_dropped` is the share of rows with a dropped, changed or wrong-side fault, and `facts_invented` with an invented one; an unresolved correction fails neither. `faults` gives each kind per judged row, per 1,000 reference words and as a share of rows, and `facts_lost` the three lossy kinds together, which keeps moving on long rows where nearly every row has a fault. Rows over 300 words get a request each.

**References after todo 112.** When every reference of a split is `gpt-6.1-sol`/high, the judge and score files are `<split>.r112.faults.judge.jsonl` and `<split>.r112.faults.scores.json`, so the old scores stay beside the new (`cleanup compare --stem test.r112.faults …`). `<split>.r112.judge.jsonl` and `.scores.json` are the pass/fail judge before fault counts; an unjudged run writes `<split>.r112.scores.json`, and the judge must be `gpt-6.1-sol` at high reasoning through codex. Model outputs are reused as they are; only rows with new ids (the re-made injected test rows) are generated. `cleanup eval --teacher --label 112` keeps the hosted teacher's outputs under the new prompt in `runs/teacher/eval-112/`.

The references for teacher-sourced rows are the teacher's own outputs, so the teacher scores high on reference metrics such as chrF by construction. Compare with it on the judge rates and the checks that do not need a reference.

### Using it in the app

The app runs the model itself, in process, on llama.cpp b11259's C API (todo 115); there is no server to start.

1. In WhisperNow, choose the provider **Local cleanup model**, pick its **Cleanup model**, and download it when asked (Settings → Models also lists both). The default is `r112-full-inst-1e-5-e3` (LFM2.5-350M) at Q8_0 (379 MB), retrained on todo 112's relabeled build; the larger choice is `r112-minicpm-full-inst-1e-5-e3` (MiniCPM5-1B) at Q8_0 (1.15 GB), which loses a fifth to a quarter fewer facts but is slower. Both are pinned in the app's model registry by URL, size and SHA-256, and both use the same prompt, decoding and pre-cleaner. The chosen model loads when the provider or model is chosen, on the GPU when one works (Vulkan on Linux and Windows, Metal on Apple silicon) and otherwise on the CPU, and drafts from the pre-cleaned transcript as the patched server does. Only one is loaded at a time.
2. Choose **Clean Up & Format**. Settings refuse any other prompt with this provider.
3. To use a `llama-server` of your own instead (`cleanup serve`), choose **Other** (OpenAI-compatible) with its URL. That path sends no `draft_text` and none of the decoding fields, so it is for trying other models, not for this one's parity.

The patched server (`cleanup build-server`, `cleanup serve`) remains for training-side evaluation: `cleanup eval` and `bench`. The parity harness that checked the app's engine against it was removed in todo 123; git history has it.

