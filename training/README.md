# WhisperNow training data

Builds the dataset for a local Clean Up & Format model (todo 108), then trains, evaluates, exports and serves that model (todo 109; see [The cleanup model](#the-cleanup-model-todo-109)).

Every row has one schema (`datagen/schema.py`): `id`, `input`, `output`, `source`, `category`, `tags`, `layout` (`plain`, `list`, `paragraphs` or `mixed`, computed from the output), `seed_group`, `split`, `license`, `teacher_model` and `teacher_reasoning` (the teacher's reasoning effort, set from the todo 112 relabel on).

## Setup

- `uv sync` installs everything. Run commands with `uv run`, and tests with `uv run pytest`.
- LLM settings come from the repo-root `.env`: the `LLM_*` variables that `LlmSettings.from_env` in `datagen/llm.py` reads. The key is never printed or logged.
- `publish` needs a Hugging Face login (`hf auth login`).
- `asr-probe` needs `uv sync --extra audio` (sherpa-onnx 1.13.6, the version the app vendors), the app's Parakeet TDT 0.6B v2 fp16 and Silero VAD models in `~/.local/share/WhisperNow/models/`, and Kokoro `kokoro-int8-multi-lang-v1_0` (Apache-2.0, from the k2-fsa/sherpa-onnx `tts-models` release) unpacked in `.cache/tts/`. `--model whisper` and `--model qwen3` need `sherpa-onnx-whisper-distil-large-v3.5` (the app's catalog build, same SHA-256) and `sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25` from the k2-fsa/sherpa-onnx `asr-models` release, unpacked in `.cache/asr/`.
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
| 7. Restyle half the aawaaz inputs (`--recheck` re-checks rejects, no calls) | `uv run datagen restyle` | `data/restyle/aawaaz.jsonl`, `rejected.jsonl` |
| 110.1 Completeness audit of every long reference and 500 short ones (`--fix` relabels the failures through the gates; `--report` only rewrites the report) | `uv run datagen audit` | `data/audit/{rows,verdicts}.jsonl`, `report.json` |
| 110.3 Filler formatting and correction cues: real dictations, the build, the injection, any files | `uv run datagen profile [--injection] [FILE …]` | stdout |
| 110.4 Long benchmark, generated rows (equal thirds of 150–300, 300–600 and 600–1,000 words) | `uv run datagen compose --run bench --total 360` | `data/bench/long/<category>.<bucket>.jsonl` |
| 110.4 Long benchmark, real AMI meeting turns (text only, CC BY 4.0) | `uv run datagen bench ami` | `data/bench/long/ami.jsonl` |
| 110.4 Noise benchmark, nine patterns and a control set | `uv run datagen bench noise` | `data/bench/noise/<pattern>.jsonl` |
| 110.4 Speak a third of the noise rows with Kokoro and transcribe them with Parakeet | `nice ionice -c3 uv run --extra audio datagen bench probe` | `data/bench/noise/probed.jsonl` |
| 110.4 Re-gate rows judged before the header rules (noise, then the long benchmark) | `uv run datagen rejudge noise`, `uv run datagen rejudge bench-long` | `data/bench/{noise,long}/rejudged.jsonl` |
| 110.4 New references for the probed rows, through the gates | `uv run datagen bench gate-probed` | `data/bench/noise/probed.gated.jsonl` |
| 110.4 Write both benchmark files, and check them against the build | `uv run datagen bench assemble` | `data/bench/{long,noise}.jsonl`, `report.json` |
| 110.4 Check every benchmark reference again, one row per request with the judge at high reasoning (a row the teacher fails twice gets the judge model's reference, through the same gates), then assemble again | `LLM_JUDGE_REASONING=high uv run datagen rejudge bench-long-strict`, `LLM_JUDGE_REASONING=high uv run datagen rejudge noise-strict`, `uv run datagen bench assemble` | `data/bench/{long,noise}/strict.jsonl` |
| 110.5 Long composed training rows (40/35/25% by bucket). `gpt-6.1-sol` through codex writes the transcripts (low) and references (high; xhigh over 600 words). The judge checks one row per request (low up to 600 words, high above), and a failure is relabelled once by the teacher. Each cell runs `compose.LANES` batches at once (2, 6 and 12 by bucket), so the slow long cells keep the codex slots busy. The first 2,558 rows had `glm-5.3` as generator, teacher and judge | `LLM_BACKEND=codex LLM_MODEL=gpt-6.1-sol LLM_REASONING=high LLM_GENERATOR_MODEL=gpt-6.1-sol LLM_GENERATOR_REASONING=low LLM_JUDGE_MODEL=gpt-6.1-sol LLM_JUDGE_REASONING=high LLM_CODEX_CONCURRENCY=64 uv run datagen compose --total 8300` | `data/stage4/<category>.<bucket>.jsonl` |
| 110.5 Re-gate the training rows judged before the header rules | `uv run datagen rejudge train` | `data/stage4/rejudged.jsonl` |
| 110.5 Gate the rows made before the `glm-5.3` cascade again, as that cascade gated a new row | `LLM_MODEL=glm-5.3 LLM_JUDGE_REASONING=max uv run datagen rejudge train-strict` | `data/stage4/strict.jsonl` |
| 110.5 Gate the rows `glm-5.3` labelled the way `compose` now gates a new row (a failure gets one `gpt-6.1-sol` reference) | the same `LLM_*` settings, `uv run datagen rejudge train-codex` | `data/stage4/codex.jsonl` |
| 112. Rewrite every reference with `gpt-6.1-sol` at high reasoning (teacher and judge; the command refuses any other setting for any role). `pilot` relabels about 200 stratified rows and writes the hand-review page; `review` rewrites them from the records; `build`, `bench-long`, `bench-noise` or `all` is the full run; `--report` only compares old and new references | `LLM_BACKEND=codex LLM_MODEL=gpt-6.1-sol LLM_REASONING=high LLM_JUDGE_BACKEND=codex LLM_JUDGE_MODEL=gpt-6.1-sol LLM_JUDGE_REASONING=high LLM_GENERATOR_BACKEND=codex LLM_GENERATOR_MODEL=gpt-6.1-sol LLM_GENERATOR_REASONING=high LLM_CODEX_CONCURRENCY=64 uv run datagen relabel all` | `data/relabel/<scope>.jsonl`, `data/relabel/inputs/`, `data/relabel/pilot/`, `runs/relabel-review.html` |
| 4. Filter, deduplicate, split, report | `uv run datagen build` | `data/build/{train,val,test}.jsonl`, `stats.md` |
| Draw rows for a hand review | `uv run datagen sample list --n 50 --seed 110` | stdout |
| Publish (private) | `uv run datagen publish <user>/<repo>` | Hugging Face dataset |
| Real-dictation sanity check (not an evaluation set) | `uv run datagen export-real` | `data/real/real.jsonl` (never uploaded) |

**Every LLM step resumes.** The judge commands append verdicts as batches finish and skip rows already judged. `generate` logs each batch in `<category>.batches.jsonl` and continues from there. Each response is cached by its exact request in `.cache/llm/`, so a rerun repeats no finished call. `generate --scale 0.05` runs a pilot at 5% of each target. `generate --category list_edits` runs one category.

**The relabel (todo 112)** sends each transcript in the app's own request (`teacher.teacher_messages`), never showing the old reference. The gates check the new reference in code (protected tokens kept, no em dash, no correction cue kept from the transcript) and then with the completeness judge, which also gets code's flags: spelled-out numbers next to a word, and hedges or "because" the reference says less often than the transcript. Rows over 300 words get a judge request each; benchmark rows always do. A failure is written once more with the judge's note added to the request. `--retry N` runs retry round N over the rows still dropped: it checks their last reference again, since the gates may have changed, and gives each that still fails one more rewrite with the new note. Rows that fail every round stay dropped. The first run copies the rows it relabels to `data/relabel/inputs/` (the build's injected rows apart, for the stats only). Once a scope has records, `build` and `bench assemble` take every reference from them, drop the rows that failed, refuse to run while a scope is partial, and skip the old audit overrides; the injected rows are made again from the relabelled parents and inherit `teacher_model` and `teacher_reasoning`. `stats.md` then compares old and new references by source. Its hedge and reason keep rates leave out the occurrences a reference may drop: a hedge said again straight away, a closing aside ("I think that's everything") and one followed within 10 words by a correction cue; `report.json` keeps the raw rates as `kept_raw`.

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

**Backend.** `LLM_BACKEND=openai` calls any OpenAI-compatible endpoint directly with `LLM_BASE_URL` and `LLM_API_KEY`.

`LLM_BACKEND=codex` runs one `codex exec` per request.

- Each request runs from an empty temporary directory, with a read-only sandbox, no web search and no session saved.
- The user's codex config, rules and any `AGENTS.md` are ignored.
- Our system message replaces codex's own instructions (`model_instructions_file`). The user message goes on stdin, and only the last message is kept.
- **Caveat:** codex still declares its tools and adds its environment note, about 4,000 tokens per request.
- Each call runs in its own process group (`llm.run_process_group`). `codex` is a node wrapper around a native binary, and a binary left behind holds the pipes open, which once hung a run for 70 minutes. On a timeout or cancellation, or 10 s after the wrapper exits with its pipes still open, the whole group is killed.
- The reasoning setting is codex's `model_reasoning_effort`: low, medium, high, xhigh or max.

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

- **Z.ai (teacher, judge and generator outputs).**
  - Z.ai's terms of use forbid using outputs "to develop, train, or enhance algorithms or models that compete with us", and forbid automated access by bots or spiders.
  - Whether a local dictation-cleanup model competes with Z.ai is a judgment call for the project owner. Rows whose output the teacher wrote carry `teacher_model`; stage 3 rows carry the license `zai-output`.
- **OpenAI (`gpt-6.1-sol` through the Codex CLI, todo 110's composed rows).**
  - OpenAI's terms of use also forbid using output to develop models that compete with OpenAI. Whether this model competes is the same judgment call as for Z.ai.
  - Each composed row's license names the models that wrote its transcript and its reference (`compose.output_license`): `openai-output` 6,961 rows, `openai-output+zai-output` 898 (a `glm-5.3` transcript with a `gpt-6.1-sol` reference) and `zai-output` 194.
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

- **`judge-aawaaz`**, all 22,772 rows, on four criteria (`prompts/judge/stage1.md`): content preserved (including hedges such as "maybe"), no hallucination, lead-in kept, and not answered.
- **`judge-register`**, the 11,866 rows of `casual_conversation`, `creative_writing`, `legal_contract`, `email_professional`, `medical_clinical` and `meeting_notes`, on register kept (`prompts/judge/register.md`): no shortening into notes or a chart summary, and no added formality. The first three categories came out of the 100-row hand review below; the last three were added after the first 50-per-layout review of the build (see [Build review](#hand-review-of-the-build)). The judge flagged casual 919, creative 6, legal 39, email 12, medical 76 and meeting notes 35.

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

The teacher is `LLM_MODEL` sent exactly what the app sends for Clean Up & Format. That is `ENHANCEMENT_SYSTEM_CONTRACT` as the system message, and the `fix_grammar` edit request wrapped as `build_enhancement_prompt` does in Cleanup mode. `prompts/teacher/` holds copies, and `tests/test_teacher.py` fails if they drift from the Rust source.

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

`add` loads handy's alpaca config and disfl_qa, drops what conflicts with the target rules, samples by category with a fixed seed, and has `glm-5.3` audit every sampled row (`prompts/judge/stage2.md`). Rows whose only failure is a list kept as prose are relabeled by the teacher and re-judged; other failures are dropped. sotto is off by default (see Terms).

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

`asr-probe` speaks 300 sampled inputs (55% aawaaz, half of those from the longest quarter; 20% teacher, 15% handy, 10% disfl_qa) with a random English Kokoro voice at speed 0.95–1.15, in chunks with 0.15–1.4 s pauses: 2.3 hours of audio. It then runs the app's path in Python, with the same Silero VAD settings and segmenter constants and the same segment join, through three recognizers on the same audio: Parakeet TDT 0.6B v2 fp16 (greedy search), Whisper distil-large-v3.5 int8 with the app's settings (automatic language, no tail padding), and Qwen3-ASR 0.6B int8, which the app does not offer yet. `data/probe/report.<model>.json` has the numbers:

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
| teacher (stage 3 and composed) | 12,694 | openai-output 7,378, zai-output 4,419, both 897 |
| handy | 10,303 | CC-BY-4.0 |
| injected | 3,717 | MIT (2,848), or its composed origin's (869) |
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
- `serving/prompt_golden.json` holds the messages, the rendered prompt and the output budget for a few transcripts. `tests/test_model_prompt.py` and the Rust test `local_cleanup_prompt_matches_the_training_golden_fixture` (whispernow-core) both check against it. Change the fixture, the template, `model/prompt.py` and `build_local_cleanup_prompt` together.
- Decoding is greedy, with `repeat_penalty` 1.05 over `repeat_last_n` 4096 tokens (the whole prompt and reply, as transformers' `repetition_penalty` works; llama.cpp's default window of 64 keeps fewer protected tokens), and `max_tokens` = max(64, ⌈2.5 × input words⌉). No build row needs more. A reply that hits the limit is treated as a loop: the app keeps the deterministic text. The fixture's `decoding` pins these fields for both sides.

### Setup

- `uv sync --extra train` adds torch (CUDA 13.0 wheels), transformers, TRL, PEFT and accelerate. `uv sync` with one extra removes the others; use `uv sync --all-extras` to keep `audio` too. Every command below runs as `uv run --extra train cleanup …`.
- Export and serving use llama.cpp release **b11259**, unpacked under `.cache/llama.cpp/` (gitignored):
  - `src/`: the release source (`https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11259.tar.gz`), for `convert_hf_to_gguf.py` and `gguf-py`;
  - `cuda/llama-b11259/`: `llama-b11259-bin-ubuntu-cuda-13.4-x64.tar.gz`, which loads the CUDA 13 runtime from `/usr/local/cuda`;
  - `cpu/llama-b11259/`: `llama-b11259-bin-ubuntu-x64.tar.gz`.
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
| Serve a GGUF with the committed template | `cleanup serve NAME [--quant Q8_0] [--cpu] [--parallel 4] [--port 8080]` | llama-server on `127.0.0.1:8080` |
| Evaluate what llama-server returns | `cleanup eval NAME --server http://127.0.0.1:8080/v1 --label q4 --split test --judge` | `runs/NAME/eval-q4/` |
| Latency for a short dictation and test rows of about 100, 200, 300 (`long`), 500 and 1,000 words | `cleanup bench http://127.0.0.1:8080/v1 --save runs/NAME/bench-gpu-q4.json` | the JSON |
| The teacher's latency on the same dictations | `cleanup bench --teacher --save runs/teacher/bench.json` | the JSON |
| Read a server's (or the teacher's) cleanup of the dictations in `data/real/` by hand; nothing is written or scored | `cleanup handcheck http://127.0.0.1:8080/v1` / `cleanup handcheck --teacher` | stdout |
| Fresh noise per epoch: one re-noised copy of the train split per epoch after the first (`noise_per_epoch` runs make it themselves) | `uv run datagen epochs [--epochs 5]` | `data/build/epochs/train.epoch-<n>.jsonl`, `stats.json` |
| Preference stage: sample an SFT model behind llama-server, judge, restore dropped facts for minimal pairs, write pairs (see [Preference stage](#preference-stage-todo-111)) | `LLM_BACKEND=codex LLM_MODEL=gpt-6.1-sol LLM_JUDGE_BACKEND=codex LLM_JUDGE_MODEL=gpt-6.1-sol LLM_CODEX_CONCURRENCY=64 cleanup prefer NAME --server http://127.0.0.1:8080/v1` | `runs/NAME/prefer/` |
| Evaluate an epoch checkpoint | `cleanup eval NAME --checkpoint epoch-1 --split val --limit 500` | `runs/NAME/eval-epoch-1/` |
| Score the split's own references, as a baseline for the judge | `cleanup eval --reference --split val --limit 500 --judge` | `runs/reference/eval/` |
| Decode with another repeat penalty, or penalty window (server only) | `cleanup eval NAME … --repeat-penalty 1.0` / `--repeat-last-n 64` | `runs/NAME/eval…-rp1/`, `…-last64/` |
| Score a benchmark (`bench-long` or `bench-noise`, from `data/bench/`) | `cleanup eval NAME --server http://127.0.0.1:8080/v1 --label gpu-q8 --split bench-long --judge` | `runs/NAME/eval-gpu-q8/bench-long.*` |
| Compare evaluations side by side | `cleanup compare teacher=teacher/eval q4=NAME/eval-q4 [--stem test] [--group test_group --group length …]` | stdout |

`--limit N` scores a fixed random sample (seed 109) of a split; the sweep scores 500 val rows, for the final model and for each `epoch-N` checkpoint that training saves before the last epoch. Evaluation reuses the outputs it already wrote, so adding `--judge` later only runs the judge.

### Configs

`configs/lfm25-350m.toml` holds every setting, and `--set` overrides any of them as TOML (`--set base_model='LiquidAI/LFM2.5-350M-Base'`, `--set method='lora'`). A sweep file names that config and lists runs, each with its overrides. Training is SFT with TRL: a conversational prompt-completion dataset rendered by the committed template, loss on the completion only, length-grouped batches of 8 with 4 accumulation steps (32 rows per optimizer step), rows up to `max_length` 2,816 tokens, cosine schedule with 3% warmup. The batch fits the longest batches of todo 110's build, which come first (`cleanup train` turns on PyTorch's expandable segments for them). A full fine-tune keeps fp32 master weights under bf16 autocast; LoRA (all linear layers) trains on bf16 weights and is merged before saving, so every run exports the same way.

`configs/minicpm5-1b.toml` fits MiniCPM5-1B's full fine-tune in 12 GB of GPU memory: bitsandbytes' 8-bit AdamW (`optim`), frozen bf16 input embeddings (`freeze_embeddings`), Liger's fused cross-entropy (`liger`), and batches of 2 with 16 accumulation steps (4 × 8 runs out of memory on the 2,800-token rows). Its sweeps are `configs/sweep-minicpm5-1b.toml` and `configs/sweep-minicpm5-1b-stage2.toml`.

Todo 111's runs on todo 110's build are in `configs/sweep-lfm25-350m-long.toml` and `configs/sweep-minicpm5-1b-long.toml`. `--set noise_per_epoch=true` trains on a freshly noised copy of the train split in each epoch (`datagen/epochs.py`): injected rows get new noise in their own target, and composed and short-source rows get their recognizer style and densified share drawn again. The other rows are the same in every epoch, and val, test and the benchmarks are never re-noised. Such a run makes one pass over the epoch files, and saves its `epoch-N` checkpoints at the same steps as a plain run.

### Preference stage (todo 111)

`cleanup prefer NAME` runs on an SFT run and writes each step to `runs/NAME/prefer/`, skipping a step whose file exists (`model/prefer.py`):

1. **Inputs:** 400 train rows from each of the 150–300, 300–600 and 600–1,000-word buckets, and 300 more with a reasoned or re-decided correction. Never a benchmark row or its seed group.
2. **Samples:** the greedy output and 4 samples at temperature 0.8 per input, from llama-server with the app's request and penalty.
3. **Verdicts:** each distinct output goes through the code checks (loop, fillers the reference lacks, Markdown, list markers, added layout, assistant opener, wrong `EMPTY`), then todo 110's completeness judge (`prompts/judge/completeness.md`). The judge takes one row per request, at low reasoning up to 600 words and high above, as in the long-data gate.
4. **Minimal pairs:** a failing output whose only fault is dropped facts goes to the teacher with the judge's note (`prompts/prefer/restore.md`). The repair is kept only if the code finds nothing but inserted words (`only_insertions`) and the judge passes it.
5. **Pairs:** up to 2 per input. The minimal pair comes first, then failing outputs against the passing output nearest in length. An input where nothing passes pairs a failure with its gated reference. `kind` names the rejected output's fault (dropped, invented, correction, filler, format, loop), and 5% of inputs by id hash are a val split. `rft.jsonl` holds one passing output per train input, for the rejection-sampling baseline.

DPO trains from the SFT model: `cleanup train configs/<model>.toml --name NAME-dpo --set objective='dpo' --set train_file='runs/NAME/prefer/pairs.jsonl' --set base_model='runs/NAME/model' --set epochs=1 --set learning_rate=1e-6`. It uses TRL's sigmoid loss plus an SFT term on the chosen reply (`beta` 0.1, `sft_weight` 1.0) and reference log-probs computed once before training. The baseline is plain SFT on `rft.jsonl` from the same model (`--set train_file='runs/NAME/prefer/rft.jsonl'`).

### What evaluation reports

`<split>.scores.json` holds each rate overall, for the command categories together, and per category, input style (punctuated or not), source, tag, input length bucket and noise pattern (`pattern:<name>` tags). For the test split, `test_group` separates todo 109's test rows (`configs/test-ids-109.txt`) from rows added since. The judge takes up to 15 rows per request, fewer when long rows would pass 16,000 characters. `cleanup serve` gives each request slot 8,192 tokens of context, enough for a 1,000-word dictation and its 2,500-token reply limit. Every rate is a percentage of the rows it applies to:

- **Mechanical checks** (`model/metrics.py`): chrF and exact match against the reference; layout matches; formatting added where the reference has none, and missed where it has some; bad list markers and markdown; protected tokens (URLs, emails, paths, flags, identifiers, numbers) kept; fillers left beyond the reference's count; passthrough rows kept unchanged; `EMPTY` answered right and given wrongly; loops (a repeated 5-gram, or hitting the output limit); assistant openers; more than 30% of the words deleted.
- **Contract checks** (todo 112): em dashes, spelled-out counts next to a word (an upper bound: names and idioms match too), correction cues kept from the transcript, and `kept_words`, the share of the transcript's "maybe", "probably", "I think", "I guess", "I believe" and "because" that the output keeps.
- **Judge** (`--judge`, `prompts/judge/eval.md`, `LLM_JUDGE_MODEL`): answered or executed the transcript instead of editing it, and every fault, one per fact, by kind: `dropped`, `changed` (another value, or attached to the wrong thing), `wrong_side` (an abandoned value kept as if it stands), `unresolved` (both sides of a correction kept with the speaker taking one back, so nothing is lost) and `invented`. `facts_dropped` is the share of rows with a dropped, changed or wrong-side fault, and `facts_invented` with an invented one; an unresolved correction fails neither. `faults` gives each kind per judged row, per 1,000 reference words and as a share of rows, and `facts_lost` the three lossy kinds together, which keeps moving on long rows where nearly every row has a fault. Rows over 300 words get a request each.

**References after todo 112.** When every reference of a split is `gpt-6.1-sol`/high, the judge and score files are `<split>.r112.faults.judge.jsonl` and `<split>.r112.faults.scores.json`, so the old scores stay beside the new (`cleanup compare --stem test.r112.faults …`). `<split>.r112.judge.jsonl` and `.scores.json` are the pass/fail judge before fault counts; an unjudged run writes `<split>.r112.scores.json`, and the judge must be `gpt-6.1-sol` at high reasoning through codex. Model outputs are reused as they are; only rows with new ids (the re-made injected test rows) are generated. `cleanup eval --teacher --label 112` keeps the hosted teacher's outputs under the new prompt in `runs/teacher/eval-112/`. `cleanup judge-check all` checks the new judge first: on the 33 hand-reviewed rows (`runs/judge-review.verdicts.jsonl`) and on 300 stratified rows the old judge scored, against the old references, writing `runs/judge-check/report.json` and every disagreement to `disagreements.md`.

The references for teacher-sourced rows are the teacher's own outputs, so the teacher scores high on reference metrics such as chrF by construction. Compare with it on the judge rates and the checks that do not need a reference.

### Using it in the app

1. `cleanup serve full-inst-1e-5-e3`, the chosen model at Q8_0 (on a machine without a GPU, `--cpu --quant Q4_K_M` is faster for long dictations). The app never starts the server or ships the model.
2. In WhisperNow, choose the provider **Local cleanup model**. It defaults to `http://127.0.0.1:8080/v1` and needs no API key. The model name is required but llama-server ignores it; any name works.
3. Choose **Clean Up & Format** and turn visual context off. Settings refuse any other prompt, and visual context, with this provider.

