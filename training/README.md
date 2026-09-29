# WhisperNow training data

Builds the dataset for a local Clean Up & Format model (todo 108). Training and evaluation come later (todo 109).

Every row has one schema (`datagen/schema.py`): `id`, `input`, `output`, `source`, `category`, `tags`, `layout` (`plain`, `list`, `paragraphs` or `mixed`, computed from the output), `seed_group`, `split`, `license`, `teacher_model`.

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
| Hear what the recognizers write (speaks with Kokoro, then transcribes as the app does; `--model whisper` or `qwen3` re-transcribes the kept audio; `--report` only rewrites the report) | `uv run --extra audio datagen asr-probe --n 300` | `data/probe/<model>*.jsonl`, `report.<model>.json` |
| 7. Restyle half the aawaaz inputs (`--recheck` re-checks rejects, no calls) | `uv run datagen restyle` | `data/restyle/aawaaz.jsonl`, `rejected.jsonl` |
| 4. Filter, deduplicate, split, report | `uv run datagen build` | `data/build/{train,val,test}.jsonl`, `stats.md` |
| Draw rows for a hand review | `uv run datagen sample list --n 50 --seed 110` | stdout |
| Publish (private) | `uv run datagen publish <user>/<repo>` | Hugging Face dataset |
| Real-dictation sanity check (not an evaluation set) | `uv run datagen export-real` | `data/real/real.jsonl` (never uploaded) |

**Every LLM step resumes.** The judge commands append verdicts as batches finish and skip rows already judged. `generate` logs each batch in `<category>.batches.jsonl` and continues from there. Each response is cached by its exact request in `.cache/llm/`, so a rerun repeats no finished call. `generate --scale 0.05` runs a pilot at 5% of each target. `generate --category list_edits` runs one category.

`build` also makes the programmatic rows: filler injection and `EMPTY`.

## The LLM client

`datagen/llm.py` reads `LLM_*` from `.env`:

- **Concurrency.** A semaphore caps requests in flight (`LLM_CONCURRENCY=3`).
- **Retries.** Rate limits, 5xx and timeouts retry up to 7 times, with exponential backoff (4 s doubling, capped at 180 s) and jitter.
- **Batched judging.** Judges see 10–30 rows per request.
- **Reasoning.** Only the final message text is kept. `<think>` blocks and reasoning parts are dropped.
- **Models.** `LLM_MODEL` (`glm-5.3-flash`) is the teacher and generator. `LLM_JUDGE_MODEL` (`glm-5.3`) is the judge and validator.

**Backend.** `LLM_BACKEND=openai` calls any OpenAI-compatible endpoint directly with `LLM_BASE_URL` and `LLM_API_KEY`.

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

That is too few to measure filler style per model. The inputs we generate cover both styles: `recognizer` (punctuated, cased, mostly digits, fillers set off with commas) and `raw` (aawaaz-like: lowercase, little punctuation, numbers as words).

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

- A group with a preset test row (aawaaz's 3% held out by category, and handy's and disfl_qa's own test splits) is test.
- Other groups hash 5% into val.
- Groups made only of generated rows also hash 5% into test.

**Stats** are written to `data/build/stats.{json,md}`: counts per split, source, category, layout, license and tag, the dropped reasons, and the input and output length distributions.

### The current build

41,793 rows: train 37,773, val 1,958, test 2,062.

| Source | Rows | License |
|---|---|---|
| aawaaz | 22,600 (54%) | MIT |
| handy | 10,303 | CC-BY-4.0 |
| teacher (stage 3) | 3,546 | zai-output |
| disfl_qa | 3,212 | CC-BY-4.0 |
| injected (from aawaaz) | 2,132 | MIT |

- **Layouts:** plain 31,882, mixed 3,317, paragraphs 3,312, list 3,282. That is 24% with layout, and 16% with a list.
- **Every row** has a source and a license. None is non-commercial or LDC.
- **Lengths:** the median input is 58 words and the median output 43; the 95th percentiles are 170 and 143.

| Dropped | Rows |
|---|---|
| output repeated too often | 1,218 |
| near-duplicate input | 188 |
| invented subject line | 49 |
| changed protected token | 28 |
| added legal formality | 19 |
| exact duplicate input | 10 |
| invented signature | 8 |
| implausible length ratio | 8 |
| markdown | 2 |
| answered | 1 |
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
