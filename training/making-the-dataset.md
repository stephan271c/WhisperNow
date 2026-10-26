# How the WhisperNow Cleanup dataset was made

[WhisperNow Cleanup](https://huggingface.co/datasets/category271/whispernow-cleanup) is a dataset of noisy speech transcripts paired with cleaned output. It has 66,557 rows (train 60,836, val 2,424, test 3,297) and trains WhisperNow's local cleanup models.

I built it in four rounds:

- **Round 1** combined public datasets with generated dictations, and one teacher model rewrote every reference to follow the same rules.
- **Rounds 2 to 4** added rows aimed at the weak spots of models trained on the previous round: long dictations, short self-corrections, misheard words, cut-off words, and correction words used literally.

Val and test stayed fixed, so every round is scored on the same rows.

The main metric is **factual errors per 1,000 words**. The models learned early to remove fillers and fix formatting. The hard part is preserving the meaning, so the metric counts facts that were dropped, changed (a different number, name or date), kept from the abandoned side of a correction, or invented. Nearly all factual errors are lost facts; invented ones are 4–7%.

| factual errors per 1,000 words | round 1 | round 2 | round 4 |
|---|---|---|---|
| LFM2.5 1.2B | | 6.26 | 6.32 |
| LFM2.5 2.6B | 5.43 | 4.87 | 4.88 |
| Qwen3.5 4B | 3.20 | 2.92 | 3.01 |

These are measured on test, bench-long and bench-noise together (4,170 rows). For comparison, `glm-5.3` at max reasoning makes 2.03 and `glm-5.3-flash` at low reasoning 3.71. Round 2 cut factual errors by about a tenth. Rounds 3 and 4 kept it level while fixing specific failures; cut-off word pieces left in the output, for example, fell from 28–40% to about 2%. The round 4 models are the published ones.

## What is the ideal cleanup output?

The cleaned output is the words the speaker meant, edited for the page:

- fillers, stutters and false starts removed;
- only the corrected version of a self-correction kept;
- punctuation and casing fixed;
- a list laid out when the speaker dictates one.

It should not answer any questions in the transcript, and should not drop or invent facts. The references are written with the same prompt used with cloud models. The [dataset card](https://huggingface.co/datasets/category271/whispernow-cleanup#the-cleanup-rules) prints the full rules.

I settled the cleanup prompt and the judge prompts by reading rows by hand. A few decisions I made:

- **Keep hedges and reasons** ("maybe", "I think", "because …"). I couldn't draw a consistent line for when to drop a hedge, and small models already had difficulties with this in preliminary testing.
- **Drop a correction's cue** ("no wait", "scratch that") together with the abandoned words.
- **Write numbers as digits**, and use no em dashes and no markdown.
- **Keep the speaker's words**, with brands in their usual spelling, acronyms in capitals and quotations in quotation marks.
- **Lay out separate findings or requirements** as list items.

I also spot-checked the eval judge: I agreed with all 33 of its verdicts that I reviewed, and about 9 in 10 of the faults it reported were real.

## How I measured

A judge (`gpt-6.1-sol`, high reasoning) reads each output against its reference and lists every fault, one per fact. Four kinds count as **factual errors**:

- `dropped`: a fact is missing.
- `changed`: a different value, or a value attached to the wrong thing.
- `wrong_side`: the abandoned side of a correction kept as if it stands.
- `invented`: a fact the speaker never said, such as an added name, number or conclusion.

Nearly all factual errors are lost facts (the first three kinds): invented facts are 4–7% of the errors of every model here. One more kind is reported but not counted:

- `unresolved`: both sides of a correction are kept along with the take-back ("starts at 8 am, or you know what, make it 9"). The output should have kept only 9, but a reader can still tell which value stands, so no fact is lost.

Errors are counted rather than passing outputs, because nearly every output on a long dictation has at least one. The rate is per 1,000 reference words.

| benchmark | rows | what it holds |
|---|---|---|
| bench-long | 416 | Dictations of 151–946 words, including 49 real single-speaker turns from AMI meetings. |
| bench-noise | 461 | Short dictations with one disfluency pattern each (filler clusters, restarts, short cues, re-decisions), 60 control rows whose filler-like words carry meaning, and 136 recognizer transcripts of synthetic speech. |
| bench-cutoff | 150 | Test rows with cut-off word starts added, scored in code. |

I checked that the benchmarks don't overlap with the training data, using 8-word n-grams. Two models are compared on the rows they were tested on, by the ratio of their factual errors, with a 90% paired bootstrap interval. "Pooled" means test, bench-long and bench-noise together.

Note: the judge is not deterministic (2.7% of pairs judged twice got different verdicts).

## Round 1: the base

**Public data.** Each source was audited by a judge, and rows below its standard were relabelled or dropped:

- **aawaaz** (MIT): synthetic transcripts in 12 categories of topic and register, such as emails, recipes, meeting notes and code.
- **handy** (CC-BY-4.0): dictation edits. Rows that overlap Switchboard transcripts (LDC data, which is left out by license) were dropped.
- **Disfl-QA** (CC-BY-4.0): self-corrected questions.

Of the real-speech corpora, only AMI's meetings are spontaneous and licensed for this use. They came in with round 3.

**Generated rows.** I wanted a more representative sample of dictations, so I had a generator LLM write them by category, inspired by aawaaz's methodology.

- **Nine short categories**, such as dictated commands that must stay text, spoken symbols, list edits ("scratch the eggs") and lists inside messages: 3,550 rows, plus 680 pinned to test.
- **Eight long categories** of 150–1,000 words that mix hard cases, such as late corrections in an email, revised choices, meeting notes whose owners and dates change, and recipes with quantities: 8,476 rows.
- **Programmatic rows:** fillers and restarts injected into clean references, and filler-only inputs whose target is `EMPTY`.

**References.** `gpt-6.1-sol` wrote every reference, imported or generated, from the transcript alone and with the app's cleanup prompt. Each reference went through two checks:

1. Code checks for kept paths, URLs and numbers, em dashes, and leftover correction cues.
2. A completeness judge checks every fact.

A rejected reference was rewritten with the judge's note rather than dropped, since dropping would remove the hardest rows. Afterwards no reference had an em dash (23.5% before), 0.1% spelled out a count (15.7% before), and 97–99% of hedges and reasons were kept. Note: GLM models wrote and labelled the first generated rows. I stopped using them because their references broke the rules far more often, and their rows were checked again or replaced.

**Recognizer-style inputs.** The inputs are shaped to look like what the app's recognizers (Parakeet, Whisper and Qwen3-ASR) write, measured on 300 inputs spoken by TTS and transcribed by each. Half the aawaaz inputs were rewritten with a recognizer's casing and punctuation. A share of the rows got more fillers, closer to the 3.6 per 100 words of real dictation.

**Filters and splits.** Filters drop references with markdown, answers instead of edits, rewritten paths or identifiers, invented sign-offs, implausible lengths, or near-duplicates. Splits go by seed group, so a row and its variants share a split. Round 1 came to 52,110 rows, 46,389 of them in train.

**What its models got wrong.** The faults of the largest model trained on round 1, a Qwen3.5 9B, pointed at three gaps:

- **Long dictations.** Long rows were 16% of train but held 74% of its factual errors, at 1.7 times `glm-5.3` max's rate. Revised choices (4.3 times) and recipes (2.4 times) were the worst, and the model often dropped a choice the speaker only reported ("I was going to do 3,500").
- **Real speech.** It made 3.5 times `glm-5.3` max's factual errors on bench-long's AMI turns, and training had none.
- **Recognizer errors.** Half of its bench-noise errors were on the 136 recognizer transcripts.

A 2.6B trained on half the rows made 1.13 times as many factual errors, so more data would still help.

## Round 2: long dictations

Round 2 added 8,302 long dictations, weighted toward the worst categories and toward length (80% over 300 words), plus 1,882 spoken versions with injected fillers. Train grew to 56,573 rows.

| factual errors, round 2 ÷ round 1 | test | bench-long | bench-noise | pooled |
|---|---|---|---|---|
| LFM2.5 2.6B | 0.82 (0.77–0.88) | 0.99 (0.92–1.06) | 0.79 (0.67–0.94) | 0.90 (0.85–0.94) |
| Qwen3.5 4B | 0.89 (0.82–0.97) | 0.93 (0.86–1.00) | 1.08 (0.89–1.32) | 0.91 (0.87–0.97) |

Long rows gained the most: 0.75 times on the 2.6B's test rows of 451–1,200 words. Two problems remained:

- **The 4B got worse on short dictations**, 1.15 times up to 25 words and 1.19 times at 26–50, often by keeping a word from the abandoned part of a short self-correction. Disfl-QA questions, technical lists and spoken symbols added the most of these errors, and 9% of the Disfl-QA references turned out to keep such a word too.
- **The models copied misheard words.** On the recognizer transcripts the 2.6B and 4B made 34 and 33 factual errors, against 8 for `glm-5.3` max, mostly by keeping words like "tie place" for "Thai place" that the cloud model repairs.

## Round 3: weak spots

| part | rows | aimed at |
|---|---|---|
| reference review | 138 train and 16 test references fixed | Disfl-QA label noise, command case |
| short generated rows | 2,800 | short categories that slipped or trailed |
| AMI turns | 818 | real meeting speech |
| Disfl-QA questions | 45 | a planned batch, cut to its pilot |
| recognizer repair rows | 4,000, replacing their origins | copying misheard words |

- **Reference review.** I had `gpt-6.1-sol` recheck suspect references, such as Disfl-QA references against the original written question. A fix was kept only if the judge preferred it without knowing which version was new. After the fixes, the models did better on Disfl-QA than `glm-5.3` max, so I added 45 new Disfl-QA questions instead of the planned 1,500.
- **Short generated rows.** 700 rows each of technical lists, spoken symbols, list edits and lists inside messages. A short row with a correction cue got a second teacher sample, and was dropped if the two disagreed.
- **AMI turns.** Single-speaker turns of 50–1,000 words from meetings that bench-long does not use.
- **Recognizer repair rows.** Transcripts of TTS speech as Parakeet or Whisper heard it, mistakes included. The reference fixes a misheard word only when the text makes the right one clear, and rows that lost a fact for good were dropped. Each repair row replaced the row it came from. In this example Parakeet heard "As discussed" as "Umez disgust". The reference repairs it and keeps only Carlos, the corrected name:

  ```text
  Umez disgust, ask Fatima to review it rather ask Carlos to review it before the end of the week.
  ```

  ```text
  As discussed, ask Carlos to review it before the end of the week.
  ```

**Result.** Only the 1.2B was retrained on round 3, and its factual errors barely moved. Real use then showed a failure no round had targeted: cut-off word starts ("interf interface") left in the cleaned text. Training had about 100 such rows in 60,000, and the round 2 models left 28–40% of these pieces in their output. Round 4 targets it.

## Round 4: cut-off words and literal cues

Round 4 added two kinds of rows:

- **Cut-off words.** 3,000 round 3 train rows got one to three cut-off word starts in their input. Two thirds of the pieces came from TTS speech transcribed by Parakeet, and the rest were written. I replaced any row where a second model, Mercury, thought a piece might be a real word, name or acronym. Here the reference drops the piece "satur" along with the abandoned days:

  ```text
  i'll send the update on tuesday well actually thursday forget that satur uh saturday
  ```

  ```text
  I'll send the update on Saturday.
  ```

- **Literal cues.** 600 short dictations that use a correction word literally ("scratch behind the dog's ears", "hold on to the receipts"), 187 of them with a real correction as well:

  ```text
  Hold on to the receipts until Friday, no, Thursday, when Dana comes.
  ```

  ```text
  Hold on to the receipts until Thursday, when Dana comes.
  ```

Note: the cut-off rows alone made the 1.2B drop correction words that were meant ("scratch the surface", "never mind the cost"), because short training rows almost never kept them. Fewer cut-off rows in short dictations, plus the literal-cue rows, fixed this.

| factual errors, round 4 ÷ round 2 | test | bench-long | bench-noise | pooled | cut-off pieces left |
|---|---|---|---|---|---|
| LFM2.5 1.2B | 1.01 (0.95–1.08) | 1.02 (0.95–1.09) | 0.88 (0.73–1.05) | 1.01 (0.96–1.06) | 40.3% → 2.1% |
| LFM2.5 2.6B | 1.02 (0.96–1.09) | 0.98 (0.91–1.05) | 1.08 (0.91–1.29) | 1.00 (0.96–1.05) | 27.7% → 2.6% |
| Qwen3.5 4B | 0.98 (0.90–1.06) | 1.11 (1.02–1.21) | 0.75 (0.60–0.93) | 1.03 (0.97–1.09) | 32.5% → 1.6% |

Factual errors stayed level, and cut-off pieces left fell to about 2%. The 1.2B keeps a correction word that is meant 94% of the time and still drops 99% of real cues. The 4B's 1.11 on bench-long is likely run-to-run noise: compared at epoch 2, the two runs give 0.91.

## What I learned

- **Find the weak spots in the data.** Before adding rows, break down where the models make factual errors. Round 2 aimed at long dictations, where round 1's faults were concentrated, and it was the round that moved the overall number.
- **Examine prompts carefully.** I refined the cleanup prompt and the judge prompts over several passes of reading rows by hand. Each pass found a rule that was missing or a verdict that was wrong, and some apparent model errors turned out to be wrong references.

Two things did not work:

- **Cheaper transcript generators.** DeepSeek V4.1 Flash, the GLM models and Nemotron 3.5 Super kept 6–68% of their transcripts, against 84% for `gpt-6.1-sol`. Most failed the layout check, which made them cost more per kept row.
- **Anything beyond plain fine-tuning.** On an LFM2.5 350M, which was too weak to publish, DPO, rejection-sampling fine-tuning and a fact-weighted loss each failed to beat plain fine-tuning without regressions elsewhere.

## Known limits

- **The same model writes and grades.** `gpt-6.1-sol` wrote the references and also judges the outputs, so the scores may favor its style.
- **Little real speech.** Most inputs are text made to look like recognizer output, and the audio came from TTS voices, not people. The AMI turns are real speech, but from meetings, not dictation.
- **Generated long dictations are denser than real ones.** They pack late corrections and dense values into every dictation.
- **Some calls rest on judgment.** Examples are narrated plan changes, hedges the speaker settles much later, and names guessed from misheard text.
- **English only.**

## Reproducing

The commands, in order, are in [the training README](README.md). The code is under `src/whispernow_training/datagen/` and the prompts are under `prompts/`.
