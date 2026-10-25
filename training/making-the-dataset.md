# How the WhisperNow Cleanup dataset was made

[WhisperNow Cleanup](https://huggingface.co/datasets/category271/whispernow-cleanup) pairs a dictation transcript, as a speech recognizer writes it, with the text the speaker meant to write. It has 66,557 rows: train 60,836, val 2,424 and test 3,297. It trains WhisperNow's local cleanup models, which get only the transcript and return the cleaned text.

The data was built in four rounds. Round 1 assembled, audited and rewrote the base. Each later round started from an evaluation of models trained on the round before: score them on the held-out test split and the benchmarks, find where they lose facts, add training rows aimed at that, and train again. val and test did not change between rounds, so every round's models were scored on the same rows.

| round | added to train | train rows |
|---|---|---|
| 1, base | imported and audited public data, generated short and long dictations, programmatic rows; every reference rewritten by one teacher | 46,389 |
| 2, targeted | 8,302 long dictations weighted toward where round 1 models lost facts, and 1,882 injected spoken versions | 56,573 |
| 3, weak spots | 2,800 short generated rows, 818 AMI turns, 45 Disfl-QA questions, 138 corrected references, 4,000 recognizer repair rows in place of their origins | 60,236 |
| 4, cut-offs and literal cues | cut-off word starts in 3,000 inputs, 600 short dictations that use a correction word literally | 60,836 |

Sections 1–7 describe the parts every round shares. Sections 8–11 go through the rounds.

## 1. The goal and the contract

The cleanup step returns the words the speaker meant, edited for the page: fillers, stutters and false starts removed, only the corrected version of a self-correction kept, punctuation and casing fixed, and a list laid out when the speaker dictates one. It never answers or carries out what it hears, and it should lose no fact.

**One prompt for the app and the teacher.** The references follow the prompt WhisperNow sends to cloud cleanup models, a system message and a cleanup prompt (both printed in the [dataset card](https://huggingface.co/datasets/category271/whispernow-cleanup#the-cleanup-rules)). The teacher gets exactly that request, and a test fails if the training copy of the prompt drifts from the app's source. A fix to the references is therefore a fix to the app's prompt, and the cloud and local models aim at the same output. Under that prompt, a reference:

- drops fillers, verbal tics, stutters and false starts, and the cue of a clear self-correction ("no wait", "scratch that") together with the abandoned words, but keeps words when it is not clear they are a correction;
- keeps every item, quantity, name, deadline and constraint, every hedge ("maybe", "I think", with whoever the speaker attributes it to) and every reason ("because …");
- fixes grammar, spelling and obvious misheard words, but keeps the speaker's words and register;
- lays out what the speaker is making: lists for shopping lists, steps, recipes and meeting notes, sentences for messages, stories and explanations;
- writes amounts, counts, dates and times in digits, spoken symbols as symbols ("dash dash help" as `--help`) and brands in their usual spelling, and keeps code, paths and URLs exactly;
- uses no em dashes and no markdown, returns a dictated question or command as text, and returns `EMPTY` when nothing was dictated.

**How the rules were settled.** By reading rows, three times:

1. The project owner reviewed 33 rows the eval judge had failed. The judge was right on 30 and partly right on 3, but in several rows the reference was worse than the model output it was compared with: numbers as words, em dashes, kept correction cues, weaker lists. The read settled that references keep hedges and reasons (the owner could not draw a consistent line for dropping them, so a small model could not learn one), drop the cue with the abandoned words, use no em dashes and write counts in digits. It also led to rewriting every reference with one model (section 3).
2. A pilot rewrote 200 stratified rows, and the owner read 60 of them. Its 7 faults led to these rules: "wanna"-style words written out, quoted speech and titles in quotation marks, brands in their usual spelling, units and technical notation as written, separate requirements or findings as list items under a label, and long mixed lists grouped under labels.
3. After the change, the owner read the same 60 rows again and found 1 fault ("sku" for "SKU"), which added acronyms in capitals.

Two rulings were added to the judges only, with the teacher prompt unchanged. A hedge the speaker settles soon after ("maybe some jasmine rice, actually yeah get it") is a correction. Thinking aloud and mic checks may be kept or dropped, since the teacher drops about three-quarters of them and keeps the rest.

## 2. Sources

### Imported datasets

| source | license | audit and filters | rows in the final data |
|---|---|---|---|
| aawaaz (`shantanugoel/aawaaz-transcript-cleanup-dataset`) | MIT | 22,772 rows in 12 categories. `glm-5.3` judged every row for kept content, invented content, kept lead-ins and answering, and a register judge read the 11,866 conversational and formal rows for stories shortened into notes and contracts made more formal. 1,775 rows were flagged: 1,691 relabelled, 84 dropped. | 20,216 |
| handy (`MagicNoThief/handy-dictation-editing`) | CC-BY-4.0 | Dropped: 7,077 rows sharing a whole line or a quarter of their word 6-grams with Switchboard transcripts (LDC), 3,503 rows with Wikipedia sentences spliced into lowercase dictation, and 13,029 rows whose number words clash with the written forms. British spelling made American, and question marks added to questions. 186 of 11,878 audited rows flagged. | 9,796 |
| Disfl-QA (`google-research-datasets/disfl_qa`) | CC-BY-4.0 | Dropped: 1,330 rows whose output uses a word the speaker never said, 52 with number words, and the 64 the audit flagged. | 3,068 |
| AMI (`edinburghcstr/ami`), round 3 | CC-BY-4.0 | See [round 3](#10-round-3-weak-spots). | 818 |

The final counts are after the rewrite, the filters, and round 3's conversion of 3,054 of these rows into recognizer repair rows.

**Left out.** sotto (`juanquivilla/sotto-transcript-cleanup`) is MIT but partly written by Grok 4.20, and xAI's terms bar using its outputs to train competing models. A survey of real-speech corpora kept only AMI: Earnings-22 is CC BY-SA, TED-LIUM is CC BY-NC-ND, SPGISpeech, Buckeye and CANDOR are non-commercial, Switchboard, Fisher and ICSI are LDC data, DisfluencySpeech re-records Switchboard text, and read-speech sets such as LibriSpeech and Common Voice hold no spontaneous speech.

### Generated rows

**Short gap categories.** Nine categories the public data lacks: dictated commands that must stay text, prose that mentions things without being a list, spoken symbols and layout commands, lists inside messages, list edits ("scratch the eggs"), long rambles that switch between prose and lists, lists of questions and instructions, list items with code and paths, and dictations where either layout fits. Each batch asked a generator for about 50 transcripts and the layout it intended. The request held the category's guidance, a seeded pick of names and settings, an input style (recognizer or raw), and the rejection notes of the previous batch. The teacher labelled each transcript, code rejected a layout that contradicted the intent, and a validator judged realism, kept content, nothing invented, not answered, layout and clean output. Pass rates were 71–95% per category: 3,550 rows, plus 680 rows pinned to test so that each small category has about 100 test rows.

**Matching real dictation.** Against 19 real Parakeet dictations (about 540 words from one speaker; only aggregate counts were taken), the generated inputs put a comma after 99% of fillers, where real fillers are bare, and used stock correction phrases. The generator prompts were rewritten with bare fillers, doubled words, restarts without a cue, and corrections that use one short cue or none. On a 254-row pilot a comma followed 6% of fillers, and doubled words were 4 times as common.

**Long composed dictations.** Eight categories of 150–1,000 words that each mix several hard cases: late corrections in an email, revised choices with a reason, meeting notes whose owners and dates change, recipes with quantities, list corrections, technical steps, dense professional notes and multi-topic dictations. 40%, 35% and 25% of rows fall in the 150–300, 300–600 and 600–1,000-word buckets, and 70% are in recognizer style.

The first composed rows used GLM models in every role. On 40 long transcripts, a strict judge passed 2 references from `glm-5.3` at low reasoning, 14 from `gpt-6.1-sol` at high reasoning and 18 at xhigh. That judge was `gpt-6.1-sol` too, so it may favour its own model, but the `glm-5.3` faults read as real contract breaks: numbers left as words, the abandoned side of a late correction left standing, fillers kept. Every role moved to `gpt-6.1-sol`: the generator at low reasoning, the teacher at high (xhigh over 600 words), and the judge one row per request, at low up to 600 words and high above. The 2,581 GLM rows were gated again: 194 kept, 898 relabelled, and 1,489 dropped and replaced. 8,476 composed rows were kept.

### Programmatic rows

- **Filler injection.** Fillers, repeats, restarts and openers were injected into 2,000 clean one-paragraph aawaaz references at three noise levels, and the clean text is the target. Every long reference laid out as plain text or paragraphs also got one injected spoken version. A filler never goes inside a name, a number or a spoken symbol.
- **EMPTY.** 150 filler-only inputs, plus 90 pinned to test from a second seed, with the target `EMPTY`. 217 remain after deduplication.

## 3. One teacher, and the gates

At the end of round 1's generation, the references came from five writers: aawaaz's and handy's authors, `glm-5.3-flash`, `glm-5.3` and `gpt-6.1-sol`, each with its own habits. 23.5% of them had an em dash, because nothing told the writers not to. So about 49,000 references were rewritten by one model under the settled contract:

1. The teacher (`gpt-6.1-sol`, high reasoning) gets the transcript in the app's request. It never sees the old reference.
2. Code checks the new reference: protected tokens (paths, URLs, identifiers, numbers in digits) kept, no em dash, no correction cue kept from the transcript, no "wanna"-style words. What code cannot settle goes to the judge as a flag: a spelled-out number next to a word, a hedge or "because" the reference says less often than the transcript, a hedge followed within ten words by a yes or a decision, a value the speaker restarts straight away.
3. The completeness judge (`gpt-6.1-sol`, high) checks the reference against its own transcript: every fact kept, the right side of each correction, nothing invented, and clean output under the contract. Rows over 300 words get a request each.
4. A rejected reference is rewritten with the judge's note added to the request. Retry rounds check the rows still failing and rewrite each with the latest note and the note before it, so a fix does not bring back the earlier fault. A row is not dropped because one rewrite failed: that would remove the hardest rows and make the benchmarks easier than they are.

| | result |
|---|---|
| rows retried after the first rewrite | 464 |
| retry rounds | 14 |
| rows dropped | 7: five transcripts too garbled to fix without guessing (the project owner's ruling), one "Well" kept as an opener in every round, one correction whose scope is ambiguous |
| references with an em dash | 23.5% → 0% |
| references that spell out a count | 15.7% → 0.1% |
| "maybe", "probably", "I think", "because" kept | 97.0%, 99.2%, 98.0%, 98.4% |

The keep rates leave out a hedge repeated straight away, a closing aside ("I think that's everything") and a hedge followed within 10 words by a correction cue. After the settled-hedge ruling, the 221 passed rows the flag picks were checked again, and 43 were rewritten. The owner also ruled that a hedge on an approximate value may carry onto its correction, and that a currency obvious from context may be written ("around four fifty" for a plumber's estimate as "around $450").

Injected rows were made again from the rewritten references. Rows of rounds 2–4 were labelled by the same teacher and gates when they were made, so they skip the rewrite.

## 4. Input shaping

### What the recognizers write

300 sampled inputs were spoken by Kokoro TTS (a random English voice at 0.95–1.15× speed, in chunks with 0.15–1.4 s pauses; 2.3 hours of audio) and run through the app's voice-activity segmenter, segment join and three recognizers:

| | Parakeet TDT 0.6B v2 | Whisper distil-large-v3.5 | Qwen3-ASR 0.6B |
|---|---|---|---|
| periods, commas per 100 words | 5.0, 6.1 | 4.7, 5.7 | 8.4, 7.4 |
| numbers written as digits | 41% | 63% | 4% |
| mid-sentence fillers followed by a comma | 16% | 32% | 62% |
| word error rate against the spoken text (median) | 5.3% | 7.1% | 6.2% |

All three keep fillers and spoken layout commands ("new line") as words. Synthetic speech is flatter than a person's, so the sentence-break numbers are rough.

### Restyle

aawaaz inputs are lowercase and unpunctuated, but every recognizer the app ships punctuates. `glm-5.3-flash` rewrote a hashed half of the aawaaz inputs in a modern recognizer's style: sentence case, long sentences, spoken commands kept as words, numbers half digits and half words. Code checked every rewrite: after casefolding and collapsing number runs, the letters of the words must match the spoken input in order ("q three" and "Q3", "m and a" and "M&A"). 10,716 of 11,441 rewrites passed; the others keep their original input.

Each restyled row then gets one of four styles by a hash of its id, applied in code: Parakeet (30%: mostly bare fillers, "$" glued to the word before), Whisper (30%: a comma after 30% of fillers, some lowercase unpunctuated segments), Qwen3 (20%: short sentences, a comma after 60% of fillers, numbers back to the spoken words) or plain (20%: the rewrite as written). Inputs over 110 words are cut into 40–110-word segments and joined the way the app joins them, so a seam can leave a period mid-sentence. The reference does not change, since the words are the same.

### Hesitation

The generated long inputs had about 0.8 fillers per 100 words, and the short sources 1.6–1.7, against 3.6 in the real dictations. So 50% of the composed rows and 30% of the train and val rows of the generated short, handy and Disfl-QA sources are densified to 3–6 fillers per 100 words: fillers, clusters, "the uh the" repeats, repeated words and, in Parakeet style only, pause punctuation. Slots are weighted the way real fillers fall: 37% at a sentence start, 21% after a conjunction or discourse word, 16% after a comma, and 11% each after a determiner and after a preposition or verb. 67% of the generated recognizer-style rows also have the commas around their fillers removed, towards Parakeet's bare fillers.

These steps change only fillers, repeats, punctuation and casing, which the reference drops or rewrites anyway, so references stay right without a judge. Each is a pure function of the row, applied after deduplication; the near-duplicate check ignores fillers, so a second copy that only adds fillers would have been dropped.

## 5. Filters, deduplication and splits

A row is dropped for the first of these that applies:

- markdown in the output (bold, backticks, `#` headings, links, tables) or a malformed list;
- an answer: an output that opens like a reply ("Sure", "Here is") where the speaker did not;
- a path, URL, email, identifier or flag the output rewrote ("x25" as "X.25"); a dropped token is not flagged, since a self-correction may remove it;
- a `Subject:` line, a signature, or legal wording ("shall", "hereby") the speaker never said;
- an output over 1.3 times the input plus 8 words, or a prose output under 30% of a 20-word input (lists and self-corrections may shrink further);
- an exact or near-duplicate input (MinHash over word 3-grams without fillers, Jaccard 0.85 or more), or an output repeated more than 3 times, which trims handy's templates (one output appeared 57 times). Test rows and earlier sources win a duplicate, and a later round's row loses to every earlier row.

Several of these came from hand reads of 100 cleaned aawaaz rows and two rounds of 200 built rows, which found stories shortened into notes, "is going to pay" written as "shall pay", and subject lines and sign-off names nobody dictated. Before the rewrite, the largest drops were repeated outputs (1,218) and near-duplicate inputs (219). Rounds 2, 3 and 4 dropped 7, 29 and 17 of their new rows.

**Splits** go by `seed_group`, so a row and every row made from it share a split:

- A group with a preset test row is test: aawaaz's 3% held out per category, handy's and Disfl-QA's own test splits, the 680 generated test rows and the 90 second-seed `EMPTY` rows.
- Other groups hash 5% into val, and groups made only of generated rows another 5% into test.
- Rounds 2–4 add rows to train only and copy val and test byte for byte. Round 4 also drops any new row that shares 6 consecutive words with a val, test or benchmark input, so a phrase the evaluation asks about is not taught by name.
- Round 3's review corrected 16 Disfl-QA test references (section 10). The correction went into the one test file all rounds share, and those rows were judged again for every earlier model, so comparisons across rounds use the same references.

## 6. Benchmarks

| benchmark | rows | what it holds |
|---|---|---|
| bench-long | 416 | Dictations of 151–946 words, 152, 146 and 118 in the 150–300, 300–600 and 600–1,000-word buckets: 367 composed rows in the eight long categories and 49 real single-speaker turns from AMI's test and validation meetings. |
| bench-noise | 461 | Dictations of 20–64 words with one disfluency pattern each (filler clusters, a filler between a repeated determiner, repeats, restarts, short cues, reasoned corrections, re-decisions, pause punctuation, thinking aloud; 44–45 rows each), and 60 control rows whose filler-like words carry meaning ("I mean it this time"). 136 rows are Parakeet transcripts of Kokoro speech, kept only when no fact changed and the pattern survived. |
| bench-cutoff | 150 | Test rows with 191 cut-off word starts added the way round 4 adds them. Built after round 3 and scored in code. |

The generated benchmark rows came from their own prompts, name and setting pools and seeds, so none shares a template or seed group with a training row, and no benchmark input shares more than 30% of its 8-word shingles with a training input. Code confirms each bench-noise row shows its pattern. The benchmark references went through the full rewrite, and a later audit corrected 31 of them by hand: a recognizer "?" after a list item kept as a question, names spelled unlike what was said, dictation commands such as "send it" kept as text, and corrections of an earlier message treated as self-corrections.

bench-cutoff reports three numbers: the share of cut-off pieces left in the output, the words lost against the model's output on the same test row without cuts, and the share of 222 look-alike word pairs that references use on purpose ("per person") that the output keeps.

## 7. Measurement

**Facts lost per 1,000 reference words.** A judge (`gpt-6.1-sol`, high reasoning) reads each output against its reference and lists every fault, one per fact, by kind: `dropped`; `changed` (another value, or attached to the wrong thing); `wrong_side` (an abandoned value kept as if it stands); `invented`; and `unresolved` (both sides of a correction kept together with the take-back, so nothing is lost). The first three are facts lost. The count is per 1,000 reference words, so one lost fact in a 10-word dictation counts as 100, and short rows have high rates. Rows over 300 words get a judge request each. A pass/fail rate per row cannot show progress on long dictations, where nearly every output loses at least one fact, which is why the count is the target.

**Checking the judge.**

- It failed all 33 rows of the owner's review, matching every verdict.
- On 300 stratified rows it agreed with the earlier `glm-5.3` judge on 80%. Most of the 55 rows that only it failed hold real faults the earlier judge missed.
- A hand read of 57 faults it found on 14 failed rows judged about 52 of them real.
- The benchmark audit (section 6) added rules for what is not a fault: a recognizer "?" after a statement, a name spelled to sound like the transcript's, a correction of something said before this dictation, the speaker on an "Attendees:" line, and remarks about the dictation itself.
- It is not deterministic: on 5,207 output pairs judged twice with byte-identical requests, 2.7% got different verdicts.

**Comparisons.** Two models are compared on the rows both have a verdict for, by the ratio of their facts lost, with a 90% paired bootstrap interval over rows. "Pooled" means test, bench-long and bench-noise together (about 4,170 rows). Two cloud models give reference points: `glm-5.3` at max reasoning and `glm-5.3-flash`, each sent WhisperNow's full prompt.

**Code checks.** Beside bench-cutoff, a word-level check reads test and bench-noise rows whose transcript says a correction word ("scratch", "never mind", "hold on", "wait" and others). It counts how often an output keeps the word where the reference keeps it (the word was meant) and drops it where the reference drops it (the speaker was correcting themselves).

## 8. Round 1: base

Round 1 is sections 2–5: 52,110 rows, 46,389 of them in train.

**Evaluation.** Facts lost pooled, per 1,000 reference words: LFM2.5 2.6B 5.12, Qwen3.5 4B 3.07, against 3.44 for `glm-5.3-flash` and 1.88 for `glm-5.3` at max reasoning.

To decide what to add, the faults of the largest model trained on round 1, a Qwen3.5 9B (1,365 facts lost over test and both benchmarks), were broken down by the rows they fell on:

- The long composed rows, 16% of train rows, held 73% of its lost facts, at 1.7 times glm-5.3 max's rate on the same rows. All other rows (84%) held 17% and were at 0.9 times glm-5.3 max.
- 79% of its lost facts were in inputs over 200 words, and 54% over 450. Such inputs were 20% of training rows.
- By category, the excess over glm-5.3 max was largest in `revised_choice` (4.2 times) and `recipe_quantities` (3.0 times), then `meeting_actions`, bench-long's AMI turns (3.4 times, with no AMI rows in training), `multi_topic` and `dense_professional`. `list_corrections`, `technical_steps` and `email_late_correction` were about level.
- 74% of the `revised_choice` faults dropped an earlier choice or side remark the speaker reported ("I was going to do 3,500"), which reads as the model applying "drop the abandoned version" where nothing was abandoned.
- Training had no real recognizer errors. bench-noise's 136 Parakeet transcripts held half of the 9B's bench-noise faults, at 5 times glm-5.3 max.

A 2.6B trained on a fixed half of round 1's train rows lost 1.14 times (1.08–1.20) the facts of the full run pooled, so more rows still help. Round 2 put the new rows where the faults were rather than adding more of everything.

## 9. Round 2: targeted

**Added.** 8,302 composed rows from the round 1 recipe, weighted toward the worst categories and the longer buckets:

| category | share | | bucket | share |
|---|---|---|---|---|
| revised_choice | 32% | | 150–300 words | 20% |
| recipe_quantities | 22% | | 300–600 words | 40% |
| meeting_actions | 12% | | 600–1,000 words | 40% |
| multi_topic | 10% | | | |
| dense_professional | 8% | | | |
| list_corrections | 6% | | | |
| technical_steps | 5% | | | |
| email_late_correction | 5% | | | |

They draw names and settings from pools of their own (66 names, 97 settings) that share none with round 1's or the benchmarks'. The teacher wrote at high reasoning, or xhigh over 600 words. 2,576 generated transcripts were rejected, mostly a layout that contradicted the generator's intent (1,039), the wrong side of a correction (613) or a missing fact (450). Each long reference laid out as plain text or paragraphs also got an injected spoken version (1,882), and the build filters dropped 7 rows: 10,184 new train rows.

**Tried and set aside: recognizer-heard inputs.** Kokoro spoke 10,153 round 1 train transcripts (133.5 hours of audio), and Parakeet (two thirds) or Whisper transcribed them as the app does. The row's reference stays right only where the cleanup can recover the spoken words, so a restore step aligned the heard words with the spoken ones and put the spoken words back wherever a difference touched a content word, a number, a name, a negation or a correction cue, or the recognizer made up a word. Casing, punctuation, sentence breaks and number formats stayed as heard. 9,296 rows qualified to replace their origins' inputs. The restored inputs differed from the spoken words in only 0.3% of words (4.1% before the restore), so they taught recognizer formatting and almost no mishearings. Round 2 was built without the swap; the heard text came back in round 3.

**Evaluation.** Facts lost, round 2 ÷ round 1:

| model | test | bench-long | bench-noise | pooled |
|---|---|---|---|---|
| LFM2.5 2.6B | 0.83 (0.77–0.88) | 0.98 (0.91–1.05) | 0.83 (0.71–0.99) | 0.90 (0.85–0.94) |
| Qwen3.5 4B | 0.91 (0.83–0.98) | 0.91 (0.85–0.98) | 1.10 (0.92–1.36) | 0.92 (0.87–0.97) |

- Long rows gained most: 0.74 times on the 2.6B's test rows of 450–1,200 words, and 0.84–0.88 times on the 4B's pooled rows of 200 words and up.
- The 4B lost more on short rows: 1.17 times (1.00–1.37) up to 25 words and 1.27 times (1.09–1.49) at 25–50 words, pooled. The 2.6B did not.
- The 4B's rise on rows under 50 words (178 → 216 facts lost over 2,480 rows) was spread over many categories. The largest were Disfl-QA questions (+11), `technical_list` (+7), `spoken_symbols` (+5) and `list_edits` (+3). Typical faults were a word kept from the abandoned part of a short self-correction, and a hesitant "A or uh B" read as a correction that drops A. The 4B also capitalized line-initial commands ("kill -9" as "Kill -9"), which the references keep lowercase.
- It was not overfitting from the third epoch: the epoch-2 checkpoint lost 1.11 times the final's facts on rows of up to 25 words.
- Reading Disfl-QA references against their source questions found label noise present since round 1: 9% of train references (263 of 2,789) and 8% of test (24 of 291) keep a content word from the abandoned part of the question.
- Two short categories still trailed glm-5.3 max: `list_edits` and `mixed_message`.
- On bench-noise's 325 text-only rows, the 2.6B and 4B lost 23–28 facts, level with the cloud models' 25–26. On its 136 recognizer transcripts they lost 32 each, against 7 for glm-5.3 max, and 26–27 of the 32 were `changed`: the models copied misheard words ("tie place" for "Thai place") that the cloud model repairs. Training had no example of such a repair, since the heard rows had restored every mishearing.

## 10. Round 3: weak spots

| part | rows | aimed at |
|---|---|---|
| reference review | 138 train and 16 test references corrected | Disfl-QA label noise; command case |
| short generated rows | 2,800 | the four short categories above |
| AMI turns | 818 | real meeting speech, absent from training |
| Disfl-QA questions | 45 | a planned batch, cut to its pilot |
| recognizer repair rows | 4,000, replacing their origins | copying misheard words |

**Reference review.** `gpt-6.1-sol` reviewed each Disfl-QA reference that differs from the written question its transcript was made from, evidence the teacher never had, and each reference line that starts with a capitalized command or file name. Every proposed fix was shown blind to the judge role, old and new reference in random order, and counted only when the new one won: 158 of 164 did. 138 train references changed (135 Disfl-QA, 3 command case), and the 16 test fixes went into the shared test file.

**Disfl-QA.** With the test references fixed, Disfl-QA was no longer a weak spot: on its 291 test rows the 2.6B and 4B lost 20–30 facts, against 42 for glm-5.3 max and 63 for glm-5.3-flash. The planned 1,500 unused questions were cut to the 50-row pilot, drawn from Disfl-QA's train and validation splits with no question that a test row asks; 45 passed.

**Short generated rows.** 700 rows each of `technical_list`, `spoken_symbols`, `list_edits` and `mixed_message`, from round 1's generation loop and category prompts, with `gpt-6.1-sol` as generator (low reasoning), teacher and judge (high), plus one more check. A row under 100 words with a correction cue got a second teacher sample, and if the two samples kept different words the row was dropped, since the transcript then does not settle what the speaker meant. 89 of 2,945 checked rows were dropped, and 3 references were fixed for command case. Generation continued until each category had 700 rows.

**AMI turns.** After round 1, the 9B lost 3.4 times glm-5.3 max's facts on bench-long's AMI turns, and training had no real speech. Round 3 took single-speaker turns from the meetings of AMI's train split, which share no meeting with bench-long's: every turn of 150–1,000 words and 400 turns of 50–150 words, at most 3 from a meeting. A turn reads through other speakers' backchannels of up to 3 words and ends at anything longer or at a silence over 4 seconds. AMI's upper-case, unpunctuated text becomes raw-style lowercase. The teacher labelled the turns under the usual gates, without the realism and layout checks a generated row gets. 821 of 920 passed (the rejections were mostly `clean`, 69, and a missing fact, 27), and 818 remained after the filters.

**Recognizer repair rows.** These reuse round 2's heard records, this time with the recognizer's raw text, misheard words and all. The pool is the records where the restore had to put a word back. The teacher labelled the heard text without seeing what was said, so it can fix only what the text itself shows. The judge then read what was said and failed any row that lost a fact, so a mishearing nobody could undo drops the row; a name spelled differently from what was said does not count as lost. The project owner's condition was to correct only what sounds similar and is clearly wrong, and never to add what was not said. For example:

```text
Umez disgust, ask Fatima to review it rather ask Carlos to review it before the end of the week.
```

```text
As discussed, ask Carlos to review it before the end of the week.
```

Of 6,838 records labelled in pool order, 4,132 passed (60%). 1,834 failed because the recognizer had changed a fact the text cannot recover ("switched" heard as "switch", "er" as "error"), and 872 failed the gates. The first 4,000 to pass the build filters replaced the train rows they were made from, so no transcript is trained on twice: 2,687 Parakeet and 1,313 Whisper transcripts.

**Evaluation.** The 2.6B and 4B were not retrained on round 3; an LFM2.5 1.2B, first trained on round 2, was. Round 3 ÷ round 2 for the 1.2B:

- test facts lost 4.34 against 4.52 per 1,000 words, but on test dictations of up to 25 words 6.59 against 5.94 (1.11 times);
- bench-noise 4.64 against 4.88. Its `changed` rate, the fault the repair rows aimed at, stayed at 3.11 per 1,000 words, and on the 136 recognizer transcripts it lost 36 facts against 35;
- bench-long was not fully judged by `gpt-6.1-sol`.

The evaluation cannot separate the effect of each part of round 3.

**A new problem: cut-off words.** After round 3, a cut-off word start in a real dictation reached the cleaned text. Speakers often cut a word off and say it again, and the cleanup should drop the piece, but the training data held about 100 such rows in 60,000. bench-cutoff was built to measure it (section 6). The round 2 1.2B left 40.3% of the pieces in its output, the round 3 1.2B 17.8%, and the round 2 2.6B and 4B 27.7% and 32.5%.

## 11. Round 4: cut-offs and literal cues

**Cut-off rows.** 3,000 round 3 train rows got one to three cut-off word starts in their input, with a fixed count per length bucket, and kept their id and reference.

- In 1,992 rows, a short window around each cut word was spoken with the word's start said first, by Kokoro or Phonon TTS, and transcribed by Parakeet. Parakeet writes the piece as a bare prefix ("inv invoice"), a sound-alike word ("throw, through") or letters ("EV every"). The heard piece went into the input only when the words around it were heard as said. The other 1,008 rows got a written piece: a bare prefix, a prefix and a letter ("set s settings"), or a word the cut word starts with.
- A piece is kept only where the text shows it is a stumble. It starts with the cut word's first two letters; it is one word, or one word and spelled letters; that word is not a common word ("not" before "notify"), a number or an acronym the references use ("SEC" before "security"); and the two words are not a pair the references use on purpose ("per person").
- Mercury then asked three yes/no questions about every cut: could the piece be a word the speaker meant, would keeping it state a different fact, could it be a name or acronym. A row with any flagged cut was replaced. On 44 hand-labelled cuts, the threshold caught 15 of the 19 unsafe ones and rejected 3 of 25 safe-looking ones. 4,732 rows were tried to keep 3,000.
- The 3,000 rows hold 6,576 cuts: 3,467 bare prefixes, 1,477 prefixes with a letter, and 1,632 pieces heard or written otherwise.

```text
i'll send the update on tuesday well actually thursday forget that satur uh saturday
```

```text
I'll send the update on Saturday.
```

**A first build, and what it showed.** A 1.2B trained on round 3 plus cut-off rows spread over lengths in proportion to the pool left only 4.7% of pieces on bench-cutoff. But on test dictations of up to 25 words it lost 1.12 times (1.00–1.25) the facts of the round 3 1.2B, and 1.24 times (1.08–1.44) those of the round 2 1.2B. Its new short failures were phrases that use a correction word literally: "scratch the surface", "forget the password reset, I already did it", "never mind the cost", "hold on", "that's wrong". In train rows of up to 25 words, such a word was almost never kept: "scratch" in 0 of 220 rows, "never mind" in 0 of 89, "that's wrong" in 2 of 49 and "hold on" in 1 of 53. Round 3's short generated rows had added 80 more short rows where "scratch" is a cue.

So the round 4 build halves the short bucket's share of cut-off rows (492 rows under 50 words) and adds literal-cue rows.

**Literal-cue rows.** 600 generated short dictations that use a correction word literally ("scratch behind the dog's ears", "make that call before noon"): 448 of up to 25 words and 152 of 26–60. 197 open with the word, and 187 also hold a real correction:

```text
Hold on to the receipts until Friday, no, Thursday, when Dana comes.
```

```text
Hold on to the receipts until Thursday, when Dana comes.
```

Every row got a second teacher sample, and 6 of 636 were dropped because the two samples kept different words. 14 more were dropped for sharing 6 consecutive words with a held-out input, and 3 as near-duplicates.

**Evaluation.** Facts lost, round 4 ÷ round 2, and bench-cutoff pieces left:

| model | test | bench-long | bench-noise | pooled | pieces left, round 2 → round 4 |
|---|---|---|---|---|---|
| LFM2.5 1.2B | 1.00 (0.94–1.07) | 1.02 (0.95–1.10) | 0.88 (0.73–1.05) | 1.01 (0.96–1.06) | 40.3% → 2.1% |
| LFM2.5 2.6B | 1.03 (0.96–1.09) | 0.98 (0.91–1.06) | 1.07 (0.89–1.29) | 1.01 (0.96–1.06) | 27.7% → 2.6% |
| Qwen3.5 4B | 0.96 (0.89–1.05) | 1.13 (1.03–1.23) | 0.78 (0.63–0.97) | 1.03 (0.97–1.10) | 32.5% → 1.6% |

- Facts lost on held-out rows stayed level for all three sizes, while cut-off pieces left fell to 1.6–2.6%.
- The 1.2B's short-row loss was gone: 5.94 per 1,000 words on test dictations of up to 25 words, as on round 2, and 0.81 times (0.69–0.92) the first build's. Where a correction word is meant, in test and bench-noise rows of up to 25 words, it kept the word in 94% of 50 cases (88% on round 2, 74% on the first build), and it still dropped 99% of 373 real cues. The 2.6B and 4B kept 92% and 94%, as on round 2.
- The 4B's bench-long ratio flips between epochs: at epoch 2 the round 4 run lost 0.90 times (0.83–0.98) the round 2 run's facts on bench-long. So the 1.13 at the final epoch reads as run-to-run variation, not an effect of the data. The 4B's short rows improved on round 2: 0.78 times (0.66–0.92) at 25–50 words, pooled.
- On bench-noise's 136 recognizer transcripts, facts lost were 30 (1.2B), 31 (2.6B) and 20 (4B), against 35, 32 and 32 on round 2 and 7 for glm-5.3 max. Between round 2 and round 4, this gap narrowed clearly only for the 4B.

These round 4 models are the published ones.

## 12. What did not work, or was left out

- **Heard-input swap** (round 2): set aside, as described there.
- **Other transcript generators.** On round 2's run, writing a transcript was about 17% of the cost of a kept row (the teacher at xhigh on a long row costs about 4 times a generator request), so another generator pays only if it keeps at least about 70% of its transcripts. `gpt-6.1-sol` kept 84%. DeepSeek V4.1 Flash kept 55–68%, GLM-5.3 and GLM-5.3 Flash through NVIDIA 7–33%, and Nemotron 3.5 Super 6–13%. Most of the losses were the layout check: the GLM models labelled 62–78% of these list-heavy dictations as prose. The 93 pilot rows that passed every gate are in round 2.
- **More Disfl-QA questions** (round 3): cut from 1,500 to 50 once the reference fixes showed Disfl-QA was no weak spot.
- **An LFM2.5 350M model.** We also trained the smallest LFM2.5, at 350M parameters. It made too many mistakes to be competitive, so it is not published. The early experiments ran on it: drawing new recognizer styles and hesitation for each epoch scored the same as one fixed draw, and DPO on its own samples, rejection-sampling fine-tuning and fact-weighted loss each failed to beat plain fine-tuning without regressions elsewhere (the DPO run that cut the most dropped facts looped on 32% of bench-long rows). Every published model is plain fine-tuning on one fixed draw.
- **A rule for thinking aloud and mic checks.** The teacher is inconsistent about them, and the judges accept either choice rather than adding a rule to the app's prompt.
- **Relabelling narrated plan changes**: left ambiguous on purpose.

## 13. Known limits

- **The same model writes, gates and scores.** `gpt-6.1-sol` wrote the references, judged them, and judges the models against them, so it may favour its own habits. The owner's hand reads are the outside check.
- **Little real speech.** Most inputs are text styled like recognizer output. Recognizer transcripts of speech appear in the repair rows and the spoken cut-off pieces, and that speech came from TTS voices, which are flatter than people and do not stutter. The 818 AMI turns transcribe real speech, and they are meetings rather than dictation.
- **Generated long dictations are denser than real speech.** They pack late corrections and dense values into every dictation.
- **Narrated plan changes** go both ways, 29% kept and 71% dropped.
- **Repair references can carry a lucky guess.** One heard "a lyph" and wrote "Elif". The heard text alone does not settle the name, but it matched what was said, so the judge passed it.
- **Settled hedges are found by a flag.** The recheck covered hedges followed within ten words by a yes or a decision; a hedge settled later rests on the judge alone.
- **Some literal-cue phrasings repeat:** "hold on to the receipts" in 10 of the 600 rows, "I mean it this time" in 7.
- **Round 1 is kept as built.** Today's code rebuilds it with small differences, so later rounds build from the stored round 1 files.
- **English only.**

## 14. Reproducing

The commands are in [the training README](README.md). The code is under `src/whispernow_training/datagen/` and the prompts (teacher, generators, judges, reviews) under `prompts/`. Every LLM step resumes where it stopped and caches each response by its exact request, so a rerun repeats no finished call. `datagen build` rebuilds rounds 2–4 from the stored round 1. Speaking and transcribing need the audio extras and local copies of the TTS and recognizer models.
