---
license: apache-2.0
base_model: openbmb/MiniCPM5-1B
base_model_relation: finetune
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- dictation
- transcript-cleanup
- speech-to-text
- minicpm5
- gguf
datasets:
- OWNER/whispernow-cleanup
---

# WhisperNow cleanup model (MiniCPM5 1B)

This model turns a speech recognizer's transcript into the text the speaker meant to write. It removes fillers, false starts and repeats, keeps only the corrected version when the speaker corrects themselves, fixes punctuation, and lays out a list when the speaker dictates one. It is a full fine-tune of [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B), and one of the local cleanup models in [WhisperNow](https://github.com/stephan271c/WhisperNow), a desktop dictation app.

It is the middle one of three cleanup models trained the same way. It loses a fifth fewer facts than the 350M on the test split, and takes about twice as long:

| model | Q8_0 size | facts lost per 1,000 words (test) | 100 / 1,000 words, GPU | 100 / 1,000 words, CPU |
|---|---|---|---|---|
| [LFM2.5 350M](https://huggingface.co/OWNER/whispernow-cleanup-lfm2.5-350m) | 379 MB | 9.41 | 0.11 / 1.1 s | 0.7 / 8.5 s |
| **MiniCPM5 1B** (this model) | 1.15 GB | 7.54 | 0.18 / 1.9 s | 1.3 / 16.5 s |
| [LFM2.5 2.6B](https://huggingface.co/OWNER/whispernow-cleanup-lfm2.5-2.6b) | 2.87 GB | 4.21 | 0.34 / 3.9 s | 4.5 / 56 s |

[Evaluation](#evaluation) explains the numbers.

## Examples

Two transcripts from the test split, with this model's output (llama.cpp, Q8_0, one request at a time). The second matches the reference exactly; in the first, the reference has no "And" before "Giuliana".

A dictation with several self-corrections:

```text
alright so the hides need to soak in the lime bath for uh three days no actually five days for these thicker buffalo hides and the ph should be around 11 no wait 12.5 to properly loosen the hair and giuliana said to add an extra 20 no 30 pounds of hydrated lime to this batch because the last one came out underprocessed
```

```text
The hides need to soak in the lime bath for 5 days for these thicker buffalo hides, and the pH should be around 12.5 to properly loosen the hair. And Giuliana said to add an extra 30 pounds of hydrated lime to this batch because the last one came out underprocessed.
```

A list inside a message. The request at the end is dictated text, so it stays in the output:

```text
Hey, hardware store list before I lose it. Paint, the same gray as the hallway, painter's tape, 2 more rollers, and um drop cloths. Actually just add it all to the list app, I'm driving.
```

```text
Hey, hardware store list before I lose it:
- Paint (the same gray as the hallway)
- Painter's tape
- 2 more rollers
- Drop cloths

Just add it all to the list app. I'm driving.
```

## What it does

The model gets no instructions, only the transcript. It learned the cleanup rules that WhisperNow sends to hosted models (the full prompt is in the [dataset card](https://huggingface.co/datasets/OWNER/whispernow-cleanup#the-cleanup-rules)):

- It removes fillers and verbal tics ("um", "like", "you know", "so" opening a sentence), stutters, repeated words and false starts.
- When the speaker clearly corrects themselves, it keeps only the corrected version and drops the cue ("no wait", "scratch that").
- It keeps hedges ("maybe", "I think") and reasons ("because ...").
- It fixes grammar, spelling, punctuation and capitalization, but keeps the speaker's words and tone.
- It lays out shopping lists, to-do lists, steps, recipes and meeting notes as lists under a short label taken from the speaker's words. Messages, stories and explanations stay as sentences.
- It writes amounts, counts, dates, times and percentages in digits, and turns spoken punctuation and symbols ("dash dash help") into the real thing. Code, paths and URLs stay exactly as dictated.
- A question or command in the transcript is cleaned up and returned as text, never answered or carried out.
- When the transcript holds no dictated words (only fillers or noise), it returns `EMPTY`.
- It uses no bold, italics, headings or em dashes.

## How to use

### Prompt

There is no system prompt. Send one user turn that holds only the transcript, with leading and trailing whitespace trimmed. The chat template (`chat_template.jinja`, also embedded in the GGUF files) renders it as:

```text
<s><|im_start|>user
{transcript}<|im_end|>
<|im_start|>assistant
```

The template rejects a system turn.

### Settings

WhisperNow runs the model, and every score below was measured, with:

- greedy decoding (temperature 0);
- a repeat penalty of 1.05 over the whole prompt and reply (llama.cpp: `repeat_penalty` 1.05, `repeat_last_n` 4096);
- a reply budget of 2.5 tokens per transcript word, and at least 64.

These differ from the base model's suggested sampling (temperature 0.7, top-p 0.95). Training rows were at most 2,816 tokens of prompt and reply together, which is about 1,000 words of dictation.

### llama.cpp

```bash
llama-server -m model-Q8_0.gguf --jinja -c 4096
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "um can you send me the report by friday no wait thursday"}],
  "temperature": 0,
  "repeat_penalty": 1.05,
  "repeat_last_n": 4096,
  "max_tokens": 64
}'
```

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "OWNER/whispernow-cleanup-minicpm5-1b"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="bfloat16")

transcript = "um can you send me the report by friday no wait thursday".strip()
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": transcript}],
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
reply = model.generate(
    **inputs,
    do_sample=False,
    repetition_penalty=1.05,
    max_new_tokens=max(64, int(2.5 * len(transcript.split()))),
)
print(tokenizer.decode(reply[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The scores below come from llama.cpp at Q8_0. Transformers in BF16 can pick a different word where two are almost equally likely.

## Files

| file | what it is |
|---|---|
| `model.safetensors`, `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | BF16 weights for transformers |
| `model-Q8_0.gguf` | 1.15 GB. WhisperNow downloads this file, and the scores below use it. |
| `model-Q4_K_M.gguf` | 688 MB. Not scored. On the 350M model, Q4_K_M lost about 10% more facts than Q8_0. |

## Evaluation

**Test sets.**

- **test:** 3,297 held-out rows of the [training dataset](https://huggingface.co/datasets/OWNER/whispernow-cleanup), from 1 to about 1,000 words.
- **bench-long:** 416 dictations of 150–1,000 words, 49 of them real meeting turns from the AMI corpus.
- **bench-noise:** 461 short dictations, each with one kind of disfluency (restarts, repeats, filler clusters, corrections and others), plus a control set. 136 of them are real Parakeet transcripts.

The benchmarks were generated with their own prompts, names and seeds, so no benchmark row shares a template or seed with a training row.

**Scoring.** A judge model (OpenAI's `gpt-6.1-sol` at high reasoning) compares each output with the reference and lists its faults. **Facts lost** counts facts that were dropped, changed, or put on the wrong side of a correction, per 1,000 reference words. Lower is better. Our models ran at Q8_0 in llama.cpp with the settings above. The hosted glm-5.3 models received WhisperNow's full cleanup prompt. Rates are over the rows every model has a verdict for: 3,293 test rows, 413 bench-long rows and all 461 bench-noise rows.

**Facts lost per 1,000 reference words:**

| model | test | bench-long | bench-noise |
|---|---|---|---|
| glm-5.3 at max reasoning (hosted) | 1.73 | 2.15 | 2.01 |
| glm-5.3-flash (hosted) | 2.84 | 4.67 | 2.50 |
| WhisperNow LFM2.5 350M | 9.41 | 15.48 | 6.83 |
| **WhisperNow MiniCPM5 1B** | **7.54** | **11.19** | **6.16** |
| WhisperNow LFM2.5 2.6B | 4.21 | 6.92 | 4.39 |

**On the test split, by length.** One lost fact in a 10-word dictation counts as 100 per 1,000 words, so short dictations have high rates.

| transcript words | rows | glm-5.3 max | glm-5.3-flash | 350M | **1B** | 2.6B |
|---|---|---|---|---|---|---|
| up to 25 | 1,426 | 6.77 | 8.79 | 7.81 | **8.92** | 5.99 |
| 25–200 | 1,335 | 1.69 | 2.97 | 5.71 | **5.60** | 2.57 |
| 200–450 | 292 | 1.76 | 2.47 | 9.59 | **6.86** | 3.70 |
| 450–1,200 | 240 | 1.18 | 2.30 | 11.51 | **8.77** | 5.16 |

**Other checks:**

| | 350M | **1B** | 2.6B |
|---|---|---|---|
| test rows that lose at least one fact | 26.1% | **25.7%** | 17.0% |
| bench-long rows that lose at least one fact | 93.7% | **90.1%** | 79.9% |
| invented facts per test row | 0.061 | **0.034** | 0.023 |
| test rows keeping every number, time, URL, path and code name exactly | 68.7% | **71.2%** | 75.6% |
| looping outputs, test / bench-long rows | 12 / 15 | **1 / 3** | 4 / 2 |

**Speed.** Seconds per dictation, Q8_0, one request at a time, median of 3 runs. GPU: RTX 5070 with CUDA. CPU: Ryzen 9 8945HS.

| words | 350M GPU | 1B GPU | 2.6B GPU | 350M CPU | 1B CPU | 2.6B CPU |
|---|---|---|---|---|---|---|
| 100 | 0.11 | **0.18** | 0.34 | 0.70 | **1.33** | 4.5 |
| 500 | 0.52 | **0.89** | 1.73 | 3.2 | **9.0** | 24.3 |
| 1,000 | 1.14 | **1.86** | 3.87 | 8.5 | **16.5** | 56.2 |

These times use WhisperNow's speculative decoding. A patched llama.cpp drafts up to 4 tokens at a time from a rule-based clean-up of the transcript. The model checks each drafted token and keeps only those it would write itself, apart from rare numerical near-ties. Stock llama.cpp without the patch took about 1.5 times as long for the 350M and 1B models.

## Training

- **Base model:** [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B), the instruct checkpoint.
- **Data:** the train split of the [WhisperNow cleanup dataset](https://huggingface.co/datasets/OWNER/whispernow-cleanup) (46,389 rows). Its val split (2,424 rows) gave the validation loss.
- **Recipe:** a full fine-tune for 3 epochs, with loss on the reply only and the input embeddings frozen. Learning rate 1e-5 with a cosine schedule and 3% warmup, no weight decay, 32 rows per step (2 rows × 16 accumulation steps), at most 2,816 tokens per row, paged 8-bit AdamW, Liger's fused loss, seed 109.
- **Hardware:** one RTX 5070 (12 GB), 11 hours 20 minutes.
- These are the weights at the end of epoch 3.

## Limitations

- **English only.**
- **It loses more facts on long dictations.** On bench-long (150–1,000 words) it loses at least one fact in 90% of dictations, against 63% for glm-5.3-flash. Read long outputs before you send them.
- **It sometimes changes a value.** It may reformat a number, garble code, swap a name or noun, or merge two list items.
- **It adds a layout more often than the 350M** where none is expected: on 11.0% of such bench-noise rows, against 6.5%.
- **It can loop on long lists**, repeating items: 3 of 416 bench-long rows.
- **Its inputs were styled like Parakeet, Whisper and Qwen3 ASR output**, plus unpunctuated text. Other recognizers' output should work, but it was not measured.
- **It does not follow instructions.** By design, it treats everything in the transcript as dictated text.
- **It was not trained on dictations over about 1,000 words.**

## License

The weights are under Apache-2.0, the base model's license. This model is a modified version of OpenBMB's MiniCPM5-1B: fine-tuning changed every weight except the input embeddings.

Every reference in the training data was written by OpenAI's `gpt-6.1-sol`, and some transcripts by Z.ai's glm-5.3. Both providers' terms restrict using their outputs to develop competing models. The [dataset card](https://huggingface.co/datasets/OWNER/whispernow-cleanup#licenses) has the details.
