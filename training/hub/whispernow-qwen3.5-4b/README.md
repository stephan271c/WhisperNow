---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B-Base
base_model_relation: finetune
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- dictation
- transcript-cleanup
- speech-to-text
- qwen3.5
- gguf
datasets:
- category271/whispernow-cleanup
---

# WhisperNow Qwen3.5 4B

This model turns a speech recognizer's transcript into the text the speaker meant to write. It removes fillers, false starts and repeats, keeps only the corrected version when the speaker corrects themselves, fixes punctuation, and lays out a list when the speaker dictates one. It is a full fine-tune of [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base), made for the local cleanup step of [WhisperNow](https://github.com/stephan271c/WhisperNow), a desktop dictation app.

It is the largest and most accurate of three cleanup models trained the same way. Over the test split and both benchmarks it loses 0.83 times the facts of the hosted glm-5.3-flash (90% interval 0.77–0.90) and 0.63 times those of the LFM2.5 2.6B (0.59–0.67). On dictations of up to 25 words it loses fewer than glm-5.3 at max reasoning; from 25 words on, glm-5.3 at max reasoning loses fewer.

| model | Q8_0 file | memory in use | facts lost per 1,000 words | 100 / 1,000 words, GPU | 100 / 1,000 words, CPU |
|---|---|---|---|---|---|
| [LFM2.5 1.2B](https://huggingface.co/category271/whispernow-lfm2.5-1.2b) | 1.25 GB | **TODO** | 5.88 | **TODO** / **TODO** s | **TODO** / **TODO** s |
| [LFM2.5 2.6B](https://huggingface.co/category271/whispernow-lfm2.5-2.6b) | 2.87 GB | **TODO** | 4.60 | **TODO** / **TODO** s | **TODO** / **TODO** s |
| **Qwen3.5 4B** (this model) | 4.48 GB | **TODO** | 2.88 | **TODO** / **TODO** s | **TODO** / **TODO** s |

Facts lost are over the test split and both benchmarks together; [Evaluation](#evaluation) explains the numbers.

## Examples

Two transcripts from the test split, which all three models clean exactly as the reference does.

A dictation with several self-corrections:

```text
alright so the hides need to soak in the lime bath for uh three days no actually five days for these thicker buffalo hides and the ph should be around 11 no wait 12.5 to properly loosen the hair and giuliana said to add an extra 20 no 30 pounds of hydrated lime to this batch because the last one came out underprocessed
```

```text
The hides need to soak in the lime bath for 5 days for these thicker buffalo hides, and the pH should be around 12.5 to properly loosen the hair. Giuliana said to add an extra 30 pounds of hydrated lime to this batch because the last one came out underprocessed.
```

A list inside a message. The request after it is dictated text, so it stays in the output:

```text
for the farm run tomorrow we're picking up fence posts feed for the chickens and uh that new hose nozzle also text me when you're leaving because i need to move the truck
```

```text
For the farm run tomorrow, we're picking up:
- Fence posts
- Feed for the chickens
- That new hose nozzle

Also, text me when you're leaving because I need to move the truck.
```

## What it does

The model gets no instructions, only the transcript. It learned the cleanup rules that WhisperNow sends to hosted models (the full prompt is in the [dataset card](https://huggingface.co/datasets/category271/whispernow-cleanup#the-cleanup-rules)):

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

There is no system prompt. Send one user turn that holds only the transcript, with leading and trailing whitespace trimmed. The chat template (`chat_template.jinja`, also embedded in the GGUF file) renders it as:

```text
<|endoftext|><|im_start|>user
{transcript}<|im_end|>
<|im_start|>assistant
```

The template rejects a system turn.

### Settings

Every score below was measured with:

- greedy decoding (temperature 0);
- a repeat penalty of 1.05 over the whole prompt and reply (llama.cpp: `repeat_penalty` 1.05, `repeat_last_n` 4096);
- a reply budget of 2.5 tokens per transcript word, and at least 64.

Training rows were at most 3,072 tokens of prompt and reply together, which is about 1,000 words of dictation.

### llama.cpp

```bash
llama-server -m model-Q8_0.gguf --jinja -c 8192
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

repo = "category271/whispernow-qwen3.5-4b"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="bfloat16", device_map="auto")

transcript = "um can you send me the report by friday no wait thursday".strip()
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": transcript}],
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
reply = model.generate(
    **inputs,
    do_sample=False,
    repetition_penalty=1.05,
    max_new_tokens=max(64, int(2.5 * len(transcript.split()))),
)
print(tokenizer.decode(reply[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The weights load as `Qwen3_5ForCausalLM`, the text model alone, which needs a transformers release with Qwen3.5 support.

The scores below come from llama.cpp at Q8_0. Transformers in BF16 can pick a different word where two are almost equally likely.

## Files

| file | what it is |
|---|---|
| `model.safetensors`, `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | BF16 weights for transformers |
| `model-Q8_0.gguf` | 4.48 GB. The scores below use this file, and WhisperNow downloads it. |

## Evaluation

**Test sets.**

- **test:** 3,297 held-out rows of the [training dataset](https://huggingface.co/datasets/category271/whispernow-cleanup), from 1 to about 1,000 words.
- **bench-long:** 416 dictations of 151–946 words, 49 of them real meeting turns from the AMI corpus.
- **bench-noise:** 461 dictations of 20–64 words, each with one kind of disfluency (restarts, repeats, filler clusters, corrections and others), plus a control set. For 136 of them the input is what Parakeet wrote when Kokoro spoke the transcript.

The benchmarks were generated with their own prompts, names and seeds, so no benchmark row shares a template or seed with a training row.

**Scoring.** A judge model (OpenAI's `gpt-6.1-sol` at high reasoning) compares each output with the reference and lists its faults. **Facts lost** counts facts that were dropped, changed, or put on the wrong side of a correction, per 1,000 reference words. Lower is better. Our models ran at Q8_0 in llama.cpp with the settings above; the hosted glm-5.3 models received WhisperNow's full cleanup prompt. Rates are over the rows every model has a verdict for: 3,295 test rows, 414 bench-long rows and all 461 bench-noise rows (4,170 in all).

**Facts lost per 1,000 reference words:**

| model | test | bench-long | bench-noise | all three |
|---|---|---|---|---|
| glm-5.3 at max reasoning (hosted) | 1.73 | 2.14 | 2.01 | 1.88 |
| glm-5.3-flash (hosted) | 2.87 | 4.68 | 2.50 | 3.46 |
| WhisperNow LFM2.5 1.2B | 4.53 | 8.58 | 4.27 | 5.88 |
| WhisperNow LFM2.5 2.6B | 3.54 | 6.65 | 3.90 | 4.60 |
| **WhisperNow Qwen3.5 4B** | **2.10** | **4.37** | **2.62** | **2.88** |

**On the test split, by length.** One lost fact in a 10-word dictation counts as 100 per 1,000 words, so short dictations have high rates.

| transcript words | rows | glm-5.3 max | glm-5.3-flash | 1.2B | 2.6B | **4B** |
|---|---|---|---|---|---|---|
| up to 25 | 1,426 | 6.98 | 9.46 | 5.94 | 5.48 | **4.57** |
| 25–200 | 1,335 | 1.68 | 2.97 | 3.70 | 2.83 | **1.95** |
| 200–450 | 293 | 1.77 | 2.47 | 4.48 | 3.78 | **1.84** |
| 450–1,200 | 241 | 1.18 | 2.29 | 4.84 | 3.60 | **2.04** |

**Other checks:**

|  | 1.2B | 2.6B | **4B** | glm-5.3-flash |
|---|---|---|---|---|
| test rows that lose at least one fact | 19.2% | 16.7% | **12.6%** | 17.0% |
| bench-long rows that lose at least one fact | 85.3% | 79.2% | **68.6%** | 63.0% |
| invented facts per test row | 0.029 | 0.018 | **0.010** | 0.019 |
| invented facts per bench-long row | 0.263 | 0.181 | **0.072** | 0.118 |
| test rows keeping every number, URL, path and code name exactly | 75.6% | 76.3% | **79.0%** | 56.2% |
| bench-long rows keeping every number, URL, path and code name exactly | 26.5% | 33.9% | **35.9%** | 11.4% |
| looping outputs, test / bench-long rows | 4 / 5 | 2 / 3 | **2 / 1** | 1 / 1 |

**Speed.** Seconds per dictation, Q8_0, one request at a time.

**TODO: speed and memory table**

These times use WhisperNow's speculative decoding: llama.cpp drafts tokens from a rule-based clean-up of the transcript, and the model keeps only the drafted tokens it would write itself, apart from rare numerical near-ties.

## Training

- **Base model:** [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base). Only the text model is loaded, without the vision tower or the multi-token-prediction block: 4.2B weights. Qwen3.5 has no beginning-of-sequence token, so prompts start with the base model's end-of-document token, `<|endoftext|>`.
- **Data:** the train split of the [WhisperNow Cleanup dataset](https://huggingface.co/datasets/category271/whispernow-cleanup) (60,836 rows). A fixed 600 rows of its val split gave the validation loss.
- **Recipe:** a full fine-tune for 3 epochs, with loss on the reply only. Learning rate 1e-5 with a cosine schedule and 3% warmup, no weight decay, 32 rows per step, at most 3,072 tokens per row, fp32 master weights under bf16 autocast, fused AdamW, gradient checkpointing, TRL's chunked loss, seed 109. Peak GPU memory was 113 GB.
- **Hardware:** one H200 141 GB on Modal; the run paused at step 1,800 of 5,706 and resumed from its checkpoint, and the resumed part took 2.7 hours.
- These are the weights at the end of epoch 3. Validation loss was lower at epoch 2 (0.065 against 0.079).

## Limitations

- **Speed depends on the hardware.** **TODO: CPU seconds for 100 and 1,000 words.**
- **English only.**
- **It loses more facts on long dictations.** On bench-long (151–946 words) it loses at least one fact in 68.6% of dictations, against 63.0% for glm-5.3-flash. Read long outputs before you send them.
- **It can change or invent a value**: a number, a name, a code token, or a list item merged with another. It invents a fact in about 0.072 per bench-long dictation.
- **It can loop on long lists**, repeating items: the loop check flagged 2 test and 1 bench-long outputs. WhisperNow caps the reply at 2.5 tokens per transcript word and keeps the rule-based clean-up when a reply hits the cap.
- **Its inputs were styled like Parakeet, Whisper and Qwen3-ASR output**, plus unpunctuated text. Other recognizers' output should work, but it was not measured.
- **It does not follow instructions.** By design, it treats everything in the transcript as dictated text.
- **It was not trained on dictations over about 1,000 words.**

## License

The weights are under the Apache License 2.0, the base model's license. This model is a modified version of Qwen's Qwen3.5-4B-Base: fine-tuning changed every weight.

Every reference in the training data was written by OpenAI's `gpt-6.1-sol`, and some transcripts by Z.ai's GLM models. Both providers' terms restrict using their outputs to develop competing models. The [dataset card](https://huggingface.co/datasets/category271/whispernow-cleanup#licenses-and-terms) has the details.
