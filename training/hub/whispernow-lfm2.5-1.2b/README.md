---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-1.2B-Base
base_model_relation: finetune
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- dictation
- transcript-cleanup
- speech-to-text
- lfm2.5
- gguf
datasets:
- category271/whispernow-cleanup
---

# WhisperNow LFM2.5 1.2B

This model turns a speech recognizer's transcript into the text the speaker meant to write. It removes fillers, false starts and repeats, keeps only the corrected version when the speaker corrects themselves, fixes punctuation, and lays out a list when the speaker dictates one. It is a full fine-tune of [LiquidAI/LFM2.5-1.2B-Base](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base), made for the local cleanup step of [WhisperNow](https://github.com/stephan271c/WhisperNow), a desktop dictation app.

It is the smallest and fastest of three cleanup models trained the same way, and the one to pick for a CPU. Over the test split and both benchmarks it loses 1.70 times as many facts as the hosted glm-5.3-flash (90% interval 1.58–1.82). On dictations of up to 25 words it loses fewer than glm-5.3-flash and is level with glm-5.3 at max reasoning.

| model | Q8_0 file | memory in use | facts lost per 1,000 words | 100 / 1,000 words, GPU | 100 / 1,000 words, CPU |
|---|---|---|---|---|---|
| **LFM2.5 1.2B** (this model) | 1.25 GB | 1.6 GB | 5.88 | 0.18 / 2.0 s | 2.3 / 27 s |
| [LFM2.5 2.6B](https://huggingface.co/category271/whispernow-lfm2.5-2.6b) | 2.87 GB | 3.3 GB | 4.60 | 0.34 / 3.8 s | 5.4 / 65 s |
| [Qwen3.5 4B](https://huggingface.co/category271/whispernow-qwen3.5-4b) | 4.48 GB | 5.3 GB | 2.88 | 0.62 / 7.2 s | 8.9 / 118 s |

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
<|startoftext|><|im_start|>user
{transcript}<|im_end|>
<|im_start|>assistant
```

The template rejects a system turn.

### Settings

Every score below was measured with:

- greedy decoding (temperature 0);
- a repeat penalty of 1.05 over the whole prompt and reply (llama.cpp: `repeat_penalty` 1.05, `repeat_last_n` 4096);
- a reply budget of 2.5 tokens per transcript word, and at least 64.

Training rows were at most 2,816 tokens of prompt and reply together, which is about 1,000 words of dictation.

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

repo = "category271/whispernow-lfm2.5-1.2b"
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

The scores below come from llama.cpp at Q8_0. Transformers in BF16 can pick a different word where two are almost equally likely.

## Files

| file | what it is |
|---|---|
| `model.safetensors`, `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | BF16 weights for transformers |
| `model-Q8_0.gguf` | 1.25 GB. The scores below use this file, and WhisperNow downloads it. |

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
| **WhisperNow LFM2.5 1.2B** | **4.53** | **8.58** | **4.27** | **5.88** |
| WhisperNow LFM2.5 2.6B | 3.54 | 6.65 | 3.90 | 4.60 |
| WhisperNow Qwen3.5 4B | 2.10 | 4.37 | 2.62 | 2.88 |

**By transcript length.** All three sets together. One lost fact in a 10-word dictation counts as 100 per 1,000 words, so short dictations have high rates.

![Facts lost per 1,000 reference words by transcript length, for the three WhisperNow models and the two hosted glm-5.3 models](charts/facts-by-length.png)

| transcript words | rows | glm-5.3 max | glm-5.3-flash | **1.2B** | 2.6B | 4B |
|---|---|---|---|---|---|---|
| up to 25 | 1,443 | 7.16 | 9.34 | **6.01** | 5.50 | 4.60 |
| 26–50 | 1,057 | 2.71 | 3.83 | **5.10** | 4.01 | 2.95 |
| 51–100 | 465 | 1.47 | 2.26 | **3.33** | 3.05 | 1.90 |
| 101–200 | 289 | 1.06 | 2.76 | **3.82** | 2.82 | 1.67 |
| 201–450 | 484 | 2.34 | 3.35 | **6.30** | 5.06 | 2.98 |
| 451 and more | 432 | 1.39 | 3.33 | **6.36** | 4.83 | 3.00 |

**Against the hosted models.** Each point is a WhisperNow model's facts lost divided by a hosted model's, on the same rows, with a 90% paired bootstrap interval. Below 1×, the local model loses fewer.

![Facts lost by each WhisperNow model divided by glm-5.3-flash's and by glm-5.3 at max reasoning's, by transcript length](charts/against-hosted.png)

Up to 25 words, it loses 0.64 times the facts glm-5.3-flash loses (90% interval 0.52–0.80) and is level with glm-5.3 at max reasoning (0.84, 0.66–1.06). Above 25 words it loses more than either: 1.3–1.9 times as many facts as glm-5.3-flash, and 1.9–4.6 times as many as glm-5.3 at max reasoning.

**Other checks:**

|  | **1.2B** | 2.6B | 4B | glm-5.3-flash |
|---|---|---|---|---|
| test rows that lose at least one fact | **19.2%** | 16.7% | 12.6% | 17.0% |
| bench-long rows that lose at least one fact | **85.3%** | 79.2% | 68.6% | 63.0% |
| invented facts per test row | **0.029** | 0.018 | 0.010 | 0.019 |
| invented facts per bench-long row | **0.263** | 0.181 | 0.072 | 0.118 |
| test rows keeping every number, URL, path and code name exactly | **75.6%** | 76.3% | 79.0% | 56.2% |
| bench-long rows keeping every number, URL, path and code name exactly | **26.5%** | 33.9% | 35.9% | 11.4% |
| looping outputs, test / bench-long rows | **4 / 5** | 2 / 3 | 2 / 1 | 1 / 1 |

**Other models.** Seven more hosted models received WhisperNow's full cleanup prompt on an 800-row sample of the test split and both benchmarks (1,670 rows, judged the same way). Rates are higher than above because bench-long is a larger share of these rows.

![Facts lost per 1,000 reference words for every model evaluated, sorted](charts/all-models.png)

![Facts lost per 1,000 reference words for every model evaluated, by transcript length](charts/all-models-by-length.png)

On these rows only glm-5.3 at max reasoning loses fewer facts than the Qwen3.5 4B, and Claude Haiku 5.5 and glm-5.3-flash come next. The LFM2.5 2.6B loses fewer than six of the nine hosted models, and the 1.2B fewer than four. The fastest hosted model, Nemotron 3.5 Lightning, loses 1.8 times the facts of the Qwen3.5 4B.

**Speed.** Seconds for the whole reply, median of 5 runs, one request at a time. Ours ran at Q8_0 in llama.cpp; the hosted models were called over the internet from the same machine with WhisperNow's full prompt.

![Seconds per reply by transcript length on a GPU and a CPU, against the range of hosted models](charts/latency.png)

| transcript words | **1.2B GPU** | 2.6B GPU | 4B GPU | **1.2B CPU** | 2.6B CPU | 4B CPU | glm-5.3-flash | glm-5.3 max | fastest hosted |
|---|---|---|---|---|---|---|---|---|---|
| 16 | **0.03** | 0.06 | 0.15 | **0.29** | 0.66 | 1.5 | 1.6 | 2.8 | 0.35 (Gemma 4 26B-A4B) |
| 100 | **0.18** | 0.34 | 0.62 | **2.3** | 5.4 | 8.9 | 2.6 | 5.6 | 0.65 (Nemotron 3.5 Lightning) |
| 199 | **0.79** | 1.3 | 2.3 | **8.9** | 21 | 34 | 5.7 | 32 | 0.99 (Nemotron 3.5 Lightning) |
| 299 | **0.70** | 1.4 | 2.6 | **9.6** | 23 | 42 | 6.8 | 44 | 1.3 (Nemotron 3.5 Lightning) |
| 500 | **0.79** | 1.5 | 2.8 | **11** | 26 | 47 | 15 | 86 | 1.7 (Nemotron 3.5 Lightning) |
| 1,005 | **2.0** | 3.8 | 7.2 | **27** | 65 | 118 | 19 | 99 | 2.9 (Nemotron 3.5 Lightning) |

On an RTX 5070 it replies in 0.18 s for 100 words and 2.0 s for 1,000, 7.3–19 times faster than glm-5.3-flash over the internet (2.6 s and 19 s); glm-5.3 at max reasoning took 5.6 s and 99 s. On a Ryzen 9 8945HS CPU it takes 2.3 s for 100 words and 27 s for 1,000: from 100 words up, between 1.3 times faster and 1.5 times slower than glm-5.3-flash. For long dictations without a GPU, a hosted model answers sooner.

These times use WhisperNow's speculative decoding: llama.cpp drafts tokens from a rule-based clean-up of the transcript, and the model keeps only the drafted tokens it would write itself, apart from rare numerical near-ties.

## Training

- **Base model:** [LiquidAI/LFM2.5-1.2B-Base](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base). The base checkpoint rather than the post-trained one, whose chat template is a reasoning model's and opens a `<think>` block.
- **Data:** the train split of the [WhisperNow Cleanup dataset](https://huggingface.co/datasets/category271/whispernow-cleanup) (60,836 rows). A fixed 600 rows of its val split gave the validation loss.
- **Recipe:** a full fine-tune for 3 epochs, with loss on the reply only. Learning rate 1e-5 with a cosine schedule and 3% warmup, no weight decay, 32 rows per step, at most 2,816 tokens per row, fp32 master weights under bf16 autocast, fused AdamW, gradient checkpointing, TRL's chunked loss, seed 109. Peak GPU memory was 43 GB.
- **Hardware:** one H100 80 GB on Modal, 58 minutes.
- These are the weights at the end of epoch 3. Validation loss was lower at epoch 2 (0.092 against 0.106).

## Limitations

- **Speed depends on the hardware.** On a CPU it takes 2.3 seconds for 100 words and 27 seconds for 1,000.
- **English only.**
- **It loses more facts on long dictations.** On bench-long (151–946 words) it loses at least one fact in 85.3% of dictations, against 63.0% for glm-5.3-flash. Read long outputs before you send them.
- **It can change or invent a value**: a number, a name, a code token, or a list item merged with another. It invents a fact in about 0.263 per bench-long dictation.
- **It can loop on long lists**, repeating items: the loop check flagged 4 test and 5 bench-long outputs. WhisperNow caps the reply at 2.5 tokens per transcript word and keeps the rule-based clean-up when a reply hits the cap.
- **Its inputs were styled like Parakeet, Whisper and Qwen3-ASR output**, plus unpunctuated text. Other recognizers' output should work, but it was not measured.
- **It does not follow instructions.** By design, it treats everything in the transcript as dictated text.
- **It was not trained on dictations over about 1,000 words.**

## License

The weights are under the LFM Open License v1.0, the base model's license, included as `LICENSE`. That license allows commercial use only by organizations with annual revenue under $10 million (Section 5). This model is a modified version of Liquid AI's LFM2.5-1.2B-Base: fine-tuning changed every weight.

Every reference in the training data was written by OpenAI's `gpt-6.1-sol`, and some transcripts by Z.ai's GLM models. Both providers' terms restrict using their outputs to develop competing models. The [dataset card](https://huggingface.co/datasets/category271/whispernow-cleanup#licenses-and-terms) has the details.
