---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
base_model: nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16
base_model_relation: finetune
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- dictation
- transcript-cleanup
- speech-to-text
- nemotron
- gguf
datasets:
- category271/whispernow-cleanup
---

# WhisperNow Nemotron Bolt E4B

This model cleans up an ASR transcript by removing fillers, false starts and repeats, applying the speaker's self-corrections, and adjusting formatting. It is a fine-tune of [nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16), made for the local cleanup step of [WhisperNow](https://github.com/stephan271c/WhisperNow), a desktop dictation app.

| model | Q8_0 file | memory in use | factual errors per 1,000 words |
|---|---|---|---|
| [LFM2.5 1.2B](https://huggingface.co/category271/whispernow-lfm2.5-1.2b) | 1.25 GB | 1.6 GB | 6.32 |
| [LFM3 1.7B](https://huggingface.co/category271/whispernow-lfm3-1.7b) | 1.85 GB | 2.2 GB | 5.04 |
| [LFM2.5 2.6B](https://huggingface.co/category271/whispernow-lfm2.5-2.6b) | 2.87 GB | 3.3 GB | 4.88 |
| [Qwen3.5 4B](https://huggingface.co/category271/whispernow-qwen3.5-4b) | 4.48 GB | 5.3 GB | 3.01 |
| Nemotron Bolt E4B (this model) | 7.97 GB | 8.4 GB | 2.59 |

## Examples

Two transcripts from the test split, which this model cleans exactly as the reference does.

A dictation with several self-corrections:

```text
alright so the hides need to soak in the lime bath for uh three days no actually five days for these thicker buffalo hides and the ph should be around 11 no wait 12.5 to properly loosen the hair and giuliana said to add an extra 20 no 30 pounds of hydrated lime to this batch because the last one came out underprocessed
```

```text
The hides need to soak in the lime bath for 5 days for these thicker buffalo hides, and the pH should be around 12.5 to properly loosen the hair. Giuliana said to add an extra 30 pounds of hydrated lime to this batch because the last one came out underprocessed.
```

A list inside a message, then a request that stays in the output:

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

It learned the cleanup rules WhisperNow sends to cloud models ([full prompt](https://huggingface.co/datasets/category271/whispernow-cleanup#the-cleanup-rules)):

- It removes fillers and verbal tics ("um", "like", "you know", "so" opening a sentence), stutters, repeated words and false starts.
- When the speaker clearly corrects themselves, it keeps only the corrected version and drops the cue ("no wait", "scratch that").
- It fixes grammar, spelling, punctuation and capitalization, but keeps the speaker's words, tone, hedges ("maybe", "I think") and reasons ("because ...").
- It lays out shopping lists, to-do lists, steps, recipes and meeting notes as lists; messages, stories and explanations stay as sentences. It uses no bold, italics, headings or em dashes.
- It writes amounts, counts, dates, times and percentages in digits, and turns spoken punctuation and symbols ("dash dash help") into the real thing. Code, paths and URLs stay exactly as dictated.
- A question or command in the transcript is cleaned up and returned as text, never answered or carried out.
- When the transcript holds no dictated words (only fillers or noise), it returns `EMPTY`.

## How to use

Send the trimmed transcript as the only user turn, with no system prompt. The chat template (`chat_template.jinja`, also in the GGUF file) does the rest.

### Settings

Every score below was measured with:

- greedy decoding (temperature 0);
- a repeat penalty of 1.05 over the whole prompt and reply (llama.cpp: `repeat_penalty` 1.05, `repeat_last_n` 4096);
- a reply limit of 2.5 tokens per transcript word, and at least 64.

### llama.cpp

```bash
llama-server -hf category271/whispernow-nemotron-bolt-e4b:Q8_0 --jinja -c 8192
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

The weights are saved under the multimodal model's key names. Load them with `key_mapping` as below; `AutoModelForCausalLM` without it initializes the model at random.

```python
from transformers import AutoTokenizer, Gemma4ForCausalLM

repo = "category271/whispernow-nemotron-bolt-e4b"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = Gemma4ForCausalLM.from_pretrained(
    repo, dtype="bfloat16", device_map="auto", key_mapping={r"^model\.language_model\.": "model."}
)

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
    temperature=None,
    top_k=None,
    top_p=None,
    repetition_penalty=1.05,
    max_new_tokens=max(64, int(2.5 * len(transcript.split()))),
)
print(tokenizer.decode(reply[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Evaluation

**Test sets.**

- **test:** 3,297 held-out rows of the [training dataset](https://huggingface.co/datasets/category271/whispernow-cleanup), from 1 to about 1,000 words.
- **bench-long:** 416 dictations of 151–946 words, 49 of them real meeting turns from the AMI corpus.
- **bench-noise:** 461 dictations of 20–64 words, each with one kind of disfluency (restarts, repeats, filler clusters, corrections and others), plus a control set.

Scores cover 4,170 rows: 3,295 of test, 414 of bench-long and all 461 of bench-noise.

**Scoring.** A judge model, OpenAI's `gpt-6.1-sol` at high reasoning, compares each output with the reference. **Factual errors** are facts dropped, changed, put on the wrong side of a correction, or invented, per 1,000 reference words. The cloud models received WhisperNow's full cleanup prompt.

The tables and the first charts below cover the three models WhisperNow lists first. The LFM2.5 2.6B and Qwen3.5 4B have the same tables on their own cards.

**Factual errors per 1,000 reference words:**

| model | test | bench-long | bench-noise | all three |
|---|---|---|---|---|
| glm-5.3 at max reasoning (cloud) | 1.88 | 2.30 | 2.07 | 2.03 |
| glm-5.3-flash at low reasoning (cloud) | 3.08 | 4.98 | 2.80 | 3.71 |
| WhisperNow LFM2.5 1.2B | 4.84 | 9.25 | 4.69 | 6.32 |
| WhisperNow LFM3 1.7B | 3.88 | 7.24 | 4.88 | 5.04 |
| WhisperNow Nemotron Bolt E4B (this model) | 1.95 | 3.77 | 2.80 | 2.59 |

**By transcript length:**

![Factual errors per 1,000 reference words by transcript length, for LFM2.5 1.2B, LFM3 1.7B, Nemotron Bolt E4B and the two cloud glm-5.3 models](charts/errors-by-length.png)

| transcript words | rows | glm-5.3 (max) | glm-5.3-flash (low) | 1.2B | LFM3 | Bolt |
|---|---|---|---|---|---|---|
| up to 25 | 1,443 | 7.67 | 10.55 | 6.14 | 5.44 | 4.41 |
| 26–50 | 1,057 | 2.80 | 4.13 | 5.39 | 4.45 | 2.74 |
| 51–100 | 465 | 2.04 | 2.44 | 3.69 | 3.41 | 1.97 |
| 101–200 | 289 | 1.17 | 2.93 | 4.32 | 3.37 | 1.42 |
| 201–450 | 484 | 2.52 | 3.63 | 6.83 | 5.49 | 2.84 |
| 451 and more | 432 | 1.46 | 3.51 | 6.79 | 5.31 | 2.58 |

**Against the cloud models.** Each point is a WhisperNow model's factual errors divided by a cloud model's, with a 90% paired bootstrap interval.

![Factual errors of LFM2.5 1.2B, LFM3 1.7B and Nemotron Bolt E4B divided by glm-5.3-flash's and by glm-5.3 at max reasoning's, by transcript length](charts/against-cloud.png)

**Other checks:**

|  | 1.2B | LFM3 | Bolt | glm-5.3-flash (low) | glm-5.3 (max) |
|---|---|---|---|---|---|
| dictations with at least one factual error | 25.8% | 24.1% | 17.2% | 21.8% | 15.9% |
| invented facts per 1,000 words (counted in factual errors) | 0.44 | 0.29 | 0.15 | 0.25 | 0.15 |
| self-corrections left in, per 1,000 words | 0.45 | 0.36 | 0.28 | 2.44 | 1.32 |
| dictations keeping every number, URL and path exactly | 70.3% | 70.4% | 74.9% | 51.9% | 56.4% |
| dictations with a filler left in | 1.1% | 1.3% | 0.9% | 6.7% | 1.8% |
| plain text given a list or paragraph breaks it shouldn't have | 2.7% | 3.0% | 2.4% | 5.0% | 4.3% |
| a list or paragraphs run together into one block | 4.9% | 4.5% | 4.1% | 7.9% | 5.8% |
| counts written as words instead of digits | 0.1% | 0.1% | 0.1% | 9.3% | 6.2% |
| replies that answer or act on the dictation instead of cleaning it | 0 | 1 | 1 | 8 | 3 |
| replies that loop | 9 | 5 | 4 | 2 | 2 |

The last two rows are counts. A self-correction left in is one the output keeps as spoken ("Tuesday, no wait, Wednesday"). Nearly all factual errors are lost facts: invented facts are 4–7% of them for our models and both glm-5.3 models.

**Other models.** Seven more cloud models were scored on an 800-row sample of the test split plus both benchmarks (1,670 rows). Rates are higher than above because bench-long is a larger share of these rows. Invented facts are a larger share of the errors only for Granite 4.2 8B (16%) and Claude Haiku 5.5 (12%).

![Factual errors per 1,000 reference words for every model evaluated, sorted](charts/all-models.png)

The same rates by transcript length, sorted by the overall rate:

![Factual errors per 1,000 reference words for every model evaluated, by transcript length](charts/all-models-by-length.png)

**Speed.** Seconds for the whole reply, the median of 5 runs on an RTX 5070 GPU and of 3 on a Ryzen 9 8945HS CPU. Ours ran at Q8_0 in llama.cpp b12048 with WhisperNow's speculative decoding, which drafts tokens from a rule-based cleanup of the transcript. The cloud models were called through their providers' APIs.

![Seconds per reply by transcript length on a GPU and a CPU, against the range of cloud models](charts/latency.png)

| transcript words | 1.2B GPU | LFM3 GPU | Bolt GPU | 1.2B CPU | LFM3 CPU | Bolt CPU | glm-5.3-flash (low) | glm-5.3 (max) |
|---|---|---|---|---|---|---|---|---|
| 16 | 0.03 | 0.04 | 0.10 | 0.26 | 0.36 | 1.1 | 1.6 | 2.8 |
| 100 | 0.17 | 0.25 | 0.62 | 2.1 | 2.6 | 8.3 | 2.6 | 5.6 |
| 199 | 0.64 | 0.76 | 2.0 | 7.2 | 8.3 | 25 | 5.7 | 32 |
| 299 | 0.66 | 0.94 | 2.8 | 7.7 | 11 | 36 | 6.8 | 44 |
| 500 | 0.75 | 1.1 | 2.8 | 10 | 15 | 36 | 15 | 86 |
| 1,005 | 1.8 | 2.7 | 7.9 | 21 | 34 | 96 | 19 | 99 |

## Training

- **Base model:** [nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16). Only the text model is loaded: 7.5B weights, of which 2.8B are per-layer embedding tables kept frozen in bf16. The other 4.7B train.
- **Data:** the train split of the [WhisperNow Cleanup dataset](https://huggingface.co/datasets/category271/whispernow-cleanup) (60,836 rows).
- **Recipe:** a fine-tune for 3 epochs, with loss on the reply only. AdamW at learning rate 1e-5 with a cosine schedule and 3% warmup, no weight decay, 32 rows per step, at most 3,072 tokens per row, bf16 mixed precision.
- **Hardware:** one H200 141 GB on Modal.
- **Code:** the [training code](https://github.com/stephan271c/WhisperNow/tree/master/training) is in the WhisperNow repository. This model's settings are in [`configs/nemotron-bolt-e4b.toml`](https://github.com/stephan271c/WhisperNow/blob/master/training/configs/nemotron-bolt-e4b.toml).

## Limitations

- **English only.**
- **It makes more factual errors on long dictations.** On bench-long (151–946 words) it makes at least one in 66.4% of dictations, against 64.7% for glm-5.3-flash at low reasoning.
- **It is no better than the Qwen3.5 4B on dense disfluencies.** On bench-noise it makes 2.80 factual errors per 1,000 words, against 2.62 for the 4B.
- **It was not trained on dictations over about 1,000 words.**

## License

The weights are under the [OpenMDW License Agreement, version 1.1](https://openmdw.ai/license/1-1/), included as `LICENSE`. The base model is NVIDIA's post-training of Google's Gemma 4 E4B, which is under the Apache License 2.0, included as `LICENSE-Apache-2.0`.
