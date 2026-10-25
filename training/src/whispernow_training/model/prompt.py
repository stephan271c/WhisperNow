"""The local cleanup model's request: one user turn holding only the transcript.

`serving/chat_template.jinja` turns the messages into the model's text (Gemma 4 models use
`chat_template_gemma4.jinja`). Training renders with it and `llama-server --chat-template-file`
serves with it. `serving/prompt_golden.json` pins the messages and the ChatML text; the Rust
builder (`build_local_cleanup_prompt` in whispernow-core) is tested against the same file.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from ..datagen.paths import TRAINING_ROOT

SERVING = TRAINING_ROOT / "serving"
GOLDEN_FILE = SERVING / "prompt_golden.json"

# LFM2.5's; `render` takes another model's from its tokenizer.
BOS = "<|startoftext|>"


@dataclass(frozen=True)
class ChatFormat:
    template_file: Path
    end_of_turn: str

    def template(self) -> str:
        return self.template_file.read_text(encoding="utf-8")


CHATML = ChatFormat(SERVING / "chat_template.jinja", "<|im_end|>")
GEMMA4 = ChatFormat(SERVING / "chat_template_gemma4.jinja", "<turn|>")


def chat_format(model_type: str) -> ChatFormat:
    """Gemma 4's turn markers for Gemma 4 models, whose vocabulary lacks ChatML's; ChatML for the rest."""
    return GEMMA4 if model_type.startswith("gemma4") else CHATML


def model_chat_format(model_dir: Path) -> ChatFormat:
    return chat_format(json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["model_type"])

# Decoding the app sends with each request: greedy, a light repeat penalty, and a reply budget
# that grows with the transcript. 2.5 tokens per input word covers every row in the dataset.
# The penalty looks back over the whole prompt and reply, as transformers' `repetition_penalty`
# does; llama.cpp's default window is the last 64 tokens, which writes more spoken numbers as
# words and keeps fewer protected tokens.
TEMPERATURE = 0.0
REPEAT_PENALTY = 1.05
REPEAT_LAST_N = 4096
MAX_TOKENS_PER_WORD = 2.5
MIN_MAX_TOKENS = 64

Message = dict[str, str]


def messages(transcript: str) -> list[Message]:
    """What the app sends: the trimmed transcript as the only message."""
    return [{"role": "user", "content": transcript.strip()}]


def max_tokens(transcript: str) -> int:
    return max(MIN_MAX_TOKENS, math.ceil(MAX_TOKENS_PER_WORD * len(transcript.split())))


def _raise(message: str) -> None:
    raise TemplateError(message)


def render(
    conversation: list[Message], *, add_generation_prompt: bool = True, bos_token: str = BOS, chat: ChatFormat = CHATML
) -> str:
    """Renders like transformers' `apply_chat_template`, with the same Jinja settings."""
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    environment.globals["raise_exception"] = _raise
    template = environment.from_string(chat.template())
    return template.render(messages=conversation, add_generation_prompt=add_generation_prompt, bos_token=bos_token)


def prompt_completion(transcript: str, output: str) -> dict[str, list[Message]]:
    """A training row in TRL's conversational prompt-completion shape."""
    return {"prompt": messages(transcript), "completion": [{"role": "assistant", "content": output}]}


def golden() -> dict:
    return json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))


def golden_cases() -> list[dict]:
    return golden()["cases"]


def decoding() -> dict[str, float | int]:
    return {"temperature": TEMPERATURE, "repeat_penalty": REPEAT_PENALTY, "repeat_last_n": REPEAT_LAST_N}
