import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import litellm
from litellm import completion, completion_cost
from pydantic import BaseModel, ConfigDict

from ...utils.logger import get_logger

logger = get_logger(__name__)


PROVIDERS: Dict[str, tuple] = {
    "openai": ("OpenAI", None, "OPENAI_API_KEY"),
    "anthropic": ("Anthropic", None, "ANTHROPIC_API_KEY"),
    "openrouter": ("OpenRouter", None, "OPENROUTER_API_KEY"),
    "ollama": ("Ollama (Local)", "http://localhost:11434", None),
    "gemini": ("Google Gemini", None, "GEMINI_API_KEY"),
    "other": ("Other", None, None),
}


def get_models_for_provider(provider: str) -> List[str]:
    """Get models for provider. Dropdown is editable so users can type custom names."""
    if provider == "other":
        return []  # User will type custom model name

    try:
        all_models = list(litellm.model_cost.keys())

        if provider == "openai":
            # OpenAI models: gpt-*, o1-*, o3-*, chatgpt-*
            filtered = [
                m
                for m in all_models
                if m.startswith(("gpt-", "o1-", "o3-", "chatgpt-")) and "/" not in m
            ]
        elif provider == "anthropic":
            filtered = [
                m for m in all_models if m.startswith("claude-") and "/" not in m
            ]
        elif provider == "gemini":
            filtered = [
                m
                for m in all_models
                if m.startswith("gemini/") or (m.startswith("gemini-") and "/" not in m)
            ]
        elif provider == "ollama":
            filtered = [m for m in all_models if m.startswith("ollama/")]
        elif provider == "openrouter":
            filtered = [m for m in all_models if m.startswith("openrouter/")]
        else:
            filtered = []

        result = sorted(filtered)[:100]
        return result if result else _get_fallback_models(provider)

    except Exception as e:
        logger.warning(f"Failed to get models for provider {provider}: {e}")
        return _get_fallback_models(provider)


from typing import List


def _get_fallback_models(provider: str) -> List[str]:
    models = {
        "openai": [
            "gpt-5.2",
            "gpt-5-nano",
            "gpt-4o",
            "o3",
            "o4-mini",
        ],
        "anthropic": [
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-5-20251101",
            "claude-haiku-4-5-20251001",
            "claude-3-5-sonnet-20241022",
        ],
        "openrouter": [
            "openrouter/auto",
            "openrouter/openai/gpt-5.2",
            "openrouter/anthropic/claude-sonnet-4.5",
            "openrouter/google/gemini-3-flash",
        ],
        "ollama": [
            "ollama/llama3.3",
            "ollama/gemma3",
            "ollama/mistral",
            "ollama/phi4",
        ],
        "gemini": [
            "gemini/gemini-3-flash",
            "gemini/gemini-3-pro",
            "gemini/gemini-2.5-flash",
        ],
    }
    return models.get(provider, [])


class Enhancement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    title: str
    prompt: str

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "Enhancement":
        return cls.model_validate(data)


@dataclass
class LLMResponse:
    content: str
    cost_usd: Optional[float] = None
    usage: Optional[dict] = (
        None  # token counts: prompt_tokens, completion_tokens, total_tokens
    )


class LLMProcessor:

    @staticmethod
    def format_model_name(model: str, provider: str) -> str:
        known_prefixes = (
            "openrouter/",
            "ollama/",
            "gemini/",
            "openai/",
            "anthropic/",
            "azure/",
            "huggingface/",
        )

        if model.startswith(known_prefixes):
            return model

        prefix_map = {
            "openrouter": "openrouter/",
            "ollama": "ollama/",
            "gemini": "gemini/",
        }

        prefix = prefix_map.get(provider)
        if prefix:
            return f"{prefix}{model}"

        return model

    def __init__(
        self,
        model: str = "gpt-5-nano",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base

        model_info = litellm.model_cost.get(model, {})
        self._supports_system_messages = model_info.get(
            "supports_system_messages", True
        )

        if not self._supports_system_messages:
            logger.info(
                f"Model {model} does not support system messages (per model_cost)"
            )

        logger.info(
            f"LLMProcessor initialized with model: {model}, api_base: {api_base}"
        )

    def process(self, text: str, enhancement: Enhancement) -> LLMResponse:
        if not text or not text.strip():
            return LLMResponse(content=text)

        logger.info(
            f"Applying enhancement '{enhancement.title}' to text ({len(text)} chars)"
        )

        try:
            if self._supports_system_messages:
                messages = [
                    {"role": "system", "content": enhancement.prompt},
                    {"role": "user", "content": text},
                ]
            else:
                merged_content = f"{enhancement.prompt}\n\n{text}"
                messages = [{"role": "user", "content": merged_content}]
                logger.debug(f"Merged system prompt with user prompt for {self.model}")

            kwargs = {
                "model": self.model,
                "messages": messages,
            }

            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.api_base:
                kwargs["api_base"] = self.api_base

            response = completion(**kwargs)

            result_text = response.choices[0].message.content

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Pydantic serializer warnings",
                        category=UserWarning,
                    )
                    cost = completion_cost(completion_response=response)
            except Exception:
                cost = None

            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            logger.info(
                f"Enhancement complete: {len(text)} -> {len(result_text)} chars, cost=${cost:.6f}"
                if cost
                else f"Enhancement complete: {len(text)} -> {len(result_text)} chars"
            )

            return LLMResponse(content=result_text, cost_usd=cost, usage=usage)

        except Exception as e:
            logger.error(f"LLM processing failed: {e}", exc_info=True)
            return LLMResponse(content=text)

    def is_configured(self) -> bool:
        if self.api_key:
            return True

        if self.model.startswith("ollama/"):
            return True

        env_vars = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "AZURE_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
        ]
        return any(os.environ.get(var) for var in env_vars)


def load_default_enhancements() -> List[Enhancement]:
    import json
    from pathlib import Path

    json_path = Path(__file__).parent / "enhancement_prompts.json"

    with open(json_path, "r") as f:
        data = json.load(f)
    return [Enhancement.model_validate(item) for item in data]


_default_enhancements: Optional[List[Enhancement]] = None


def get_default_enhancements() -> List[Enhancement]:
    global _default_enhancements
    if _default_enhancements is None:
        _default_enhancements = load_default_enhancements()
    return _default_enhancements


DEFAULT_ENHANCEMENTS = get_default_enhancements()
