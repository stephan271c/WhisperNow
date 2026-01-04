"""
LLM processing for text enhancement.

Uses LiteLLM to apply configurable prompts to transcribed text.
"""

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import os

import litellm
from litellm import completion
from litellm.exceptions import BadRequestError

from ..utils.logger import get_logger

logger = get_logger(__name__)


# Provider configurations
# Format: provider_id -> (display_name, default_api_base, env_var_name)
PROVIDERS: Dict[str, tuple] = {
    "openai": ("OpenAI", None, "OPENAI_API_KEY"),
    "anthropic": ("Anthropic", None, "ANTHROPIC_API_KEY"),
    "openrouter": ("OpenRouter", None, "OPENROUTER_API_KEY"),
    "ollama": ("Ollama (Local)", "http://localhost:11434", None),
    "gemini": ("Google Gemini", None, "GEMINI_API_KEY"),
    "other": ("Other", None, None),
}


def get_models_for_provider(provider: str) -> List[str]:
    """
    Get available models for a provider from litellm.model_cost.
    
    Uses fast prefix-based filtering on the local model_cost dict
    (no network requests, no slow get_llm_provider calls).
    The dropdown is editable so users can type any model name.
    """
    if provider == "other":
        return []  # User will type custom model name
    
    try:
        all_models = list(litellm.model_cost.keys())
        
        # Filter by model name patterns (fast, no API calls)
        if provider == "openai":
            # OpenAI models: gpt-*, o1-*, o3-*, chatgpt-*
            filtered = [m for m in all_models 
                       if m.startswith(('gpt-', 'o1-', 'o3-', 'chatgpt-'))
                       and '/' not in m]  # Exclude provider-prefixed versions
        elif provider == "anthropic":
            # Anthropic: claude-* (not provider-prefixed)
            filtered = [m for m in all_models 
                       if m.startswith('claude-') and '/' not in m]
        elif provider == "gemini":
            # Gemini: gemini/* or gemini-*
            filtered = [m for m in all_models 
                       if m.startswith('gemini/') or 
                       (m.startswith('gemini-') and '/' not in m)]
        elif provider == "ollama":
            # Ollama: ollama/*
            filtered = [m for m in all_models if m.startswith('ollama/')]
        elif provider == "openrouter":
            # OpenRouter: openrouter/*
            filtered = [m for m in all_models if m.startswith('openrouter/')]
        else:
            filtered = []
        
        # Sort and limit to keep UI responsive
        result = sorted(filtered)[:100]
        return result if result else _get_fallback_models(provider)
        
    except Exception as e:
        logger.warning(f"Failed to get models for provider {provider}: {e}")
        return _get_fallback_models(provider)


def _get_fallback_models(provider: str) -> List[str]:
    """Fallback curated models if litellm.model_cost filtering fails."""
    models = {
        "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "o1-preview", "o1-mini"],
        "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
        "openrouter": ["openrouter/auto", "openrouter/anthropic/claude-3.5-sonnet", "openrouter/openai/gpt-4o"],
        "ollama": ["ollama/llama3.2", "ollama/mistral", "ollama/codellama", "ollama/phi3"],
        "gemini": ["gemini/gemini-2.0-flash", "gemini/gemini-1.5-flash", "gemini/gemini-1.5-pro"],
    }
    return models.get(provider, [])


@dataclass
class Enhancement:
    """A named prompt template for enhancing transcribed text."""
    id: str
    title: str
    prompt: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Enhancement":
        """Create from dictionary."""
        return cls(**data)


class LLMProcessor:
    """
    Processes text through an LLM using enhancement prompts.
    
    Uses LiteLLM for provider-agnostic API access.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None
    ):
        """
        Initialize the LLM processor.
        
        Args:
            model: The LLM model to use (e.g., "gpt-4o-mini", "ollama/llama3")
            api_key: Optional API key. If not provided, uses environment variables.
            api_base: Optional API base URL (required for Ollama, OpenRouter, etc.)
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        
        # State to track if we need to merge system prompts into user messages
        # (for models that reject developer instructions/system prompts)
        self._force_merged_prompt = False
        
        logger.info(f"LLMProcessor initialized with model: {model}, api_base: {api_base}")
    
    def process(self, text: str, enhancement: Enhancement) -> str:
        """
        Apply an enhancement prompt to the given text.
        
        Args:
            text: The raw transcribed text to enhance
            enhancement: The enhancement containing the prompt to apply
            
        Returns:
            The enhanced text from the LLM, or original text if processing fails
        """
        if not text or not text.strip():
            return text
        
        logger.info(f"Applying enhancement '{enhancement.title}' to text ({len(text)} chars)")
        
        try:
            # Build completion kwargs
            kwargs = {
                "model": self.model,
            }
            
            # Use cached fallback strategy if we know this model needs it
            if self._force_merged_prompt:
                merged_content = f"{enhancement.prompt}\n\n{text}"
                kwargs["messages"] = [{"role": "user", "content": merged_content}]
            else:
                kwargs["messages"] = [
                    {"role": "system", "content": enhancement.prompt},
                    {"role": "user", "content": text}
                ]
            
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.api_base:
                kwargs["api_base"] = self.api_base
            
            response = completion(**kwargs)
            
            result = response.choices[0].message.content
            logger.info(f"Enhancement complete: {len(text)} -> {len(result)} chars")
            return result
            
        except BadRequestError as e:
            # Handle models that don't support system prompts (like some Google/OpenRouter models)
            err_str = str(e)
            if "Developer instruction is not enabled" in err_str or "unsupported_country_region_territory" in err_str:
                # Log the specific error for debugging
                logger.warning(f"Model {self.model} rejected system prompt or raised 400. Retrying with merged prompt. Error: {e}")
                
                try:
                    # Merge system prompt into user message
                    merged_content = f"{enhancement.prompt}\n\n{text}"
                    kwargs["messages"] = [{"role": "user", "content": merged_content}]
                    
                    response = completion(**kwargs)
                    result = response.choices[0].message.content
                    logger.info(f"Enhancement complete (fallback): {len(text)} -> {len(result)} chars")
                    
                    # Cache this success so we don't error next time
                    self._force_merged_prompt = True
                    return result
                except Exception as retry_err:
                    logger.error(f"Fallback processing failed: {retry_err}", exc_info=True)
                    return text
            
            # Re-raise other bad requests
            logger.error(f"LLM processing failed with BadRequest: {e}", exc_info=True)
            return text

        except Exception as e:
            logger.error(f"LLM processing failed: {e}", exc_info=True)
            # Return original text on failure
            return text
    
    def is_configured(self) -> bool:
        """
        Check if the LLM processor has valid configuration.
        
        Returns True if API key is set (either directly or via environment),
        or if using a provider that doesn't require auth (like local Ollama).
        """
        # Check for direct API key
        if self.api_key:
            return True
        
        # Ollama (local) doesn't need API key
        if self.model.startswith("ollama/"):
            return True
        
        # Check common environment variables
        env_vars = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY", 
            "AZURE_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
        ]
        return any(os.environ.get(var) for var in env_vars)


# Default enhancement presets
DEFAULT_ENHANCEMENTS: List[Enhancement] = [
    Enhancement(
        id="fix_grammar",
        title="Fix Grammar & Spelling",
        prompt="Fix any grammar or spelling errors in the following text. "
               "Preserve the original meaning and tone. "
               "Only output the corrected text, nothing else."
    ),
    Enhancement(
        id="professional",
        title="Make Professional",
        prompt="Rewrite the following text in a professional, formal tone "
               "suitable for business communication. "
               "Preserve the original meaning. "
               "Only output the rewritten text, nothing else."
    ),
    Enhancement(
        id="concise",
        title="Make Concise",
        prompt="Rewrite the following text to be more concise and clear. "
               "Remove filler words and unnecessary phrases. "
               "Only output the rewritten text, nothing else."
    ),
]
