"""
Utility functions for ASR model management.

Provides functions to discover installed/cached ASR models from the
HuggingFace Hub cache.
"""

from typing import List

from .backends import _NEMO_PREFIXES, _HUGGINGFACE_ASR_PREFIXES


def get_installed_asr_models() -> List[str]:
    """
    Get list of installed ASR models from the HuggingFace cache.
    
    Scans the HuggingFace Hub cache directory and returns model names
    that match known ASR model prefixes (NeMo and HuggingFace ASR models).
    
    Returns:
        List of model names (e.g., ["nvidia/parakeet-tdt-0.6b-v3", "openai/whisper-large-v3"])
    """
    try:
        from huggingface_hub import scan_cache_dir
        
        cache_info = scan_cache_dir()
        asr_models = []
        
        for repo in cache_info.repos:
            repo_id = repo.repo_id
            repo_lower = repo_id.lower()
            
            # Check if it matches NeMo ASR model prefixes
            is_nemo = any(
                repo_lower.startswith(prefix) or prefix in repo_lower
                for prefix in _NEMO_PREFIXES
            )
            
            # Check if it matches HuggingFace ASR model prefixes
            is_hf_asr = any(
                repo_lower.startswith(prefix)
                for prefix in _HUGGINGFACE_ASR_PREFIXES
            )
            
            if is_nemo or is_hf_asr:
                asr_models.append(repo_id)
        
        # Sort alphabetically for consistent ordering
        asr_models.sort()
        return asr_models
        
    except Exception:
        # If scanning fails, return empty list
        return []
