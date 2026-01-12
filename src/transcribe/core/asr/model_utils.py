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
            
            is_nemo = any(
                repo_lower.startswith(prefix) or prefix in repo_lower
                for prefix in _NEMO_PREFIXES
            )
            
            is_hf_asr = any(
                repo_lower.startswith(prefix)
                for prefix in _HUGGINGFACE_ASR_PREFIXES
            )
            
            if is_nemo or is_hf_asr:
                asr_models.append(repo_id)
        
        asr_models.sort()
        return asr_models
        
    except Exception:
        return []


def delete_asr_model(model_name: str) -> tuple[bool, str]:
    """
    Delete an ASR model from the HuggingFace cache.
    
    Args:
        model_name: The model name to delete (e.g., "nvidia/parakeet-tdt-0.6b-v3")
        
    Returns:
        Tuple of (success: bool, message: str) where message contains
        freed space on success or error description on failure.
    """
    try:
        from huggingface_hub import scan_cache_dir
        
        cache_info = scan_cache_dir()
        
        target_repo = None
        for repo in cache_info.repos:
            if repo.repo_id == model_name:
                target_repo = repo
                break
        
        if target_repo is None:
            return False, f"Model '{model_name}' not found in cache"
        
        revision_hashes = [rev.commit_hash for rev in target_repo.revisions]
        
        if not revision_hashes:
            return False, f"No revisions found for model '{model_name}'"
        
        delete_strategy = cache_info.delete_revisions(*revision_hashes)
        
        freed_size_str = delete_strategy.expected_freed_size_str
        
        delete_strategy.execute()
        
        return True, f"Deleted '{model_name}', freed {freed_size_str}"
        
    except Exception as e:
        return False, f"Failed to delete model: {str(e)}"
