"""Vocabulary replacement processor."""
from typing import List, Tuple
import re

from ...utils.logger import get_logger

logger = get_logger(__name__)


def apply_vocabulary_replacements(
    text: str,
    replacements: List[Tuple[str, str]],
    case_sensitive: bool = True
) -> str:
    """
    Apply vocabulary replacements to text.
    
    Replaces all occurrences of 'original' with 'replacement' for each rule.
    Processes rules in order they were defined.
    
    Args:
        text: The input transcription text
        replacements: List of (original, replacement) tuples
        case_sensitive: Whether to match case-sensitively (default True)
        
    Returns:
        Text with all replacements applied
    """
    if not replacements:
        return text
    
    result = text
    for original, replacement in replacements:
        if not original:  # Skip empty originals
            continue
        
        if case_sensitive:
            result = result.replace(original, replacement)
        else:
            # Case-insensitive replacement
            result = re.sub(re.escape(original), replacement, result, flags=re.IGNORECASE)
    
    if result != text:
        logger.debug(f"Applied vocabulary replacements: '{text[:50]}...' -> '{result[:50]}...'")
    
    return result
