"""Tests for vocabulary processor."""

import pytest

from src.transcribe.core.transcript_processor.vocabulary_processor import (
    apply_vocabulary_replacements,
)


class TestApplyVocabularyReplacements:

    def test_empty_replacements(self):
        text = "Hello world"
        result = apply_vocabulary_replacements(text, [])
        assert result == "Hello world"

    def test_single_replacement(self):
        text = "Hello Charlie"
        replacements = [("Charlie", "Charles")]
        result = apply_vocabulary_replacements(text, replacements)
        assert result == "Hello Charles"

    def test_multiple_occurrences(self):
        text = "Charlie said hello to Charlie"
        replacements = [("Charlie", "Charles")]
        result = apply_vocabulary_replacements(text, replacements)
        assert result == "Charles said hello to Charles"

    def test_multiple_replacements(self):
        text = "Hello Charlie, link calendar please"
        replacements = [
            ("Charlie", "Charles"),
            ("link calendar", "https://example.com"),
        ]
        result = apply_vocabulary_replacements(text, replacements)
        assert result == "Hello Charles, https://example.com please"

    def test_case_sensitive_by_default(self):
        text = "charlie CHARLIE Charlie"
        replacements = [("Charlie", "Charles")]
        result = apply_vocabulary_replacements(text, replacements)
        # Only exact case match should be replaced
        assert result == "charlie CHARLIE Charles"

    def test_case_insensitive_when_disabled(self):
        text = "charlie CHARLIE Charlie"
        replacements = [("Charlie", "Charles")]
        result = apply_vocabulary_replacements(text, replacements, case_sensitive=False)
        # All variations should be replaced
        assert result == "Charles Charles Charles"

    def test_empty_original_skipped(self):
        text = "Hello world"
        replacements = [("", "NOTHING"), ("world", "universe")]
        result = apply_vocabulary_replacements(text, replacements)
        assert result == "Hello universe"

    def test_empty_replacement_removes_word(self):
        text = "Hello um world"
        replacements = [("um ", "")]
        result = apply_vocabulary_replacements(text, replacements)
        assert result == "Hello world"

    def test_phrase_replacement(self):
        text = "Please send the document to the main office"
        replacements = [("the main office", "HQ")]
        result = apply_vocabulary_replacements(text, replacements)
        assert result == "Please send the document to HQ"

    def test_no_change_returns_original(self):
        text = "Hello world"
        replacements = [("foo", "bar")]
        result = apply_vocabulary_replacements(text, replacements)
        assert result == "Hello world"

    def test_special_regex_chars_in_original(self):
        text = "Use (regex) for [matching]"
        replacements = [("(regex)", "patterns"), ("[matching]", "selection")]
        result = apply_vocabulary_replacements(text, replacements)
        assert result == "Use patterns for selection"
