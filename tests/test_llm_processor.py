"""
Tests for LLM processor.

Verifies Enhancement dataclass and LLMProcessor functionality.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.transcribe.core.llm_processor import (
    Enhancement,
    LLMProcessor,
    LLMResponse,
    DEFAULT_ENHANCEMENTS,
)


class TestEnhancement:
    """Tests for Enhancement dataclass."""
    
    def test_creation(self):
        """Test basic enhancement creation."""
        enh = Enhancement(
            id="test_id",
            title="Test Enhancement",
            prompt="Fix the text"
        )
        assert enh.id == "test_id"
        assert enh.title == "Test Enhancement"
        assert enh.prompt == "Fix the text"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        enh = Enhancement(
            id="test_id",
            title="Test Enhancement",
            prompt="Fix the text"
        )
        data = enh.to_dict()
        assert data == {
            "id": "test_id",
            "title": "Test Enhancement",
            "prompt": "Fix the text"
        }
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "id": "test_id",
            "title": "Test Enhancement",
            "prompt": "Fix the text"
        }
        enh = Enhancement.from_dict(data)
        assert enh.id == "test_id"
        assert enh.title == "Test Enhancement"
        assert enh.prompt == "Fix the text"
    
    def test_roundtrip(self):
        """Test dict -> Enhancement -> dict roundtrip."""
        original = {
            "id": "test_id",
            "title": "Test Enhancement",
            "prompt": "Fix the text"
        }
        enh = Enhancement.from_dict(original)
        result = enh.to_dict()
        assert result == original


class TestDefaultEnhancements:
    """Tests for default enhancement presets."""
    
    def test_defaults_exist(self):
        """Test that default enhancements are defined."""
        assert len(DEFAULT_ENHANCEMENTS) >= 3
    
    def test_defaults_have_required_fields(self):
        """Test that default enhancements have required fields."""
        for enh in DEFAULT_ENHANCEMENTS:
            assert enh.id
            assert enh.title
            assert enh.prompt


class TestLLMProcessor:
    """Tests for LLMProcessor class."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = LLMProcessor(model="gpt-4o-mini", api_key="test-key")
        assert processor.model == "gpt-4o-mini"
        assert processor.api_key == "test-key"
    
    def test_initialization_defaults(self):
        """Test processor with default values."""
        processor = LLMProcessor()
        assert processor.model == "gpt-4o-mini"
        assert processor.api_key is None
    
    def test_is_configured_with_api_key(self):
        """Test is_configured returns True when API key is set."""
        processor = LLMProcessor(api_key="test-key")
        assert processor.is_configured() is True
    
    def test_is_configured_without_api_key(self):
        """Test is_configured returns False when no API key."""
        processor = LLMProcessor()
        # Clear any environment variables
        with patch.dict('os.environ', {}, clear=True):
            assert processor.is_configured() is False
    
    def test_is_configured_with_env_var(self):
        """Test is_configured returns True when env var is set."""
        processor = LLMProcessor()
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'env-key'}):
            assert processor.is_configured() is True
    
    @patch('src.transcribe.core.llm_processor.completion')
    def test_process_calls_completion(self, mock_completion):
        """Test that process calls litellm completion and returns LLMResponse."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Fixed text"
        mock_response.usage = None
        mock_completion.return_value = mock_response
        
        processor = LLMProcessor(model="gpt-4o-mini", api_key="test-key")
        enhancement = Enhancement(
            id="test",
            title="Test",
            prompt="Fix the text"
        )
        
        result = processor.process("Original text", enhancement)
        
        # Result is now LLMResponse object
        assert isinstance(result, LLMResponse)
        assert result.content == "Fixed text"
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        assert call_args.kwargs['model'] == "gpt-4o-mini"
        assert call_args.kwargs['api_key'] == "test-key"
        assert len(call_args.kwargs['messages']) == 2
        assert call_args.kwargs['messages'][0]['role'] == 'system'
        assert call_args.kwargs['messages'][0]['content'] == "Fix the text"
        assert call_args.kwargs['messages'][1]['role'] == 'user'
        assert call_args.kwargs['messages'][1]['content'] == "Original text"
    
    @patch('src.transcribe.core.llm_processor.completion')
    def test_process_returns_original_on_error(self, mock_completion):
        """Test that process returns original text in LLMResponse on error."""
        mock_completion.side_effect = Exception("API error")
        
        processor = LLMProcessor(model="gpt-4o-mini", api_key="test-key")
        enhancement = Enhancement(id="test", title="Test", prompt="Fix")
        
        result = processor.process("Original text", enhancement)
        
        assert isinstance(result, LLMResponse)
        assert result.content == "Original text"
    
    def test_process_empty_text(self):
        """Test that process returns empty text unchanged in LLMResponse."""
        processor = LLMProcessor(model="gpt-4o-mini", api_key="test-key")
        enhancement = Enhancement(id="test", title="Test", prompt="Fix")
        
        result = processor.process("", enhancement)
        assert isinstance(result, LLMResponse)
        assert result.content == ""
        
        result2 = processor.process("   ", enhancement)
        assert result2.content == "   "
