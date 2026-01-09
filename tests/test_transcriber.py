"""
Tests for TranscriptionEngine and backend detection.

Uses mocking to avoid large model downloads.
"""

from unittest.mock import patch, MagicMock

import pytest

from src.transcribe.core.asr.backends import BackendType, detect_backend_type
from src.transcribe.core.asr.transcriber import TranscriptionEngine, EngineState


class TestBackendDetection:
    """Tests for automatic backend type detection."""
    
    def test_detect_nemo_nvidia_model(self):
        """Test NVIDIA models detected as NeMo backend."""
        assert detect_backend_type("nvidia/parakeet-tdt-0.6b-v3") == BackendType.NEMO
        assert detect_backend_type("nvidia/canary-1b") == BackendType.NEMO
    
    def test_detect_huggingface_whisper(self):
        """Test Whisper models detected as HuggingFace backend."""
        assert detect_backend_type("openai/whisper-base") == BackendType.HUGGINGFACE
        assert detect_backend_type("openai/whisper-large-v3") == BackendType.HUGGINGFACE
    
    def test_detect_huggingface_wav2vec(self):
        """Test Wav2Vec2 models detected as HuggingFace backend."""
        assert detect_backend_type("facebook/wav2vec2-large-960h") == BackendType.HUGGINGFACE
    
    def test_unknown_model_defaults_to_huggingface(self):
        """Test unknown model names default to HuggingFace."""
        result = detect_backend_type("unknown/custom-model")
        assert result == BackendType.HUGGINGFACE


class TestEngineState:
    """Tests for engine state machine."""
    
    def test_initial_state_not_loaded(self):
        """Test engine starts in NOT_LOADED state."""
        engine = TranscriptionEngine(model_name="test/model")
        assert engine.state == EngineState.NOT_LOADED
    
    def test_is_ready_false_initially(self):
        """Test is_ready is False before loading."""
        engine = TranscriptionEngine(model_name="test/model")
        assert engine.is_ready is False


class TestEngineCallbacks:
    """Tests for engine state change callbacks."""
    
    def test_state_change_callback(self):
        """Test on_state_change callback is called."""
        states = []
        
        def on_state(state, msg):
            states.append((state, msg))
        
        engine = TranscriptionEngine(
            model_name="test/model",
            on_state_change=on_state
        )
        
        # Manually trigger state change
        engine._set_state(EngineState.LOADING, "Loading model...")
        
        assert len(states) == 1
        assert states[0][0] == EngineState.LOADING
        assert states[0][1] == "Loading model..."


class TestEngineConfiguration:
    """Tests for engine configuration."""
    
    def test_model_name_stored(self):
        """Test model name is stored correctly."""
        engine = TranscriptionEngine(model_name="nvidia/parakeet-tdt-0.6b-v3")
        assert engine.model_name == "nvidia/parakeet-tdt-0.6b-v3"
    
    def test_use_gpu_default_true(self):
        """Test GPU is enabled by default."""
        engine = TranscriptionEngine(model_name="test/model")
        assert engine.use_gpu is True
    
    def test_use_gpu_can_be_disabled(self):
        """Test GPU can be disabled."""
        engine = TranscriptionEngine(model_name="test/model", use_gpu=False)
        assert engine.use_gpu is False
    
    def test_backend_type_auto_resolves(self):
        """Test AUTO backend type resolves to actual backend."""
        engine = TranscriptionEngine(model_name="nvidia/parakeet-tdt-0.6b-v3")
        # AUTO resolves to NEMO for nvidia models
        assert engine.backend_type == BackendType.NEMO


class TestEngineUnload:
    """Tests for model unloading."""
    
    def test_unload_without_load(self):
        """Test unload() is safe without prior load."""
        engine = TranscriptionEngine(model_name="test/model")
        # Should not raise
        engine.unload()
        assert engine.state == EngineState.NOT_LOADED


class TestTranscribeChunked:
    """Tests for chunked transcription orchestration."""
    
    @patch('src.transcribe.core.asr.transcriber.needs_chunking')
    def test_transcribe_chunked_orchestration(self, mock_needs_chunking):
        """Test proper orchestration of splitting, transcribing, and combining."""
        # Setup mocks
        mock_needs_chunking.return_value = True
        
        engine = TranscriptionEngine()
        engine._audio_processor = MagicMock()
        
        # Mock split to return 2 chunks
        chunk1 = MagicMock()
        chunk2 = MagicMock()
        engine._audio_processor.split_audio.return_value = [chunk1, chunk2]
        
        # Mock internal transcribe to return text for chunks
        engine.transcribe = MagicMock(side_effect=["Part1", "Part2"])
        
        # Mock combine
        engine._audio_processor.combine_transcriptions.return_value = "Part1 Part2"
        
        # Execute
        result = engine.transcribe_chunked(MagicMock(), 16000)
        
        # Verify
        assert result == "Part1 Part2"
        assert engine.transcribe.call_count == 2
        engine._audio_processor.split_audio.assert_called_once()
        engine._audio_processor.combine_transcriptions.assert_called_once_with(["Part1", "Part2"])


# Integration tests that require actual models
# These are marked with pytest.mark.slow and skipped by default
@pytest.mark.slow
class TestEngineIntegration:
    """Integration tests requiring real model downloads.
    
    Run with: pytest -m slow
    """
    
    def test_load_real_model(self):
        """Test loading a real NeMo model."""
        pytest.skip("Requires ~5GB model download - run manually")
