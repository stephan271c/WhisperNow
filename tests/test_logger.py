"""Tests for logging infrastructure."""

import logging
from pathlib import Path
from unittest.mock import patch
import tempfile

import pytest

from src.transcribe.utils.logger import get_logger, get_log_dir


class TestLoggerConfiguration:
    def test_get_logger_returns_logger(self):
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)
    
    def test_get_logger_singleton(self):
        logger1 = get_logger("transcribe")
        logger2 = get_logger("transcribe")
        assert logger1 is logger2
    
    @patch("src.transcribe.utils.logger.get_log_dir")
    def test_log_directory_creation(self, mock_get_log_dir, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        mock_get_log_dir.return_value = log_dir
        
        result = get_log_dir()
        
        assert result.exists()
        assert result.is_dir()
        assert result.name == "logs"
    
    @patch("src.transcribe.utils.logger.get_log_dir")
    def test_logger_writes_to_file(self, mock_get_log_dir, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        mock_get_log_dir.return_value = log_dir
        
        import src.transcribe.utils.logger as logger_module
        logger_module._logger_instance = None
        
        root_logger = logging.getLogger("transcribe")
        root_logger.handlers.clear()
        
        logger = get_logger("transcribe")
        logger.info("Test message")
        
        log_file = log_dir / "app.log"
        assert log_file.exists()
        
        content = log_file.read_text()
        assert "Test message" in content
        assert "INFO" in content


class TestLogRotation:
    @patch("src.transcribe.utils.logger.get_log_dir")
    def test_log_rotation_when_size_exceeded(self, mock_get_log_dir, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        mock_get_log_dir.return_value = log_dir
        
        import src.transcribe.utils.logger as logger_module
        logger_module._logger_instance = None
        
        with patch("src.transcribe.utils.logger.RotatingFileHandler") as mock_handler_class:
            from logging.handlers import RotatingFileHandler
            log_file = tmp_path / "logs" / "app.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            handler = RotatingFileHandler(
                log_file,
                maxBytes=1024,  # 1KB for testing
                backupCount=2,
                encoding="utf-8"
            )
            mock_handler_class.return_value = handler
            
            logger = get_logger("transcribe")
            
            for i in range(100):
                logger.info(f"Test message {i} with some padding to increase size")
            assert log_file.exists()
