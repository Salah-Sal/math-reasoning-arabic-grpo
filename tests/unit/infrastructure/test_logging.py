import pytest
import logging
from pathlib import Path
from src.infrastructure.logging import setup_logging, get_logger

@pytest.fixture
def log_file():
    """Create and clean up log file"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "training.log"
    yield log_file
    # Cleanup
    if log_file.exists():
        log_file.unlink()
    if log_dir.exists():
        log_dir.rmdir()

def test_logger_creation(log_file):
    """Test basic logger creation and functionality"""
    setup_logging()
    logger = get_logger("test")
    
    test_message = "Test log message"
    logger.info(test_message)
    
    assert log_file.exists()
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert test_message in log_content

def test_logger_levels(log_file):
    """Test different logging levels"""
    setup_logging()
    logger = get_logger("test_levels")
    
    debug_msg = "Debug message"
    info_msg = "Info message"
    
    logger.debug(debug_msg)
    logger.info(info_msg)
    
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert debug_msg in log_content  # Debug messages should be in file
        assert info_msg in log_content   # Info messages should be in file

def test_invalid_config():
    """Test fallback to basic configuration with invalid config"""
    setup_logging(Path("nonexistent.yaml"))
    logger = get_logger("test_invalid")
    
    # Should not raise an exception
    logger.info("Test message with invalid config") 