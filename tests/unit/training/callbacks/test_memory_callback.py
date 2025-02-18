import pytest
import torch
from unittest.mock import MagicMock, patch
from src.training.callbacks.memory import MemoryMonitorCallback

@pytest.fixture
def mock_trainer():
    """Mock trainer for testing."""
    trainer = MagicMock()
    trainer.state = MagicMock()
    trainer.state.current_step = 0
    trainer.state.max_steps = 100
    trainer.model = MagicMock()
    return trainer

def test_memory_callback_initialization():
    """Test memory callback initialization."""
    callback = MemoryMonitorCallback(
        warning_threshold=0.8,
        critical_threshold=0.9,
        logging_steps=10
    )
    assert callback.warning_threshold == 0.8
    assert callback.critical_threshold == 0.9
    assert callback.logging_steps == 10

def test_memory_thresholds_validation():
    """Test validation of memory thresholds."""
    with pytest.raises(ValueError):
        MemoryMonitorCallback(warning_threshold=0.9, critical_threshold=0.8)
    
    with pytest.raises(ValueError):
        MemoryMonitorCallback(warning_threshold=-0.1)
    
    with pytest.raises(ValueError):
        MemoryMonitorCallback(critical_threshold=1.1)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_memory_monitoring(mock_trainer):
    """Test memory monitoring functionality."""
    callback = MemoryMonitorCallback(logging_steps=1)
    callback.bind_trainer(mock_trainer)
    
    # Test monitoring on step begin
    with patch('torch.cuda.memory_allocated') as mock_allocated:
        with patch('torch.cuda.memory_reserved') as mock_reserved:
            mock_allocated.return_value = 1000
            mock_reserved.return_value = 2000
            callback.on_step_begin()
            
            assert hasattr(callback, '_last_memory_check')
            assert callback._last_memory_check['allocated'] == 1000
            assert callback._last_memory_check['reserved'] == 2000

def test_memory_warning_triggered(mock_trainer):
    """Test memory warning is triggered appropriately."""
    callback = MemoryMonitorCallback(
        warning_threshold=0.5,
        critical_threshold=0.9,
        logging_steps=1
    )
    callback.bind_trainer(mock_trainer)
    
    with patch('torch.cuda.memory_allocated') as mock_allocated:
        with patch('torch.cuda.max_memory_allocated') as mock_max_allocated:
            mock_allocated.return_value = 6000
            mock_max_allocated.return_value = 10000
            
            # Should trigger warning (60% usage)
            callback.on_step_begin()
            # Verify warning was logged (would need to check logs)

def test_memory_critical_triggered(mock_trainer):
    """Test critical memory threshold handling."""
    callback = MemoryMonitorCallback(
        warning_threshold=0.5,
        critical_threshold=0.8,
        logging_steps=1
    )
    callback.bind_trainer(mock_trainer)
    
    with patch('torch.cuda.memory_allocated') as mock_allocated:
        with patch('torch.cuda.max_memory_allocated') as mock_max_allocated:
            mock_allocated.return_value = 9000
            mock_max_allocated.return_value = 10000
            
            # Should trigger critical warning (90% usage)
            callback.on_step_begin()
            # Verify critical action was taken (e.g., garbage collection)

def test_logging_steps_respected(mock_trainer):
    """Test that logging only occurs at specified steps."""
    callback = MemoryMonitorCallback(logging_steps=10)
    callback.bind_trainer(mock_trainer)
    
    with patch('torch.cuda.memory_allocated') as mock_allocated:
        mock_allocated.return_value = 1000
        
        # Step 1 should not log
        mock_trainer.state.current_step = 1
        callback.on_step_begin()
        
        # Step 10 should log
        mock_trainer.state.current_step = 10
        callback.on_step_begin()
        
        # Verify logging occurred only at step 10

def test_memory_cleanup_on_exit(mock_trainer):
    """Test memory cleanup on training end."""
    callback = MemoryMonitorCallback()
    callback.bind_trainer(mock_trainer)
    
    with patch('torch.cuda.empty_cache') as mock_empty_cache:
        callback.on_training_end()
        mock_empty_cache.assert_called_once()

def test_gpu_not_available():
    """Test callback behavior when GPU is not available."""
    with patch('torch.cuda.is_available', return_value=False):
        callback = MemoryMonitorCallback()
        # Should not raise error but log warning about GPU unavailability 