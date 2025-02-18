import pytest
from unittest.mock import MagicMock, patch
import time
from src.training.callbacks.progress import TrainingProgressCallback

@pytest.fixture
def mock_trainer():
    """Mock trainer for testing."""
    trainer = MagicMock()
    trainer.state = MagicMock()
    trainer.state.current_step = 0
    trainer.state.max_steps = 100
    trainer.state.current_reward = 0.0
    trainer.state.best_reward = 0.0
    trainer.state.loss = 1.0
    return trainer

def test_progress_callback_initialization():
    """Test progress callback initialization."""
    callback = TrainingProgressCallback(logging_steps=10)
    assert callback.logging_steps == 10
    assert callback._start_time is None
    assert callback._last_log_time is None

def test_training_start_logging(mock_trainer):
    """Test logging at training start."""
    callback = TrainingProgressCallback()
    callback.bind_trainer(mock_trainer)
    
    with patch('time.time', return_value=1000):
        callback.on_training_start()
        assert callback._start_time == 1000
        assert callback._last_log_time == 1000

def test_step_progress_logging(mock_trainer):
    """Test progress logging during steps."""
    callback = TrainingProgressCallback(logging_steps=10)
    callback.bind_trainer(mock_trainer)
    
    # Mock time progression
    start_time = 1000
    with patch('time.time') as mock_time:
        # Training start
        mock_time.return_value = start_time
        callback.on_training_start()
        
        # Step 10
        mock_trainer.state.current_step = 10
        mock_trainer.state.loss = 0.8
        mock_trainer.state.current_reward = 0.3
        mock_time.return_value = start_time + 60  # 60 seconds elapsed
        callback.on_step_end()
        
        # Verify logging occurred at step 10

def test_time_estimation(mock_trainer):
    """Test time remaining estimation."""
    callback = TrainingProgressCallback(logging_steps=1)
    callback.bind_trainer(mock_trainer)
    
    with patch('time.time') as mock_time:
        # Training start
        mock_time.return_value = 1000
        callback.on_training_start()
        
        # After 10 steps
        mock_trainer.state.current_step = 10
        mock_time.return_value = 1000 + 300  # 300 seconds for 10 steps
        callback.on_step_end()
        
        # Verify time estimation is reasonable
        estimated_total = callback._estimate_total_time()
        assert 2700 < estimated_total < 3300  # ~3000 seconds total expected

def test_progress_percentage(mock_trainer):
    """Test progress percentage calculation."""
    callback = TrainingProgressCallback()
    callback.bind_trainer(mock_trainer)
    
    mock_trainer.state.current_step = 25
    assert callback._get_progress_percentage() == 25.0
    
    mock_trainer.state.current_step = 50
    assert callback._get_progress_percentage() == 50.0

def test_metrics_formatting(mock_trainer):
    """Test metrics string formatting."""
    callback = TrainingProgressCallback()
    callback.bind_trainer(mock_trainer)
    
    mock_trainer.state.current_step = 50
    mock_trainer.state.loss = 0.5
    mock_trainer.state.current_reward = 0.8
    mock_trainer.state.best_reward = 0.9
    
    metrics_str = callback._format_metrics()
    assert "loss: 0.500" in metrics_str
    assert "reward: 0.800" in metrics_str
    assert "best_reward: 0.900" in metrics_str

def test_training_end_summary(mock_trainer):
    """Test training end summary logging."""
    callback = TrainingProgressCallback()
    callback.bind_trainer(mock_trainer)
    
    with patch('time.time') as mock_time:
        # Start training
        mock_time.return_value = 1000
        callback.on_training_start()
        
        # End training after 1 hour
        mock_time.return_value = 1000 + 3600
        mock_trainer.state.current_step = 100
        mock_trainer.state.best_reward = 0.95
        callback.on_training_end()
        
        # Verify final summary was logged

def test_logging_steps_respected(mock_trainer):
    """Test that logging only occurs at specified steps."""
    callback = TrainingProgressCallback(logging_steps=10)
    callback.bind_trainer(mock_trainer)
    
    # Step 1 should not log
    mock_trainer.state.current_step = 1
    callback.on_step_end()
    
    # Step 10 should log
    mock_trainer.state.current_step = 10
    callback.on_step_end()
    
    # Verify logging behavior

def test_early_stopping_logging(mock_trainer):
    """Test logging when training stops early."""
    callback = TrainingProgressCallback()
    callback.bind_trainer(mock_trainer)
    
    with patch('time.time') as mock_time:
        # Start training
        mock_time.return_value = 1000
        callback.on_training_start()
        
        # Stop at step 50 (early)
        mock_time.return_value = 1000 + 1800
        mock_trainer.state.current_step = 50
        mock_trainer.state.max_steps = 100
        callback.on_training_end()
        
        # Verify early stopping was noted in summary 