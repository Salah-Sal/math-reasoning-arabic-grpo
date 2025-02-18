import pytest
from unittest.mock import Mock, patch
import logging
from src.training.callbacks.early_stopping import EarlyStoppingCallback

@pytest.fixture
def mock_trainer():
    """Mock trainer for testing."""
    trainer = Mock()
    trainer.state = Mock()
    trainer.state.current_step = 0
    trainer.state.max_steps = 100
    trainer.state.current_reward = 0.0
    trainer.state.best_reward = 0.0
    return trainer

@pytest.fixture
def callback():
    """Create callback instance with default settings."""
    return EarlyStoppingCallback(
        patience=5,
        min_improvement=0.01,
        min_steps=10
    )

def test_early_stopping_initialization():
    """Test callback initialization."""
    callback = EarlyStoppingCallback(patience=5, min_improvement=0.01)
    assert callback.patience == 5
    assert callback.min_improvement == 0.01
    assert callback.min_steps == 0
    assert callback._best_reward == float('-inf')
    assert callback._steps_without_improvement == 0

def test_early_stopping_validation():
    """Test validation of initialization parameters."""
    with pytest.raises(ValueError):
        EarlyStoppingCallback(patience=-1)
    
    with pytest.raises(ValueError):
        EarlyStoppingCallback(min_improvement=-0.1)
    
    with pytest.raises(ValueError):
        EarlyStoppingCallback(min_steps=-1)

def test_early_stopping_reset(callback):
    """Test resetting early stopping state."""
    callback._best_reward = 0.5
    callback._steps_without_improvement = 3
    
    callback.reset()
    assert callback._best_reward == float('-inf')
    assert callback._steps_without_improvement == 0

def test_early_stopping_before_min_steps(callback, mock_trainer, caplog):
    """Test that stopping doesn't occur before minimum steps."""
    callback.bind_trainer(mock_trainer)
    mock_trainer.state.current_step = 5  # Less than min_steps
    
    with caplog.at_level(logging.INFO):
        callback._on_step_end()
    
    assert not mock_trainer.stop_training.called
    assert "Minimum steps not reached" in caplog.text

def test_early_stopping_improvement(callback, mock_trainer):
    """Test behavior when improvement occurs."""
    callback.bind_trainer(mock_trainer)
    mock_trainer.state.current_step = 20
    
    # Initial reward
    mock_trainer.state.current_reward = 0.5
    callback._on_step_end()
    assert callback._best_reward == 0.5
    assert callback._steps_without_improvement == 0
    
    # Improved reward
    mock_trainer.state.current_reward = 0.6
    callback._on_step_end()
    assert callback._best_reward == 0.6
    assert callback._steps_without_improvement == 0

def test_early_stopping_no_improvement(callback, mock_trainer, caplog):
    """Test stopping when no improvement occurs for patience steps."""
    callback.bind_trainer(mock_trainer)
    mock_trainer.state.current_step = 20
    
    # Initial good reward
    mock_trainer.state.current_reward = 0.5
    callback._on_step_end()
    
    # Several steps without improvement
    for _ in range(callback.patience + 1):
        mock_trainer.state.current_reward = 0.49  # Below min_improvement threshold
        callback._on_step_end()
    
    assert mock_trainer.stop_training.called
    assert "Stopping early" in caplog.text

def test_early_stopping_small_improvement(callback, mock_trainer):
    """Test handling of improvements below min_improvement threshold."""
    callback.bind_trainer(mock_trainer)
    mock_trainer.state.current_step = 20
    
    # Initial reward
    mock_trainer.state.current_reward = 0.5
    callback._on_step_end()
    
    # Small improvement (below min_improvement)
    mock_trainer.state.current_reward = 0.505  # Less than min_improvement (0.01)
    callback._on_step_end()
    
    assert callback._steps_without_improvement == 1

def test_early_stopping_training_end(callback, mock_trainer):
    """Test cleanup at training end."""
    callback.bind_trainer(mock_trainer)
    
    # Set some state
    callback._best_reward = 0.5
    callback._steps_without_improvement = 3
    
    callback._on_training_end()
    
    # Verify cleanup
    assert callback._best_reward == float('-inf')
    assert callback._steps_without_improvement == 0

def test_early_stopping_deactivated(callback, mock_trainer):
    """Test behavior when callback is deactivated."""
    callback.bind_trainer(mock_trainer)
    callback.deactivate()
    
    mock_trainer.state.current_step = 20
    mock_trainer.state.current_reward = 0.0
    
    callback._on_step_end()
    assert not mock_trainer.stop_training.called 