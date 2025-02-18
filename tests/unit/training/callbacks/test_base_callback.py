import pytest
from typing import List, Dict, Any
import logging
from unittest.mock import MagicMock

logger = logging.getLogger(__name__)

@pytest.fixture
def mock_trainer():
    """Mock trainer for testing callbacks."""
    trainer = MagicMock()
    trainer.state = MagicMock()
    trainer.state.current_step = 0
    trainer.state.max_steps = 100
    trainer.state.best_reward = 0.0
    trainer.state.current_reward = 0.0
    return trainer

def test_base_callback_initialization():
    """Test basic callback initialization."""
    from src.training.callbacks.base import BaseCallback
    
    callback = BaseCallback()
    assert callback.order == 0
    assert callback._trainer is None
    assert callback.is_active

def test_callback_activation():
    """Test callback activation/deactivation."""
    from src.training.callbacks.base import BaseCallback
    
    callback = BaseCallback()
    assert callback.is_active
    
    callback.deactivate()
    assert not callback.is_active
    
    callback.activate()
    assert callback.is_active

def test_callback_order():
    """Test callback ordering."""
    from src.training.callbacks.base import BaseCallback
    
    callback1 = BaseCallback(order=1)
    callback2 = BaseCallback(order=2)
    callback3 = BaseCallback(order=0)
    
    callbacks = sorted([callback1, callback2, callback3])
    assert callbacks[0].order == 0
    assert callbacks[-1].order == 2

def test_callback_trainer_binding(mock_trainer):
    """Test binding callback to trainer."""
    from src.training.callbacks.base import BaseCallback
    
    callback = BaseCallback()
    callback.bind_trainer(mock_trainer)
    assert callback.trainer is mock_trainer

def test_callback_event_hooks(mock_trainer):
    """Test all callback event hooks are called."""
    from src.training.callbacks.base import BaseCallback
    
    class TestCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.events_called = []
        
        def on_training_start(self):
            self.events_called.append("training_start")
        
        def on_training_end(self):
            self.events_called.append("training_end")
        
        def on_step_begin(self):
            self.events_called.append("step_begin")
        
        def on_step_end(self):
            self.events_called.append("step_end")
        
        def on_evaluate(self):
            self.events_called.append("evaluate")
        
        def on_save(self):
            self.events_called.append("save")
    
    callback = TestCallback()
    callback.bind_trainer(mock_trainer)
    
    # Simulate training events
    callback.on_training_start()
    callback.on_step_begin()
    callback.on_step_end()
    callback.on_evaluate()
    callback.on_save()
    callback.on_training_end()
    
    expected_events = [
        "training_start",
        "step_begin",
        "step_end",
        "evaluate",
        "save",
        "training_end"
    ]
    assert callback.events_called == expected_events

def test_callback_error_handling(mock_trainer):
    """Test error handling in callbacks."""
    from src.training.callbacks.base import BaseCallback
    
    class ErrorCallback(BaseCallback):
        def _on_step_begin(self):
            raise ValueError("Test error")
    
    callback = ErrorCallback()
    callback.bind_trainer(mock_trainer)
    
    # Should not raise but log error
    callback.on_step_begin()
    # Verify error was logged (would need to check logs)

def test_callback_state_access(mock_trainer):
    """Test callback access to trainer state."""
    from src.training.callbacks.base import BaseCallback
    
    callback = BaseCallback()
    callback.bind_trainer(mock_trainer)
    
    assert callback.trainer.state.current_step == 0
    assert callback.trainer.state.max_steps == 100
    
    # Update mock state
    mock_trainer.state.current_step = 50
    assert callback.trainer.state.current_step == 50

def test_callback_validation():
    """Test callback validation rules."""
    from src.training.callbacks.base import BaseCallback
    
    # Test invalid order
    with pytest.raises(ValueError):
        BaseCallback(order=-1)
    
    # Test trainer access before binding
    callback = BaseCallback()
    with pytest.raises(RuntimeError):
        _ = callback.trainer 