import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import shutil
from src.training.callbacks.checkpoint import ModelCheckpointCallback

@pytest.fixture
def mock_trainer():
    """Mock trainer for testing."""
    trainer = MagicMock()
    trainer.state = MagicMock()
    trainer.state.current_step = 0
    trainer.state.max_steps = 100
    trainer.state.best_reward = 0.0
    trainer.state.current_reward = 0.0
    trainer.model = MagicMock()
    return trainer

@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create a temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    yield checkpoint_dir
    # Cleanup
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

def test_checkpoint_callback_initialization(temp_checkpoint_dir):
    """Test checkpoint callback initialization."""
    callback = ModelCheckpointCallback(
        checkpoint_dir=temp_checkpoint_dir,
        save_steps=10,
        max_checkpoints=3
    )
    assert callback.checkpoint_dir == temp_checkpoint_dir
    assert callback.save_steps == 10
    assert callback.max_checkpoints == 3
    assert callback._saved_checkpoints == []

def test_checkpoint_dir_creation():
    """Test checkpoint directory is created if it doesn't exist."""
    with patch('pathlib.Path.mkdir') as mock_mkdir:
        callback = ModelCheckpointCallback(
            checkpoint_dir=Path("/nonexistent/dir"),
            save_steps=10
        )
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

def test_save_checkpoint(mock_trainer, temp_checkpoint_dir):
    """Test saving a checkpoint."""
    callback = ModelCheckpointCallback(
        checkpoint_dir=temp_checkpoint_dir,
        save_steps=10
    )
    callback.bind_trainer(mock_trainer)
    
    # Simulate step 10
    mock_trainer.state.current_step = 10
    mock_trainer.state.current_reward = 0.5
    
    callback.on_step_end()
    
    # Verify checkpoint was saved
    expected_path = temp_checkpoint_dir / f"checkpoint-{10}"
    assert expected_path in callback._saved_checkpoints
    mock_trainer.model.save_pretrained.assert_called_once()

def test_save_best_checkpoint(mock_trainer, temp_checkpoint_dir):
    """Test saving checkpoint when new best reward is achieved."""
    callback = ModelCheckpointCallback(
        checkpoint_dir=temp_checkpoint_dir,
        save_steps=10,
        save_best=True
    )
    callback.bind_trainer(mock_trainer)
    
    # Simulate new best reward
    mock_trainer.state.current_step = 5
    mock_trainer.state.current_reward = 0.8
    mock_trainer.state.best_reward = 0.8
    
    callback.on_step_end()
    
    # Verify best checkpoint was saved
    expected_path = temp_checkpoint_dir / "checkpoint-best"
    assert expected_path.name in [p.name for p in callback._saved_checkpoints]

def test_checkpoint_rotation(mock_trainer, temp_checkpoint_dir):
    """Test checkpoint rotation when max_checkpoints is reached."""
    callback = ModelCheckpointCallback(
        checkpoint_dir=temp_checkpoint_dir,
        save_steps=10,
        max_checkpoints=2
    )
    callback.bind_trainer(mock_trainer)
    
    # Save 3 checkpoints
    for step in [10, 20, 30]:
        mock_trainer.state.current_step = step
        callback.on_step_end()
    
    # Verify only latest 2 checkpoints are kept
    assert len(callback._saved_checkpoints) == 2
    assert temp_checkpoint_dir / f"checkpoint-{30}" in callback._saved_checkpoints
    assert temp_checkpoint_dir / f"checkpoint-{20}" in callback._saved_checkpoints
    assert temp_checkpoint_dir / f"checkpoint-{10}" not in callback._saved_checkpoints

def test_save_steps_respected(mock_trainer, temp_checkpoint_dir):
    """Test that checkpoints are only saved at specified steps."""
    callback = ModelCheckpointCallback(
        checkpoint_dir=temp_checkpoint_dir,
        save_steps=10
    )
    callback.bind_trainer(mock_trainer)
    
    # Step 5 should not save
    mock_trainer.state.current_step = 5
    callback.on_step_end()
    assert len(callback._saved_checkpoints) == 0
    
    # Step 10 should save
    mock_trainer.state.current_step = 10
    callback.on_step_end()
    assert len(callback._saved_checkpoints) == 1

def test_final_checkpoint(mock_trainer, temp_checkpoint_dir):
    """Test saving final checkpoint at training end."""
    callback = ModelCheckpointCallback(
        checkpoint_dir=temp_checkpoint_dir,
        save_steps=10,
        save_final=True
    )
    callback.bind_trainer(mock_trainer)
    
    mock_trainer.state.current_step = 100
    callback.on_training_end()
    
    # Verify final checkpoint was saved
    expected_path = temp_checkpoint_dir / "checkpoint-final"
    assert expected_path.name in [p.name for p in callback._saved_checkpoints]

def test_checkpoint_cleanup(mock_trainer, temp_checkpoint_dir):
    """Test cleanup of old checkpoints."""
    callback = ModelCheckpointCallback(
        checkpoint_dir=temp_checkpoint_dir,
        save_steps=10,
        max_checkpoints=1
    )
    callback.bind_trainer(mock_trainer)
    
    # Create some test checkpoints
    for step in [10, 20]:
        mock_trainer.state.current_step = step
        callback.on_step_end()
    
    # Verify old checkpoints are removed
    assert len(callback._saved_checkpoints) == 1
    assert callback._saved_checkpoints[0].name == f"checkpoint-{20}"

def test_error_handling(mock_trainer, temp_checkpoint_dir):
    """Test error handling during checkpoint saving."""
    callback = ModelCheckpointCallback(
        checkpoint_dir=temp_checkpoint_dir,
        save_steps=10
    )
    callback.bind_trainer(mock_trainer)
    
    # Simulate save error
    mock_trainer.model.save_pretrained.side_effect = Exception("Save failed")
    
    # Should not raise but log error
    mock_trainer.state.current_step = 10
    callback.on_step_end()
    # Verify error was logged 