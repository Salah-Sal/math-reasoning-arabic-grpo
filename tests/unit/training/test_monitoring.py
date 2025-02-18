import pytest
import torch
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset
from transformers import PreTrainedTokenizer
from src.training.monitoring import TrainingMonitor
from datetime import datetime

@pytest.fixture
def temp_log_dir(tmp_path):
    log_dir = tmp_path / "test_logs"
    log_dir.mkdir()
    return log_dir

@pytest.fixture
def monitor(temp_log_dir):
    return TrainingMonitor(temp_log_dir)

def test_monitor_initialization(temp_log_dir):
    """Test monitor initialization and logging setup."""
    monitor = TrainingMonitor(temp_log_dir)
    
    assert monitor.log_dir == temp_log_dir
    assert monitor.log_dir.exists()
    assert monitor.start_time is None
    assert monitor.best_reward == float('-inf')
    assert monitor.training_steps == 0
    
    # Check logger setup
    assert monitor.logger.level == logging.INFO
    assert any(isinstance(h, logging.FileHandler) for h in monitor.logger.handlers)

@patch('torch.cuda.is_available', return_value=True)
@patch('torch.cuda.get_device_name', return_value='Test GPU')
@patch('torch.cuda.get_device_properties')
def test_log_system_info(mock_props, mock_name, mock_cuda, monitor, caplog):
    """Test system information logging."""
    mock_props.return_value = Mock(total_memory=10 * 1024 * 1024 * 1024)  # 10GB
    
    with caplog.at_level(logging.INFO):
        monitor.log_system_info()
    
    # Verify logged information
    assert "System Information" in caplog.text
    assert "Python Version" in caplog.text
    assert "PyTorch Version" in caplog.text
    assert "GPU: Test GPU" in caplog.text
    assert "CPU Count" in caplog.text
    assert "Memory Available" in caplog.text

def test_log_model_info(monitor, caplog):
    """Test model information logging."""
    mock_model = Mock()
    mock_params = [torch.nn.Parameter(torch.randn(10, 10)) for _ in range(2)]
    mock_model.parameters = Mock(return_value=mock_params)
    
    with caplog.at_level(logging.INFO):
        monitor.log_model_info(mock_model)
    
    # Verify logged information
    assert "Model Information" in caplog.text
    assert "Total Parameters" in caplog.text
    assert "Trainable Parameters" in caplog.text

def test_log_dataset_info(monitor, caplog):
    """Test dataset information logging."""
    mock_dataset = Dataset.from_dict({
        'input': ['text1', 'text2'],
        'label': [1, 2],
        'score': [0.5, 0.7]
    })
    
    with caplog.at_level(logging.INFO):
        monitor.log_dataset_info(mock_dataset)
    
    # Verify logged information
    assert "Dataset Information" in caplog.text
    assert "Dataset Size: 2" in caplog.text
    assert "Features" in caplog.text
    assert "Column Names" in caplog.text

def test_log_batch_processing(monitor, caplog):
    """Test batch processing logging."""
    batch = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]]),
        'text': 'sample text'
    }
    
    with caplog.at_level(logging.INFO):
        monitor.log_batch_processing(batch)
    
    # Verify logged information
    assert "Batch Processing" in caplog.text
    assert "input_ids" in caplog.text
    assert "attention_mask" in caplog.text
    assert "text" in caplog.text

def test_log_arabic_processing(monitor, caplog):
    """Test Arabic text processing logging."""
    mock_tokenizer = Mock(spec=PreTrainedTokenizer)
    mock_tokenizer.tokenize.return_value = ['token1', 'token2']
    text = "مرحبا"  # Arabic text
    
    with caplog.at_level(logging.INFO):
        monitor.log_arabic_processing(text, mock_tokenizer)
    
    # Verify logged information
    assert "Arabic Processing" in caplog.text
    assert "Original text" in caplog.text
    assert "Token count" in caplog.text
    assert "Tokens" in caplog.text
    assert "Special characters" in caplog.text

def test_on_training_start(monitor, caplog):
    """Test training start logging."""
    mock_trainer = Mock()
    mock_trainer.args = Mock(
        learning_rate=0.001,
        per_device_train_batch_size=8,
        max_steps=1000
    )
    monitor.bind_trainer(mock_trainer)
    
    with caplog.at_level(logging.INFO):
        monitor._on_training_start()
    
    # Verify logged information
    assert monitor.start_time is not None
    assert "Training started at" in caplog.text
    assert "Training Configuration" in caplog.text
    assert "Learning rate" in caplog.text
    assert "Batch size" in caplog.text
    assert "Max steps" in caplog.text

def test_on_step_end(monitor, caplog):
    """Test step end logging."""
    metrics = {'loss': 0.5, 'reward': 1.0}
    args = {'metrics': metrics}
    
    with caplog.at_level(logging.INFO):
        monitor._on_step_end(args)
    
    # Verify logged information
    assert monitor.training_steps == 1
    assert "Step 1 metrics" in caplog.text
    assert "loss: 0.5" in caplog.text
    assert "New best reward" in caplog.text
    assert monitor.best_reward == 1.0

def test_on_training_end(monitor, caplog):
    """Test training end logging."""
    monitor.start_time = datetime.now()
    monitor.training_steps = 100
    monitor.best_reward = 0.95
    
    with caplog.at_level(logging.INFO):
        monitor._on_training_end()
    
    # Verify logged information
    assert "Training Summary" in caplog.text
    assert "Training completed at" in caplog.text
    assert "Total duration" in caplog.text
    assert "Total steps: 100" in caplog.text
    assert "Best reward achieved: 0.95" in caplog.text

def test_error_handling(monitor, caplog):
    """Test error handling in monitoring functions."""
    # Test with invalid model
    invalid_model = None
    with caplog.at_level(logging.ERROR):
        monitor.log_model_info(invalid_model)
    
    # Test with invalid dataset
    invalid_dataset = None
    with caplog.at_level(logging.ERROR):
        monitor.log_dataset_info(invalid_dataset)
    
    # Verify error logging
    assert any(record.levelno == logging.ERROR for record in caplog.records) 