import pytest
import os
import torch
from unittest.mock import patch, MagicMock
from src.core.model.qwen_model import QwenModel

@pytest.fixture
def qwen_model():
    return QwenModel()

@pytest.fixture
def mock_fast_language_model():
    with patch('src.core.model.qwen_model.FastLanguageModel') as mock:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock.from_pretrained.return_value = (mock_model, mock_tokenizer)
        yield mock

def test_qwen_model_initialization():
    """Test Qwen model initialization."""
    model = QwenModel()
    assert model.model is None
    assert model.tokenizer is None

def test_load_model(qwen_model, mock_fast_language_model):
    """Test model loading functionality."""
    model, tokenizer = qwen_model.load_model(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_seq_length=384
    )
    
    # Verify FastLanguageModel.from_pretrained was called with correct arguments
    mock_fast_language_model.from_pretrained.assert_called_once_with(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_seq_length=384,
        load_in_4bit=True,
        fast_inference=False,
        gpu_memory_utilization=0.7
    )
    
    assert model is not None
    assert tokenizer is not None

def test_setup_peft_without_model(qwen_model):
    """Test PEFT setup fails without loaded model."""
    with pytest.raises(ValueError, match="Model must be loaded before setting up PEFT"):
        qwen_model.setup_peft()

@patch('src.core.model.qwen_model.FastLanguageModel')
def test_setup_peft_with_model(mock_fast_language_model, qwen_model):
    """Test PEFT setup with loaded model."""
    # First load the model
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_fast_language_model.from_pretrained.return_value = (mock_model, mock_tokenizer)
    qwen_model.load_model()
    
    # Then setup PEFT
    mock_fast_language_model.get_peft_model.return_value = mock_model
    result = qwen_model.setup_peft()
    
    assert result is not None
    mock_fast_language_model.get_peft_model.assert_called_once()

def test_save_model_without_loading(qwen_model, tmp_path):
    """Test save_model fails when model is not loaded."""
    with pytest.raises(ValueError, match="Model and tokenizer must be loaded before saving"):
        qwen_model.save_model(str(tmp_path))

@patch('src.core.model.qwen_model.FastLanguageModel')
def test_save_model_lora(mock_fast_language_model, qwen_model, tmp_path):
    """Test saving model with LoRA method."""
    # Setup mock model and tokenizer
    mock_model = MagicMock()
    mock_base_model = MagicMock()
    mock_tokenizer = MagicMock()
    
    # Setup the model hierarchy to match PEFT structure
    mock_model.base_model = mock_base_model
    mock_fast_language_model.from_pretrained.return_value = (mock_model, mock_tokenizer)
    qwen_model.load_model()
    
    # Test saving
    save_path = str(tmp_path / "test_model")
    qwen_model.save_model(save_path, save_method="lora")
    
    # Verify that save_pretrained was called on the base_model
    mock_base_model.save_pretrained.assert_called_once_with(save_path)

@patch('src.core.model.qwen_model.FastLanguageModel')
def test_save_model_merged(mock_fast_language_model, qwen_model, tmp_path):
    """Test saving model with merged_4bit method."""
    # Setup mock model and tokenizer
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_fast_language_model.from_pretrained.return_value = (mock_model, mock_tokenizer)
    qwen_model.load_model()
    
    # Test saving
    save_path = str(tmp_path / "test_model")
    qwen_model.save_model(save_path, save_method="merged_4bit")
    
    mock_model.save_pretrained_merged.assert_called_once_with(
        save_path,
        mock_tokenizer,
        save_method="merged_4bit"
    )

def test_save_model_invalid_method(qwen_model, tmp_path):
    """Test save_model fails with invalid save method."""
    # Setup mock model and tokenizer
    qwen_model.model = MagicMock()
    qwen_model.tokenizer = MagicMock()
    
    with pytest.raises(ValueError, match="Unsupported save method"):
        qwen_model.save_model(str(tmp_path), save_method="invalid_method") 