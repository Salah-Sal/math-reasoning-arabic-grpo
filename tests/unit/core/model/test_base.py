import pytest
import torch
from src.core.model.base import BaseLanguageModel

class TestBaseModel(BaseLanguageModel):
    """Test implementation of BaseLanguageModel for testing abstract methods."""
    def load_model(self, *args, **kwargs):
        return None, None
    
    def setup_peft(self, *args, **kwargs):
        return None
    
    def save_model(self, *args, **kwargs):
        pass

@pytest.fixture
def base_model():
    return TestBaseModel()

def test_base_model_initialization():
    """Test basic model initialization."""
    model = TestBaseModel()
    assert model.model is None
    assert model.tokenizer is None

def test_validate_gpu_setup(base_model):
    """Test GPU setup validation."""
    gpu_info = base_model.validate_gpu_setup()
    assert isinstance(gpu_info, dict)
    assert "cuda_available" in gpu_info
    
    if torch.cuda.is_available():
        assert gpu_info["cuda_available"] is True
        assert "device_count" in gpu_info
        assert "current_device" in gpu_info
        assert "device_name" in gpu_info
        assert "memory_allocated" in gpu_info
        assert "memory_reserved" in gpu_info
    else:
        assert gpu_info["cuda_available"] is False

def test_clear_gpu_memory(base_model):
    """Test GPU memory clearing."""
    # This should not raise any errors, regardless of GPU availability
    base_model.clear_gpu_memory()

def test_abstract_methods():
    """Test that abstract methods are properly defined."""
    with pytest.raises(TypeError):
        BaseLanguageModel() 