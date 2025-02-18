import pytest
import torch
from src.core.model.qwen_model import QwenModel

@pytest.mark.integration
def test_model_gpu_setup():
    """Test model initialization with GPU setup."""
    model = QwenModel()
    gpu_info = model.validate_gpu_setup()
    
    # Log GPU information
    print("\nGPU Setup Information:")
    for key, value in gpu_info.items():
        print(f"{key}: {value}")
    
    assert gpu_info["cuda_available"] == torch.cuda.is_available()
    if torch.cuda.is_available():
        assert gpu_info["device_name"] == torch.cuda.get_device_name()

@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_loading():
    """Test actual model loading on GPU."""
    model = QwenModel()
    
    try:
        # Load with minimal configuration for testing
        loaded_model, tokenizer = model.load_model(
            max_seq_length=128,  # Smaller for testing
            gpu_memory_utilization=0.5  # Conservative memory usage
        )
        
        assert loaded_model is not None
        assert tokenizer is not None
        
        # Verify model is on CUDA
        assert next(loaded_model.parameters()).device.type == "cuda"
        
        # Clean up
        model.clear_gpu_memory()
        del loaded_model
        del tokenizer
        torch.cuda.empty_cache()
        
    except Exception as e:
        pytest.fail(f"Model loading failed: {str(e)}")

@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_peft_setup():
    """Test PEFT setup with actual model."""
    model = QwenModel()
    
    try:
        # Load model first
        model.load_model(max_seq_length=128)
        
        # Setup PEFT
        peft_model = model.setup_peft(
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_rank=8  # Smaller for testing
        )
        
        assert peft_model is not None
        
        # Clean up
        model.clear_gpu_memory()
        del peft_model
        torch.cuda.empty_cache()
        
    except Exception as e:
        pytest.fail(f"PEFT setup failed: {str(e)}")

@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_save_load_cycle(tmp_path):
    """Test full save and load cycle with actual model."""
    model = QwenModel()
    save_path = tmp_path / "test_model"
    
    try:
        # Initial model setup
        model.load_model(max_seq_length=128)
        model.setup_peft(lora_rank=8)
        
        # Test LoRA saving
        model.save_model(str(save_path), save_method="lora")
        assert (save_path).exists()
        
        # Clean up
        model.clear_gpu_memory()
        torch.cuda.empty_cache()
        
    except Exception as e:
        pytest.fail(f"Save/load cycle failed: {str(e)}") 