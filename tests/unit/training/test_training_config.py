import pytest
from pathlib import Path
from pydantic import ValidationError
import yaml
from typing import Dict, Any

@pytest.fixture
def default_grpo_config() -> Dict[str, Any]:
    """Default GRPO training configuration."""
    return {
        "training": {
            "learning_rate": 3e-6,
            "max_steps": 1000,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "max_prompt_length": 256,
            "max_completion_length": 128,
            "logging_steps": 10,
            "save_steps": 100,
            "warmup_steps": 100,
            "max_checkpoints": 3
        },
        "memory": {
            "gpu_memory_utilization": 0.7,
            "use_gradient_checkpointing": True,
            "optimize_memory_use": True,
            "max_memory_MB": 8000
        },
        "reward": {
            "weights": {
                "xml_structure": 0.3,
                "format": 0.3,
                "correctness": 0.4
            },
            "cache_size": 1000
        },
        "paths": {
            "output_dir": "outputs/grpo_training",
            "checkpoint_dir": "checkpoints",
            "log_dir": "logs"
        }
    }

@pytest.fixture
def sample_yaml_config(tmp_path: Path, default_grpo_config: Dict[str, Any]) -> Path:
    """Create a sample YAML config file."""
    config_path = tmp_path / "grpo_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(default_grpo_config, f)
    return config_path

def test_grpo_config_initialization():
    """Test basic GRPOConfig initialization with default values."""
    from src.training.config import GRPOConfig
    
    config = GRPOConfig()
    assert config.training.learning_rate == 3e-6
    assert config.training.max_steps > 0
    assert config.training.per_device_train_batch_size > 0
    assert config.memory.gpu_memory_utilization <= 1.0

def test_grpo_config_validation():
    """Test configuration validation rules."""
    from src.training.config import GRPOConfig
    
    # Test invalid learning rate
    with pytest.raises(ValidationError):
        GRPOConfig(training={"learning_rate": -1.0})
    
    # Test invalid batch size
    with pytest.raises(ValidationError):
        GRPOConfig(training={"per_device_train_batch_size": 0})
    
    # Test invalid GPU memory utilization
    with pytest.raises(ValidationError):
        GRPOConfig(memory={"gpu_memory_utilization": 1.5})

def test_grpo_config_from_yaml(sample_yaml_config: Path):
    """Test loading configuration from YAML file."""
    from src.training.config import GRPOConfig
    
    config = GRPOConfig.from_yaml(sample_yaml_config)
    assert config.training.learning_rate == 3e-6
    assert config.memory.gpu_memory_utilization == 0.7
    assert config.reward.weights.xml_structure == 0.3

def test_memory_settings_validation():
    """Test memory management settings validation."""
    from src.training.config import GRPOConfig
    
    # Test valid memory settings
    config = GRPOConfig(memory={
        "gpu_memory_utilization": 0.7,
        "max_memory_MB": 8000
    })
    assert config.memory.gpu_memory_utilization == 0.7
    assert config.memory.max_memory_MB == 8000
    
    # Test invalid memory settings
    with pytest.raises(ValidationError):
        GRPOConfig(memory={"max_memory_MB": -1})

def test_checkpoint_settings():
    """Test checkpoint configuration settings."""
    from src.training.config import GRPOConfig
    
    config = GRPOConfig(training={
        "save_steps": 100,
        "max_checkpoints": 3
    })
    assert config.training.save_steps == 100
    assert config.training.max_checkpoints == 3
    
    # Test invalid settings
    with pytest.raises(ValidationError):
        GRPOConfig(training={"max_checkpoints": -1})

def test_reward_weights_validation():
    """Test reward weights configuration and validation."""
    from src.training.config import GRPOConfig
    
    # Test valid weights
    config = GRPOConfig(reward={
        "weights": {
            "xml_structure": 0.3,
            "format": 0.3,
            "correctness": 0.4
        }
    })
    assert abs(sum(config.reward.weights.dict().values()) - 1.0) < 1e-6
    
    # Test invalid weights (sum > 1)
    with pytest.raises(ValidationError):
        GRPOConfig(reward={
            "weights": {
                "xml_structure": 0.5,
                "format": 0.5,
                "correctness": 0.5
            }
        })

def test_path_resolution():
    """Test path resolution and validation."""
    from src.training.config import GRPOConfig
    
    config = GRPOConfig(paths={
        "output_dir": "outputs/test",
        "checkpoint_dir": "checkpoints"
    })
    assert isinstance(config.paths.output_dir, Path)
    assert isinstance(config.paths.checkpoint_dir, Path)

def test_config_serialization(tmp_path: Path):
    """Test configuration serialization to YAML."""
    from src.training.config import GRPOConfig
    
    config = GRPOConfig()
    save_path = tmp_path / "test_config.yaml"
    
    # Test saving
    config.save_to_yaml(save_path)
    assert save_path.exists()
    
    # Test loading back
    loaded_config = GRPOConfig.from_yaml(save_path)
    assert loaded_config.training.learning_rate == config.training.learning_rate
    assert loaded_config.memory.gpu_memory_utilization == config.memory.gpu_memory_utilization 