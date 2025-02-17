import pytest
from pathlib import Path
from src.infrastructure.config import ProjectConfig, ModelConfig, TrainingConfig, DataConfig

# Fixture for test config path
@pytest.fixture
def test_config_path():
    return Path(__file__).parent.parent.parent / "fixtures" / "test_config.yaml"

# Fixture for default config
@pytest.fixture
def default_config():
    return ProjectConfig()

class TestProjectConfig:
    def test_default_config_creation(self, default_config):
        """Test that default configuration can be created"""
        assert default_config.model.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert default_config.training.learning_rate == 5e-6
        assert default_config.seed == 3407

    def test_load_from_yaml(self, test_config_path):
        """Test loading configuration from YAML file"""
        config = ProjectConfig.load_from_yaml(test_config_path)
        assert config.model.max_seq_length == 256
        assert config.training.learning_rate == 1e-5
        assert config.data.data_dir == Path("/tmp/test_data")
        assert config.seed == 42

    def test_save_to_yaml(self, tmp_path):
        """Test saving configuration to YAML file"""
        config = ProjectConfig()
        save_path = tmp_path / "saved_config.yaml"
        config.save_to_yaml(save_path)
        
        # Load the saved config and verify
        loaded_config = ProjectConfig.load_from_yaml(save_path)
        assert loaded_config.dict() == config.dict()

    def test_invalid_gpu_utilization(self):
        """Test validation of GPU memory utilization"""
        with pytest.raises(ValueError):
            ModelConfig(gpu_memory_utilization=1.5)
        with pytest.raises(ValueError):
            ModelConfig(gpu_memory_utilization=-0.1)

    def test_path_conversion(self):
        """Test that strings are properly converted to Path objects"""
        config = DataConfig(
            data_dir="/some/path",
            output_dir="/other/path"
        )
        assert isinstance(config.data_dir, Path)
        assert isinstance(config.output_dir, Path)

    def test_missing_required_fields(self):
        """Test that missing required fields raise appropriate errors"""
        with pytest.raises(ValueError):
            ModelConfig(model_name=None)

    def test_type_validation(self):
        """Test type validation for configuration fields"""
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate="invalid")
        with pytest.raises(ValueError):
            TrainingConfig(per_device_train_batch_size=3.14)

    def test_config_immutability(self, default_config):
        """Test that configuration is immutable after creation"""
        with pytest.raises(Exception):
            default_config.seed = 42 