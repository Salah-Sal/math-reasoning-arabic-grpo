import pytest
from pathlib import Path
import torch
from src.training.trainer import Trainer
from src.data.dataset import ArabicMathDataset
from src.infrastructure.config import ProjectConfig, ModelConfig, TrainingConfig
from src.infrastructure.logging import get_logger
import pydantic_core

logger = get_logger(__name__)

@pytest.fixture
def base_config():
    """Create base configuration for testing"""
    return ProjectConfig(
        model=ModelConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            max_seq_length=256,
            load_in_4bit=True,
            fast_inference=True,
            max_lora_rank=8,
            gpu_memory_utilization=0.6
        ),
        training=TrainingConfig(
            learning_rate=5e-6,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_generations=4,
            max_steps=10,
            logging_steps=1
        )
    )

@pytest.fixture
def poc_config(base_config):
    """Create a copy of config with POC settings"""
    config_dict = base_config.model_dump()
    config_dict['training']['per_device_train_batch_size'] = 1
    config_dict['training']['gradient_accumulation_steps'] = 1
    return ProjectConfig(**config_dict)

@pytest.fixture
def sample_dataset(tmp_path):
    """Create a tiny dataset for testing"""
    data_dir = tmp_path / "sample_data"
    data_dir.mkdir()
    
    sample_file = Path(__file__).parent.parent.parent / "fixtures" / "data" / "sample_problems" / "problem_001.json"
    if not sample_file.exists():
        pytest.skip("Sample data file not found")
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(data_dir / "problem_001.json", 'w', encoding='utf-8') as f:
        f.write(content)
    
    return ArabicMathDataset(data_dir=data_dir)

class TestTrainerInitialization:
    """Test suite for Trainer initialization"""
    
    @pytest.mark.gpu
    def test_basic_initialization(self, poc_config):
        """Test basic trainer initialization without model loading"""
        trainer = Trainer(config=poc_config, poc_mode=True)
        assert trainer.poc_mode == True
        assert trainer.config == poc_config
        assert trainer.output_dir.exists()

    @pytest.mark.gpu
    def test_config_validation(self, base_config):
        """Test configuration validation"""
        invalid_config = base_config.model_dump()
        invalid_config['model']['gpu_memory_utilization'] = 2.0
        with pytest.raises(
            pydantic_core.ValidationError,
            match="Input should be less than or equal to 1"
        ):
            Trainer(config=ProjectConfig(**invalid_config), poc_mode=True)

class TestTrainerModelLoading:
    """Test suite for model loading functionality"""

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_loading(self, poc_config):
        """Test model loading and initialization"""
        trainer = Trainer(config=poc_config, poc_mode=True)
        assert trainer.model is not None
        assert trainer.tokenizer is not None
        assert trainer.model.config.model_type == "qwen2"

    @pytest.mark.gpu
    def test_memory_management(self, poc_config):
        """Test GPU memory management during model loading"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        initial_memory = torch.cuda.memory_allocated()
        trainer = Trainer(config=poc_config, poc_mode=True)
        loaded_memory = torch.cuda.memory_allocated()
        
        assert loaded_memory > initial_memory
        del trainer
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        assert final_memory <= initial_memory

class TestTrainerDatasetHandling:
    """Test suite for dataset handling"""

    def test_poc_dataset_preparation(self, poc_config, sample_dataset):
        """Test dataset preparation in POC mode"""
        trainer = Trainer(config=poc_config, poc_mode=True)
        poc_dataset = trainer.prepare_dataset(sample_dataset)
        
        assert len(poc_dataset) <= 100
        sample = poc_dataset[0]
        assert 'prompt' in sample
        assert 'answer' in sample
        assert len(sample['prompt']) == 2

    def test_full_dataset_preparation(self, poc_config, sample_dataset):
        """Test dataset preparation in full mode"""
        trainer = Trainer(config=poc_config, poc_mode=False)
        full_dataset = trainer.prepare_dataset(sample_dataset)
        assert len(full_dataset) == len(sample_dataset)

def _adjust_config_for_gpu(self, config: ProjectConfig) -> ProjectConfig:
    """Create a new config with adjusted settings for GPU constraints"""
    config_dict = config.model_dump()
    if torch.cuda.get_device_properties(0).total_memory < 10 * 1024**3:
        config_dict['model']['max_seq_length'] = 256
        config_dict['model']['gpu_memory_utilization'] = 0.6
        config_dict['training']['per_device_train_batch_size'] = 1
        config_dict['training']['gradient_accumulation_steps'] = 1
        config_dict['training']['num_generations'] = 4
        logger.info("Adjusted settings for 8GB GPU")
    return ProjectConfig(**config_dict)

def __init__(self, config: ProjectConfig, poc_mode: bool = False):
    self.config = self._adjust_config_for_gpu(config)
    self.poc_mode = poc_mode
    self._initialize_model() 