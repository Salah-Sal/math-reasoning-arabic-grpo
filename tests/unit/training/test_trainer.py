import pytest
from pathlib import Path
from src.training.trainer import Trainer
from src.data.dataset import ArabicMathDataset
from src.infrastructure.config import ProjectConfig
from src.infrastructure.logging import get_logger
import torch

logger = get_logger(__name__)

@pytest.fixture
def sample_config(tmp_path):
    """Create a sample configuration for POC training"""
    return ProjectConfig(
        model=dict(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            max_seq_length=384,
            load_in_4bit=True,
            fast_inference=False,
            max_lora_rank=8,
            gpu_memory_utilization=0.7
        ),
        training=dict(
            learning_rate=5e-6,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            max_steps=10,  # Small number of steps for testing
            logging_steps=1
        )
    )

@pytest.fixture
def sample_dataset(tmp_path):
    """Create a tiny dataset for testing"""
    data_dir = tmp_path / "sample_data"
    data_dir.mkdir()
    
    # Copy our sample problem
    sample_file = Path(__file__).parent.parent.parent / "fixtures" / "data" / "sample_problems" / "problem_001.json"
    if not sample_file.exists():
        pytest.skip("Sample data file not found")
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(data_dir / "problem_001.json", 'w', encoding='utf-8') as f:
        f.write(content)
    
    return ArabicMathDataset(data_dir=data_dir)

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available"
)
class TestTrainer:
    """Test suite for Trainer class"""
    
    @pytest.fixture(autouse=True)
    def setup(self, sample_config):
        """Setup for each test - adjust config for available GPU memory"""
        total_memory = torch.cuda.get_device_properties(0).total_memory
        total_gb = total_memory / 1024**3
        
        if total_gb < 10:
            # Create new config instead of modifying
            config_dict = sample_config.model_dump()
            config_dict['model']['max_seq_length'] = 256
            config_dict['model']['gpu_memory_utilization'] = 0.6
            config_dict['training']['per_device_train_batch_size'] = 4
            config_dict['training']['gradient_accumulation_steps'] = 4
            return ProjectConfig(**config_dict)
        
        return sample_config
    
    def test_initialization(self, sample_config, sample_dataset):
        """Test basic trainer initialization"""
        trainer = Trainer(config=sample_config, poc_mode=True)
        assert trainer.poc_mode == True
        assert trainer.model is not None
        assert trainer.tokenizer is not None
    
    def test_memory_management(self, sample_config):
        """Test memory clearing functionality"""
        # Skip if not enough memory
        if torch.cuda.get_device_properties(0).total_memory < 8 * 1024**3:
            pytest.skip("Not enough GPU memory")
        
        initial_memory = torch.cuda.memory_allocated()
        trainer = Trainer(config=sample_config, poc_mode=True)
        del trainer
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        assert final_memory <= initial_memory
    
    def test_dataset_preparation(self, sample_config, sample_dataset):
        """Test dataset preparation in POC mode"""
        trainer = Trainer(config=sample_config, poc_mode=True)
        prepared_dataset = trainer.prepare_dataset(sample_dataset)
        assert len(prepared_dataset) <= 100  # POC size limit
        
        # Test full mode
        trainer = Trainer(config=sample_config, poc_mode=False)
        full_dataset = trainer.prepare_dataset(sample_dataset)
        assert len(full_dataset) == len(sample_dataset)
    
    def test_config_validation(self, sample_config):
        """Test configuration validation"""
        # Test invalid memory utilization
        invalid_config = sample_config.model_dump()
        invalid_config['model']['gpu_memory_utilization'] = 2.0
        with pytest.raises(ValueError):
            Trainer(config=ProjectConfig(**invalid_config), poc_mode=True)
    
    def test_memory_adjustment(self, sample_config):
        """Test memory utilization adjustment for small GPUs"""
        trainer = Trainer(config=sample_config, poc_mode=True)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_gb < 10:
            assert trainer.config.model.gpu_memory_utilization <= 0.8 