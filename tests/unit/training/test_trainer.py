import pytest
from pathlib import Path
from src.training.trainer import Trainer
from src.data.dataset import ArabicMathDataset
from src.infrastructure.config import ProjectConfig, ModelConfig, TrainingConfig, DataConfig
from src.infrastructure.logging import get_logger
import torch

logger = get_logger(__name__)

@pytest.fixture
def sample_config(tmp_path):
    """Create a sample configuration for testing"""
    config = ProjectConfig(
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
            max_prompt_length=256,
            max_completion_length=128,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type='cosine',
            optim='adamw_8bit',
            logging_steps=10,
            save_steps=500
        ),
        data=DataConfig(
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "outputs",
            cache_dir=tmp_path / "cache"
        ),
        seed=3407,
        device="cuda"
    )
    return config

@pytest.fixture
def sample_dataset(tmp_path):
    """Create a tiny dataset for testing"""
    data_dir = tmp_path / "sample_data"
    data_dir.mkdir()
    
    sample_file = tmp_path / "sample_problems" / "problem_001.json"
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_text('{"translation": {"question": "Test question?", "answer": "Answer #### 42"}}')
    
    return ArabicMathDataset(data_dir=data_dir)

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available"
)
class TestTrainer:
    """Test suite for Trainer class"""
    
    def test_initialization(self, sample_config, sample_dataset):
        """Test trainer initialization and model loading"""
        trainer = Trainer(config=sample_config, poc_mode=True)
        
        # Basic initialization
        assert trainer.poc_mode is True
        assert trainer.model is not None
        assert trainer.tokenizer is not None
        
        # Config verification
        assert trainer.config.model.load_in_4bit is True
        assert trainer.config.model.max_seq_length <= 384
        
        # Trainer's adjusted config should have specific settings
        assert trainer.config.training.per_device_train_batch_size == sample_config.training.per_device_train_batch_size
        assert trainer.config.training.gradient_accumulation_steps == sample_config.training.gradient_accumulation_steps
        
        # Check LoRA setup in attention layers
        has_lora = False
        for name, module in trainer.model.named_modules():
            if any(target in name for target in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                if hasattr(module, 'lora_A'):
                    has_lora = True
                    break
        assert has_lora, "Model should have LoRA layers"
        assert trainer.model.config.model_type == "qwen2"
    
    def test_memory_management(self, sample_config):
        """Test GPU memory management"""
        if torch.cuda.get_device_properties(0).total_memory < 8 * 1024**3:
            pytest.skip("Not enough GPU memory")
        
        initial_memory = torch.cuda.memory_allocated()
        
        # Load model
        trainer = Trainer(config=sample_config, poc_mode=True)
        loaded_memory = torch.cuda.memory_allocated()
        assert loaded_memory > initial_memory, "Model should allocate more GPU memory"
        
        # Cleanup
        del trainer
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        assert final_memory <= initial_memory, "Memory should be freed"
    
    def test_dataset_preparation(self, sample_config, sample_dataset):
        """Test dataset preparation and validation"""
        trainer = Trainer(config=sample_config, poc_mode=True)
        poc_dataset = trainer.prepare_dataset(sample_dataset)
        assert len(poc_dataset) <= 100, "POC should limit dataset size"
        
        # Verify dataset format
        sample = poc_dataset[0]
        assert 'prompt' in sample, "Dataset should have prompt field"
        assert 'answer' in sample, "Dataset should have answer field"
        assert len(sample['prompt']) == 2, "Prompt should have system and user messages"
        
        # Test full mode
        trainer_full = Trainer(config=sample_config, poc_mode=False)
        full_dataset = trainer_full.prepare_dataset(sample_dataset)
        assert len(full_dataset) == len(sample_dataset), "Full mode should use entire dataset"
    
    def test_config_validation(self, sample_config):
        """Test configuration validation"""
        # Test invalid memory utilization
        with pytest.raises(ValueError) as exc_info:
            ProjectConfig(
                model=sample_config.model.copy(update={"gpu_memory_utilization": 2.0}),
                training=sample_config.training,
                data=sample_config.data,
                seed=sample_config.seed,
                device=sample_config.device
            )
        error_message = str(exc_info.value)
        assert 'gpu_memory_utilization must be between 0 and 1' in error_message
        
        # Test invalid max_seq_length
        with pytest.raises(ValueError) as exc_info:
            ProjectConfig(
                model=sample_config.model.copy(update={"max_seq_length": -1}),
                training=sample_config.training,
                data=sample_config.data,
                seed=sample_config.seed,
                device=sample_config.device
            )
        error_message = str(exc_info.value)
        assert 'Maximum sequence length must be positive' in error_message
        
        # Test invalid batch size
        with pytest.raises(ValueError) as exc_info:
            ProjectConfig(
                model=sample_config.model,
                training=sample_config.training.copy(update={"per_device_train_batch_size": 0}),
                data=sample_config.data,
                seed=sample_config.seed,
                device=sample_config.device
            )
        error_message = str(exc_info.value)
        assert 'Batch size must be positive' in error_message.lower() 