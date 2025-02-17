import pytest
from pathlib import Path
from src.training.trainer import Trainer
from src.data.dataset import ArabicMathDataset
from src.infrastructure.config import ProjectConfig
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

@pytest.fixture
def sample_config(tmp_path):
    """Create a sample configuration for POC training"""
    return ProjectConfig(
        model=dict(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            max_seq_length=384,
            load_in_4bit=True,
            fast_inference=True,
            max_lora_rank=8,
            gpu_memory_utilization=0.5
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

def test_trainer_initialization(sample_config, sample_dataset):
    """Test basic trainer initialization"""
    trainer = Trainer(config=sample_config, poc_mode=True)
    assert trainer.poc_mode == True
    assert trainer.model is not None
    assert trainer.tokenizer is not None

def test_poc_dataset_size(sample_config, sample_dataset):
    """Test that POC mode uses a smaller dataset"""
    trainer = Trainer(config=sample_config, poc_mode=True)
    assert len(trainer.prepare_dataset(sample_dataset)) <= 100  # POC should limit dataset size 