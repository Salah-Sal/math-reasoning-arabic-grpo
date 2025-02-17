import pytest
from pathlib import Path
from src.data.dataset import ArabicMathDataset
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temporary directory with sample data"""
    data_dir = tmp_path / "sample_data"
    data_dir.mkdir()
    
    # Copy sample problem from fixtures to temp directory
    sample_file = Path(__file__).parent.parent.parent / "fixtures" / "data" / "sample_problems" / "problem_001.json"
    if not sample_file.exists():
        pytest.skip("Sample data file not found")
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(data_dir / "problem_001.json", 'w', encoding='utf-8') as f:
        f.write(content)
    
    return data_dir

def test_dataset_loading(sample_data_dir):
    """Test basic dataset loading functionality"""
    dataset = ArabicMathDataset(data_dir=sample_data_dir)
    assert len(dataset) == 1
    
    # Test first item
    item = dataset[0]
    assert 'prompt' in item
    assert 'answer' in item
    assert isinstance(item['answer'], str)
    assert len(item['prompt']) == 2  # System prompt and user question
    
def test_dataset_validation(sample_data_dir):
    """Test dataset validation"""
    dataset = ArabicMathDataset(data_dir=sample_data_dir)
    assert dataset.is_valid()
    
def test_arabic_numeral_conversion(sample_data_dir):
    """Test Arabic to English numeral conversion"""
    dataset = ArabicMathDataset(data_dir=sample_data_dir)
    item = dataset[0]
    assert item['answer'] == '3'  # Should convert ٣ to 3 