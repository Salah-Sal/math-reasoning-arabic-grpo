# Finetuning Qwen2.5-1.5B model to be a math reasoning model using GRPO
Arabic Math Problem Solver using Qwen2.5-1.5B-GRPO
Project Overview
This project involves fine-tuning the Qwen2.5-1.5B-Instruct language model to solve mathematical problems in Arabic using reinforcement learning, specifically the GRPO (Generative Reinforcement with Preference Optimization) approach. The model is trained to provide structured responses with reasoning steps and numerical answers in Arabic.
Key Features

Fine-tuned Qwen2.5-1.5B model for Arabic mathematical reasoning
GRPO-based training pipeline with specialized Arabic reward functions
Support for structured output with Arabic reasoning tags
Evaluation system for response quality and correctness

Technical Infrastructure
Development Environment

Platform: Windows 11 with WSL2 (Ubuntu 20.04.6 LTS)
GPU: NVIDIA GeForce RTX 4060 Ti (8GB VRAM)
ML Framework: PyTorch with CUDA support

Key Libraries

unsloth: Model optimization
transformers: Base transformer functionality
trl: GRPO implementation
torch: Deep learning framework
vllm: Inference optimization

Project Structure
math-reasoning-arabic-grpo/
├── src/
│   ├── core/              # Core business logic and model interfaces
│   ├── data/              # Dataset handling and preprocessing
│   ├── infrastructure/    # Training and model infrastructure
│   └── utils/             # Utility functions
├── tests/                 # Test suites
├── configs/               # Configuration files
├── notebooks/            # Jupyter notebooks for experiments
└── scripts/              # Utility scripts

## Testing Setup

### Overview
The project follows a test-driven development (TDD) approach with a comprehensive testing structure using pytest.

### Test Structure
```
tests/
├── unit/                  # Unit tests for individual components
│   ├── core/             # Tests for core business logic
│   ├── data/             # Tests for data handling
│   └── infrastructure/   # Tests for infrastructure components
├── integration/          # Integration tests
└── fixtures/             # Test fixtures and data
```

### Running Tests
1. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   # or
   pip install -r requirements.txt
   ```

2. Run the test suite:
   ```bash
   pytest                  # Run all tests
   pytest -v              # Verbose output
   pytest -k "test_name"  # Run specific test
   pytest --cov=src       # Run tests with coverage report
   ```

### Test Configuration
- Tests are configured using `pytest.ini`
- Coverage reports are automatically generated
- Test fixtures are located in `tests/fixtures/`

### Writing Tests
When contributing new features:
1. Create test files in the appropriate directory under `tests/`
2. Follow the naming convention: `test_*.py` for test files
3. Use fixtures from `tests/fixtures/` or create new ones if needed
4. Ensure tests are properly documented
5. Verify coverage for new code

### Example Test Structure
```python
# tests/unit/your_module/test_feature.py

import pytest
from src.your_module import YourFeature

@pytest.fixture
def feature_fixture():
    return YourFeature()

def test_feature_behavior(feature_fixture):
    """Test description"""
    # Test implementation
    assert feature_fixture.method() == expected_result
```

### Coverage Requirements
- Minimum coverage requirement: 90%
- Run coverage report: `pytest --cov=src --cov-report=term-missing`
- Coverage reports show:
  - Line coverage
  - Missing lines
  - Branch coverage

### Continuous Integration
- Tests run automatically on pull requests
- All tests must pass before merging
- Coverage reports are generated for each PR

For more detailed information about testing specific components, refer to the documentation in each test module.

## Logging System

### Overview
The project implements a hierarchical logging system that provides both console and file-based logging with different verbosity levels.

### Log Configuration
```yaml
# configs/logging_config.yaml
default:
  formatters:
    standard:    # For console output
      format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    detailed:    # For file output
      format: '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
```

### Log Levels
- DEBUG: Detailed debugging information
- INFO: General operational events
- WARNING: Warning messages for non-critical issues
- ERROR: Error events that might still allow the application to continue
- CRITICAL: Critical events that may lead to termination

### Using the Logger
```python
from src.infrastructure.logging import get_logger

# Create a logger for your module
logger = get_logger("your_module_name")

# Example usage
logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical error")
```

### Log Files
- Location: `logs/training.log`
- Rotation: 10MB file size with 5 backup files
- Format: Detailed format including timestamp, module, level, file location

### Console Output
- Shows INFO level and above
- Simplified format for better readability

### Adding Logging to New Components
When contributing new features:
1. Import the logger:
   ```python
   from src.infrastructure.logging import get_logger
   ```

2. Create a module-specific logger:
   ```python
   logger = get_logger(__name__)  # Uses the module name
   ```

3. Use appropriate log levels:
   ```python
   # Debug: Detailed information for debugging
   logger.debug("Processing item %s with parameters %s", item_id, params)
   
   # Info: Confirmation that things are working as expected
   logger.info("Training epoch %d completed", epoch)
   
   # Warning: Indication that something unexpected happened
   logger.warning("Memory usage above 80%%, consider reducing batch size")
   
   # Error: The software has not been able to perform some function
   logger.error("Failed to load model: %s", str(error))
   ```

### Customizing Logging
To modify logging behavior:
1. Edit `configs/logging_config.yaml`
2. Adjust log levels, formats, or handlers
3. Changes take effect after restarting the application

### Best Practices
- Use appropriate log levels
- Include relevant context in log messages
- Add structured data when possible
- Avoid logging sensitive information
- Use format strings instead of concatenation

For more detailed information about logging specific components, refer to the documentation in each module.

## Data Module

### Overview
The data module handles loading and processing of Arabic mathematical problems from JSON files. It provides a clean interface for accessing the dataset and converting it to formats needed for training.

### Dataset Structure
```json
{
  "id": "problem_001",
  "original": {
    "question": "English math question...",
    "answer": "English solution... #### 42"
  },
  "translation": {
    "question": "Arabic math question...",
    "answer": "Arabic solution... #### ٤٢"
  },
  "metadata": {
    "timestamp": "2024-02-20T10:00:00",
    "status": "completed"
  }
}
```

### Using the Dataset
```python
from src.data.dataset import ArabicMathDataset

# Initialize dataset
dataset = ArabicMathDataset(
    data_dir="/path/to/data",
    system_prompt="Optional custom system prompt"
)

# Access data
print(f"Dataset size: {len(dataset)}")
example = dataset[0]

# Convert to HuggingFace dataset
hf_dataset = dataset.to_huggingface_dataset()
```

### Data Processing
The dataset class handles:
- JSON file loading and validation
- Arabic text processing
- Answer extraction and normalization
- Arabic to English numeral conversion
- Training format preparation

### Output Format
Each dataset item is formatted as:
```python
{
    'prompt': [
        {
            'role': 'system',
            'content': 'System prompt in Arabic'
        },
        {
            'role': 'user',
            'content': 'Arabic math question'
        }
    ],
    'answer': 'Numerical answer'
}
```

### Adding New Data
When contributing new problem files:
1. Follow the JSON structure shown above
2. Place files in the data directory
3. Ensure answers are marked with #### delimiter
4. Include both English and Arabic versions
5. Validate data format using dataset validation

### Best Practices
- Use UTF-8 encoding for all files
- Include step-by-step solutions in answers
- Maintain consistent formatting
- Handle Arabic numerals appropriately
- Test with sample data before adding large datasets

For more detailed information about data processing and formats, refer to the documentation in the data module.

