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

