import pytest
from typing import List, Dict, Any
import logging
import torch

logger = logging.getLogger(__name__)

@pytest.fixture
def default_config():
    """Default configuration for reward handler."""
    return {
        "weights": {
            "xml_structure": 0.3,
            "format": 0.3,
            "correctness": 0.4
        },
        "penalties": {
            "extra_content": 0.001,
            "multiple_tags": 0.05
        },
        "cache_size": 1000
    }

@pytest.fixture
def sample_completion():
    """Sample completion with proper Arabic XML format."""
    return [{
        "content": "<تفكير>\nخطوة 1: تحليل المسألة\nخطوة 2: حل المعادلة\n</تفكير>\n<الجواب>\n42\n</الجواب>"
    }]

@pytest.fixture
def sample_completion_batch():
    """Batch of sample completions with varying formats."""
    return [
        [{"content": "<تفكير>\nخطوة 1\n</تفكير>\n<الجواب>\n42\n</الجواب>"}],
        [{"content": "<تفكير>\nخطوة 1\n</تفكير>\n<الجواب>\n24\n</الجواب>"}],
        [{"content": "invalid format"}],
        [{"content": "<تفكير>\nمحاولة\n</تفكير>\n<الجواب>\n٤٢\n</الجواب>"}]  # Arabic numerals
    ]

@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return [
        [
            {"role": "system", "content": "أنت مساعد ذكي..."},
            {"role": "user", "content": "حل المسألة التالية..."}
        ]
    ]

def test_reward_handler_initialization(default_config):
    """Test RewardHandler initialization with default and custom configs."""
    from src.training.reward_handler import RewardHandler
    
    # Test default initialization
    handler = RewardHandler()
    assert handler.config is not None
    assert all(k in handler.config["weights"] for k in ["xml_structure", "format", "correctness"])
    assert abs(sum(handler.config["weights"].values()) - 1.0) < 1e-6
    
    # Test custom config
    custom_config = default_config.copy()
    original_xml_weight = 0.5
    custom_config["weights"]["xml_structure"] = original_xml_weight
    handler = RewardHandler(config=custom_config)
    
    # Verify weights are normalized but proportions are maintained
    total = sum(custom_config["weights"].values())
    expected_normalized = original_xml_weight / total
    assert abs(handler.config["weights"]["xml_structure"] - expected_normalized) < 1e-6
    assert abs(sum(handler.config["weights"].values()) - 1.0) < 1e-6
    
    # Verify relative proportions are maintained
    weights = handler.config["weights"]
    assert weights["xml_structure"] > weights["format"]
    assert weights["xml_structure"] > weights["correctness"]

def test_xml_reward_calculation(sample_completion):
    """Test XML structure reward calculation."""
    from src.training.reward_handler import RewardHandler
    handler = RewardHandler()
    
    # Test perfect format
    reward = handler.calculate_xml_reward(sample_completion)
    assert len(reward) == 1
    assert 0.0 <= reward[0] <= 1.0
    assert reward[0] > 0.4  # Should get good reward for proper structure
    
    # Test missing tags
    bad_completion = [{"content": "<تفكير>\nsome text"}]
    reward = handler.calculate_xml_reward(bad_completion)
    assert reward[0] < 0.2  # Should get low reward for missing tags
    
    # Test extra content penalty
    extra_content = [{"content": sample_completion[0]["content"] + "\nextra stuff"}]
    reward = handler.calculate_xml_reward(extra_content)
    assert reward[0] < handler.calculate_xml_reward(sample_completion)[0]

def test_format_reward_calculation(sample_completion, sample_completion_batch):
    """Test format adherence reward calculation."""
    from src.training.reward_handler import RewardHandler
    handler = RewardHandler()
    
    # Test single perfect format
    reward = handler.calculate_format_reward(sample_completion)
    assert len(reward) == 1
    assert reward[0] == 1.0  # Should get full reward for perfect format
    
    # Test batch processing
    rewards = handler.calculate_format_reward(sample_completion_batch)
    assert len(rewards) == len(sample_completion_batch)
    assert rewards[0] == 1.0  # First example is perfect
    assert rewards[2] == 0.0  # Third example is invalid

def test_correctness_reward_calculation(sample_completion):
    """Test answer correctness reward calculation."""
    from src.training.reward_handler import RewardHandler
    handler = RewardHandler()
    
    # Test correct answer
    reward = handler.calculate_correctness_reward(
        completions=sample_completion,
        expected_answer="42"
    )
    assert len(reward) == 1
    assert reward[0] == 1.0
    
    # Test incorrect answer
    reward = handler.calculate_correctness_reward(
        completions=sample_completion,
        expected_answer="24"
    )
    assert reward[0] == 0.0
    
    # Test Arabic numerals
    arabic_completion = [{"content": "<تفكير>\nتحليل\n</تفكير>\n<الجواب>\n٤٢\n</الجواب>"}]
    reward = handler.calculate_correctness_reward(
        completions=arabic_completion,
        expected_answer="42"
    )
    assert reward[0] == 1.0

def test_combined_reward_calculation(sample_completion):
    """Test combined reward calculation with weights."""
    from src.training.reward_handler import RewardHandler
    handler = RewardHandler()
    
    rewards = handler.calculate_rewards(
        completions=sample_completion,
        expected_answer="42"
    )
    
    assert len(rewards) == 1
    assert 0.0 <= rewards[0] <= 1.0
    
    # Perfect completion should get high combined reward
    assert rewards[0] > 0.8

def test_batch_processing():
    """Test batch processing of completions."""
    pytest.skip("Not implemented yet")

def test_arabic_numeral_handling():
    """Test handling of Arabic numerals in answers."""
    pytest.skip("Not implemented yet")

def test_invalid_input_handling():
    """Test handling of invalid inputs."""
    pytest.skip("Not implemented yet")

def test_memory_efficiency():
    """Test memory usage during reward calculation."""
    pytest.skip("Not implemented yet")

def test_reward_normalization():
    """Test reward normalization and scaling."""
    pytest.skip("Not implemented yet")

def test_custom_reward_weights():
    """Test custom weight configuration for rewards."""
    pytest.skip("Not implemented yet")

def test_edge_cases():
    """Test edge cases like empty input, malformed XML, etc."""
    pytest.skip("Not implemented yet")

def test_answer_extraction():
    """Test numerical answer extraction from Arabic text."""
    pytest.skip("Not implemented yet")

def test_reward_caching():
    """Test caching mechanism for reward calculations."""
    pytest.skip("Not implemented yet")

def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize configuration, ensuring weights sum to 1.0."""
    logger.debug(f"Input config weights: {config.get('weights', {})}")
    if "weights" in config:
        weights = config["weights"]
        total = sum(weights.values())
        logger.debug(f"Total weight sum before normalization: {total}")
        if total != 0:
            config["weights"] = {k: v/total for k, v in weights.items()}
            logger.debug(f"Normalized weights: {config['weights']}")
    return config 

def calculate_format_reward(self, completions: List[Dict[str, str]]) -> List[float]:
    """Calculate reward for format adherence."""
    logger.debug(f"Input completions structure: {type(completions)}")
    logger.debug(f"First completion sample: {completions[0] if completions else None}")
    
    rewards = []
    for completion in completions:
        try:
            logger.debug(f"Processing completion: {completion}")
            text = completion["content"]
            # ... rest of the method
        except Exception as e:
            logger.error(f"Error calculating format reward: {str(e)}")
            rewards.append(0.0)
                
    return rewards 