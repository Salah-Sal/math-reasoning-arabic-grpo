import pytest
from src.core.rewards.xml_reward import ArabicXMLReward

@pytest.fixture
def reward_function():
    return ArabicXMLReward()

def test_initialization():
    """Test reward function initialization."""
    reward = ArabicXMLReward()
    assert reward.config["tag_weights"]["thinking_start"] == 0.125
    assert reward.config["tag_weights"]["thinking_end"] == 0.125
    assert reward.config["tag_weights"]["answer_start"] == 0.125
    assert reward.config["tag_weights"]["answer_end"] == 0.125
    assert reward.config["penalties"]["extra_content"] == 0.001

def test_custom_config():
    """Test initialization with custom config."""
    custom_config = {
        "tag_weights": {
            "thinking_start": 0.2
        }
    }
    reward = ArabicXMLReward(config=custom_config)
    assert reward.config["tag_weights"]["thinking_start"] == 0.2
    # Other defaults should remain
    assert reward.config["tag_weights"]["thinking_end"] == 0.125

def test_perfect_format(reward_function):
    """Test reward calculation for perfect format."""
    completion = {
        "content": "<تفكير>\nsome thinking\n</تفكير>\n<الجواب>\n42\n</الجواب>"
    }
    rewards = reward_function.calculate([completion])
    assert len(rewards) == 1
    assert rewards[0] == 0.5  # All tags present = 0.125 * 4

def test_missing_tags(reward_function):
    """Test reward calculation with missing tags."""
    completion = {
        "content": "<تفكير>\nsome thinking\n</تفكير>"
    }
    rewards = reward_function.calculate([completion])
    assert len(rewards) == 1
    assert rewards[0] == 0.25  # Only thinking tags = 0.125 * 2

def test_extra_content_penalty(reward_function):
    """Test penalty for content after closing tags."""
    completion = {
        "content": "<تفكير>\nthinking\n</تفكير>\n<الجواب>\n42\n</الجواب>\nextra content"
    }
    rewards = reward_function.calculate([completion])
    assert len(rewards) == 1
    assert rewards[0] < 0.5  # Should be less than perfect due to penalty

def test_multiple_completions(reward_function):
    """Test processing multiple completions."""
    completions = [
        {"content": "<تفكير>\nthinking1\n</تفكير>\n<الجواب>\n42\n</الجواب>"},
        {"content": "<تفكير>\nthinking2\n</تفكير>"},
    ]
    rewards = reward_function.calculate(completions)
    assert len(rewards) == 2
    assert rewards[0] > rewards[1]  # First should have higher reward

def test_invalid_input(reward_function):
    """Test handling of invalid input."""
    invalid_completions = [
        {"wrong_key": "content"},
        {"content": 42}  # Non-string content
    ]
    rewards = reward_function.calculate(invalid_completions)
    assert len(rewards) == 2
    assert all(r == 0.0 for r in rewards)

def test_malformed_tags(reward_function):
    """Test handling of malformed tags."""
    completion = {
        "content": "<تفكير>thinking</تفكير><الجواب>42</الجواب>"  # Missing newlines
    }
    rewards = reward_function.calculate([completion])
    assert len(rewards) == 1
    assert rewards[0] == 0.0  # Should get no reward for incorrect format

def test_empty_content(reward_function):
    """Test handling of empty content."""
    completion = {"content": ""}
    rewards = reward_function.calculate([completion])
    assert len(rewards) == 1
    assert rewards[0] == 0.0

def test_duplicate_tags(reward_function):
    """Test handling of duplicate tags."""
    completion = {
        "content": "<تفكير>\nthink\n</تفكير>\n<تفكير>\nmore\n</تفكير>\n<الجواب>\n42\n</الجواب>"
    }
    rewards = reward_function.calculate([completion])
    assert len(rewards) == 1
    assert rewards[0] < 0.5  # Should get partial reward but not full 