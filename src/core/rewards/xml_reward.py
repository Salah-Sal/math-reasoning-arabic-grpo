from typing import List, Dict, Any
from src.core.rewards.base import BaseReward
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

class ArabicXMLReward(BaseReward):
    """Reward function for Arabic XML structure validation."""

    DEFAULT_CONFIG = {
        "tag_weights": {
            "thinking_start": 0.125,  # <تفكير>
            "thinking_end": 0.125,    # </تفكير>
            "answer_start": 0.125,    # <الجواب>
            "answer_end": 0.125,      # </الجواب>
        },
        "penalties": {
            "extra_content": 0.001,  # Penalty per character after closing tags
        }
    }

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with configuration.
        
        Args:
            config: Configuration dictionary, will be merged with DEFAULT_CONFIG
        """
        merged_config = self.DEFAULT_CONFIG.copy()
        if config:
            merged_config.update(config)
        super().__init__(merged_config)
        logger.info("Initialized ArabicXMLReward with merged config")

    def calculate(self, completions: List[Dict[str, Any]], **kwargs) -> List[float]:
        """Calculate XML structure rewards for completions.
        
        Args:
            completions: List of completion dictionaries
            **kwargs: Additional arguments (unused)
            
        Returns:
            List of float rewards
        """
        if not self.validate_input(completions):
            logger.error("Input validation failed")
            return [0.0] * len(completions)

        rewards = []
        for completion in completions:
            try:
                reward = self._calculate_single_reward(completion["content"])
                self.log_reward_calculation(completion, reward)
                rewards.append(reward)
            except Exception as e:
                logger.error(f"Error calculating reward: {str(e)}")
                rewards.append(0.0)

        return rewards

    def _calculate_single_reward(self, text: str) -> float:
        """Calculate reward for a single completion.
        
        Args:
            text: The completion text to evaluate
            
        Returns:
            Float reward value
        """
        reward = 0.0
        weights = self.config["tag_weights"]
        penalties = self.config["penalties"]

        try:
            # Check thinking section tags
            if text.count("<تفكير>\n") == 1:
                reward += weights["thinking_start"]
                logger.debug("Found thinking start tag")
            if text.count("\n</تفكير>\n") == 1:
                reward += weights["thinking_end"]
                logger.debug("Found thinking end tag")

            # Check answer section tags
            if text.count("\n<الجواب>\n") == 1:
                reward += weights["answer_start"]
                logger.debug("Found answer start tag")
                # Apply penalty for content after answer
                extra_content = text.split("\n</الجواب>\n")[-1]
                reward -= len(extra_content) * penalties["extra_content"]
            if text.count("\n</الجواب>") == 1:
                reward += weights["answer_end"]
                logger.debug("Found answer end tag")
                # Apply penalty for content after final tag
                extra_content = text.split("\n</الجواب>")[-1]
                if len(extra_content) > 1:  # Allow for one newline
                    reward -= (len(extra_content) - 1) * penalties["extra_content"]

            # Normalize final reward
            return self.normalize_reward(reward)

        except Exception as e:
            logger.error(f"Error in reward calculation: {str(e)}")
            return 0.0 