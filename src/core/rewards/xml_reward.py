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
            # Log the merging process
            logger.debug(f"Original config: {merged_config}")
            logger.debug(f"Custom config: {config}")
            merged_config.update(config)
            logger.debug(f"Merged config: {merged_config}")
        self.config = merged_config

    def calculate(self, completions: List[Dict[str, Any]]) -> List[float]:
        """Calculate rewards for a list of completions.
        
        Args:
            completions: List of completion dictionaries with 'content' key
            
        Returns:
            List of reward values between 0 and 1
        """
        try:
            if not self.validate_input(completions):
                logger.error("Invalid input format")
                return [0.0] * len(completions)
            
            rewards = []
            for completion in completions:
                reward = self._calculate_single_reward(completion["content"])
                rewards.append(reward)
                logger.debug(f"Calculated reward {reward} for completion")
            
            return rewards
            
        except Exception as e:
            logger.error(f"Error calculating rewards: {str(e)}")
            return [0.0] * len(completions)

    def _calculate_single_reward(self, text: str) -> float:
        """Calculate reward for a single completion."""
        try:
            reward = 0.0
            weights = self.config["tag_weights"]
            penalties = self.config["penalties"]

            # Add detailed logging
            logger.debug("Starting reward calculation for text:")
            logger.debug(f"Text length: {len(text)}")
            logger.debug(f"Raw text: {repr(text)}")  # Show explicit newlines
            
            # Log tag counts with different matching methods
            strict_counts = {
                "thinking_start": text.count("<تفكير>\n"),
                "thinking_end": text.count("\n</تفكير>\n"),
                "answer_start": text.count("<الجواب>\n"),
                "answer_end": text.count("\n</الجواب>\n")
            }
            logger.debug(f"Strict tag counts (with newlines): {strict_counts}")
            
            loose_counts = {
                "thinking_start": text.count("<تفكير>"),
                "thinking_end": text.count("</تفكير>"),
                "answer_start": text.count("<الجواب>"),
                "answer_end": text.count("</الجواب>")
            }
            logger.debug(f"Loose tag counts (without newlines): {loose_counts}")
            
            # Calculate rewards based on tag presence
            for tag, weight in weights.items():
                logger.debug(f"Processing tag {tag} with weight {weight}")
                if strict_counts[tag] == 1:
                    reward += weight
                    logger.debug(f"Added weight {weight} for {tag}, current reward: {reward}")
            
            # Apply penalties for multiple tags or missing tags
            for tag, count in strict_counts.items():
                if count > 1:
                    penalty = (count - 1) * penalties["extra_content"]
                    reward -= penalty
                    logger.debug(f"Applied penalty {penalty} for multiple {tag}, current reward: {reward}")
            
            logger.debug(f"Final calculated reward: {reward}")
            return max(0.0, min(1.0, reward))  # Ensure reward is between 0 and 1
            
        except Exception as e:
            logger.error(f"Error in _calculate_single_reward: {str(e)}")
            return 0.0 