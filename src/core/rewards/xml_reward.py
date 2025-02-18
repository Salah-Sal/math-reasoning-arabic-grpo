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
            for key, value in config.items():
                if isinstance(value, dict) and key in merged_config:
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
            logger.debug(f"Merged config: {merged_config}")
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
        """Calculate reward for a single completion."""
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
        
        # Log each reward addition
        for tag, weight in weights.items():
            logger.debug(f"Processing tag {tag} with weight {weight}")
            if tag == "thinking_start" and strict_counts[tag] == 1:
                reward += weight
                logger.debug(f"Added thinking_start weight: +{weight} -> reward={reward}")
            elif tag == "thinking_end" and strict_counts[tag] == 1:
                reward += weight
                logger.debug(f"Added thinking_end weight: +{weight} -> reward={reward}")
            elif tag == "answer_start" and strict_counts[tag] == 1:
                reward += weight
                logger.debug(f"Added answer_start weight: +{weight} -> reward={reward}")
                # First extra content check
                extra_content_1 = text.split("\n</الجواب>\n")[-1]
                logger.debug(f"First extra content check: '{repr(extra_content_1)}'")
                if extra_content_1:
                    penalty_1 = len(extra_content_1) * penalties["extra_content"]
                    logger.debug(f"First penalty calculation: len={len(extra_content_1)} * {penalties['extra_content']} = {penalty_1}")
                    reward -= penalty_1
                    logger.debug(f"After first penalty: reward={reward}")
            elif tag == "answer_end" and strict_counts[tag] == 1:
                reward += weight
                logger.debug(f"Added answer_end weight: +{weight} -> reward={reward}")
                # Second extra content check
                extra_content_2 = text.split("\n</الجواب>")[-1]
                logger.debug(f"Second extra content check: '{repr(extra_content_2)}'")
                if len(extra_content_2) > 1:
                    penalty_2 = (len(extra_content_2) - 1) * penalties["extra_content"]
                    logger.debug(f"Second penalty calculation: (len={len(extra_content_2)}-1) * {penalties['extra_content']} = {penalty_2}")
                    reward -= penalty_2
                    logger.debug(f"After second penalty: reward={reward}")
            logger.debug(f"Current reward after {tag}: {reward}")
        
        # Log final reward before returning
        logger.debug(f"Final calculated reward: {reward}")
        normalized = self.normalize_reward(reward)
        logger.debug(f"Normalized reward: {normalized}")
        return normalized

        except Exception as e:
            logger.error(f"Error in reward calculation: {str(e)}")
            return 0.0 