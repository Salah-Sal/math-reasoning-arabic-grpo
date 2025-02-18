from typing import List, Dict, Any, Optional
import re
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

class RewardHandler:
    """Handles reward calculation for Arabic math reasoning responses."""

    DEFAULT_CONFIG = {
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

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the reward handler.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = self._normalize_config(config or self.DEFAULT_CONFIG.copy())
        logger.debug(f"Initialized RewardHandler with config: {self.config}")

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize configuration, ensuring weights sum to 1.0."""
        if "weights" in config:
            weights = config["weights"]
            total = sum(weights.values())
            if total != 0:
                config["weights"] = {k: v/total for k, v in weights.items()}
        return config

    def calculate_rewards(self, completions: List[Dict[str, str]], expected_answer: str) -> List[float]:
        """Calculate combined rewards for completions.
        
        Args:
            completions: List of completion dictionaries
            expected_answer: Expected numerical answer
            
        Returns:
            List of float rewards
        """
        weights = self.config["weights"]
        
        # Calculate individual rewards
        xml_rewards = self.calculate_xml_reward(completions)
        format_rewards = self.calculate_format_reward(completions)
        correctness_rewards = self.calculate_correctness_reward(completions, expected_answer)
        
        # Combine rewards
        combined_rewards = []
        for xml_r, fmt_r, corr_r in zip(xml_rewards, format_rewards, correctness_rewards):
            reward = (
                xml_r * weights["xml_structure"] +
                fmt_r * weights["format"] +
                corr_r * weights["correctness"]
            )
            combined_rewards.append(min(1.0, max(0.0, reward)))
            
        return combined_rewards

    def calculate_xml_reward(self, completions: List[Dict[str, str]]) -> List[float]:
        """Calculate reward for XML structure.
        
        Args:
            completions: List of completion dictionaries
            
        Returns:
            List of float rewards
        """
        rewards = []
        for completion in completions:
            try:
                text = completion["content"]
                reward = 0.0
                
                # Check for required tags
                if text.count("<تفكير>\n") == 1:
                    reward += 0.25
                if text.count("\n</تفكير>\n") == 1:
                    reward += 0.25
                if text.count("\n<الجواب>\n") == 1:
                    reward += 0.25
                    # Check for extra content
                    extra = text.split("\n</الجواب>\n")[-1]
                    reward -= len(extra) * self.config["penalties"]["extra_content"]
                if text.count("\n</الجواب>") == 1:
                    reward += 0.25
                
                rewards.append(max(0.0, min(1.0, reward)))
            except Exception as e:
                logger.error(f"Error calculating XML reward: {str(e)}")
                rewards.append(0.0)
                
        return rewards

    def calculate_format_reward(self, completions: List[Dict[str, str]]) -> List[float]:
        """Calculate reward for format adherence.
        
        Args:
            completions: List of completion dictionaries
            
        Returns:
            List of float rewards
        """
        pattern = r"<تفكير>.*?</تفكير>\s*<الجواب>.*?</الجواب>"
        rewards = []
        
        for completion in completions:
            try:
                text = completion["content"]
                reward = 1.0 if re.match(pattern, text, re.DOTALL) else 0.0
                rewards.append(reward)
            except Exception as e:
                logger.error(f"Error calculating format reward: {str(e)}")
                rewards.append(0.0)
                
        return rewards

    def calculate_correctness_reward(self, completions: List[Dict[str, str]], expected_answer: str) -> List[float]:
        """Calculate reward for answer correctness.
        
        Args:
            completions: List of completion dictionaries
            expected_answer: Expected numerical answer
            
        Returns:
            List of float rewards
        """
        rewards = []
        
        for completion in completions:
            try:
                text = completion["content"]
                # Extract answer from between tags
                answer = text.split("<الجواب>")[-1].split("</الجواب>")[0].strip()
                
                # Convert Arabic numerals if present
                arabic_to_english = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
                answer = answer.translate(arabic_to_english)
                
                # Extract only digits
                answer = ''.join(c for c in answer if c.isdigit())
                
                reward = 1.0 if answer == expected_answer else 0.0
                rewards.append(reward)
            except Exception as e:
                logger.error(f"Error calculating correctness reward: {str(e)}")
                rewards.append(0.0)
                
        return rewards 