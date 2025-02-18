from typing import List, Dict, Any
import re
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
            "multiple_tags": 0.05,   # Penalty for duplicate tags
        }
    }

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with configuration."""
        self.config = self._merge_configs(self.DEFAULT_CONFIG, config or {})
        logger.debug(f"Initialized with config: {self.config}")

    def _merge_configs(self, default: Dict, custom: Dict) -> Dict:
        """Deep merge configuration dictionaries."""
        merged = default.copy()
        for key, value in custom.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

    def calculate(self, completions: List[Dict[str, Any]]) -> List[float]:
        """Calculate rewards for a list of completions."""
        try:
            # Handle None input
            if completions is None:
                logger.error("Received None as completions")
                return []
            
            if not self.validate_input(completions):
                logger.error("Invalid input format")
                return [0.0] * (len(completions) if isinstance(completions, list) else 0)
            
            rewards = []
            for completion in completions:
                reward = self._calculate_single_reward(completion["content"])
                rewards.append(reward)
                logger.debug(f"Calculated reward {reward} for completion")
            
            return rewards
            
        except Exception as e:
            logger.error(f"Error calculating rewards: {str(e)}")
            return [0.0] * (len(completions) if isinstance(completions, list) else 0)

    def _calculate_single_reward(self, text: str) -> float:
        """Calculate reward for a single completion."""
        try:
            reward = 0.0
            weights = self.config["tag_weights"]
            penalties = self.config["penalties"]

            logger.debug("=== Starting reward calculation ===")
            logger.debug(f"Input text: {repr(text)}")

            # Validate basic tag structure first
            if not self._validate_tag_structure(text):
                logger.debug("Invalid tag structure detected")
                return 0.0

            # Define tag patterns with strict matching
            patterns = {
                "thinking_start": r"<تفكير>\s*",
                "thinking_end": r"\s*</تفكير>",
                "answer_start": r"<الجواب>\s*",
                "answer_end": r"\s*</الجواب>"
            }

            # Find all tag matches
            tag_matches = {
                tag: list(re.finditer(pattern, text))
                for tag, pattern in patterns.items()
            }

            logger.debug(f"Found tag matches: {[(k, len(v)) for k, v in tag_matches.items()]}")

            # Validate tag ordering and completeness
            if not self._validate_tag_ordering(tag_matches):
                logger.debug("Invalid tag ordering detected")
                return 0.0

            # Calculate base reward for correct tag presence
            for tag, matches in tag_matches.items():
                match_count = len(matches)
                if match_count == 1:
                    reward += weights[tag]
                    logger.debug(f"Added weight {weights[tag]} for {tag}")
                elif match_count > 1:
                    penalty = (match_count - 1) * penalties["multiple_tags"]
                    reward -= penalty
                    logger.debug(f"Applied penalty {penalty} for multiple {tag}")

            # Check for extra content after final tag
            last_tag_pos = max(
                (m.end() for matches in tag_matches.values() for m in matches),
                default=0
            )
            extra_content = text[last_tag_pos:].strip()
            if extra_content:
                penalty = len(extra_content) * penalties["extra_content"]
                reward -= penalty
                logger.debug(f"Applied extra content penalty: {penalty} for {len(extra_content)} chars")

            # Normalize final reward
            final_reward = max(0.0, min(1.0, reward))
            logger.debug(f"Final reward (raw={reward}, normalized={final_reward})")
            
            return final_reward

        except Exception as e:
            logger.error(f"Error in reward calculation: {str(e)}")
            return 0.0

    def _validate_tag_structure(self, text: str) -> bool:
        """Validate basic XML tag structure."""
        # Check for matching opening/closing tags
        if text.count("<تفكير>") != text.count("</تفكير>"):
            return False
        if text.count("<الجواب>") != text.count("</الجواب>"):
            return False
        
        # Check for complete tag pairs
        if not (text.count("<تفكير>") >= 1 and text.count("</تفكير>") >= 1):
            return False
        if not (text.count("<الجواب>") >= 1 and text.count("</الجواب>") >= 1):
            return False
        
        return True

    def _validate_tag_ordering(self, tag_matches: Dict[str, List[re.Match]]) -> bool:
        """Validate the ordering of tags."""
        # All tags should appear exactly once
        if not all(len(matches) == 1 for matches in tag_matches.values()):
            return False
        
        # Get positions of all tags
        thinking_start = tag_matches["thinking_start"][0].start()
        thinking_end = tag_matches["thinking_end"][0].start()
        answer_start = tag_matches["answer_start"][0].start()
        answer_end = tag_matches["answer_end"][0].start()
        
        # Validate ordering
        return (thinking_start < thinking_end < answer_start < answer_end) 