from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

class BaseReward(ABC):
    """Base class for reward functions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize reward function with optional configuration.
        
        Args:
            config: Configuration dictionary for the reward function
        """
        self.config = config or {}
        logger.info(f"Initializing {self.__class__.__name__} with config: {self.config}")

    @abstractmethod
    def calculate(self, completions: List[Dict[str, Any]], **kwargs) -> List[float]:
        """Calculate rewards for a batch of completions.
        
        Args:
            completions: List of completion dictionaries, each containing at least
                       a 'content' key with the model's response
            **kwargs: Additional arguments specific to the reward function
            
        Returns:
            List of float rewards, one for each completion
        """
        pass

    def validate_input(self, completions: List[Dict[str, Any]]) -> bool:
        """Validate the input format of completions.
        
        Args:
            completions: List of completion dictionaries to validate
            
        Returns:
            True if input format is valid, False otherwise
        """
        try:
            if not isinstance(completions, list):
                logger.error("Completions must be a list")
                return False
            
            if not completions:
                logger.error("Completions list is empty")
                return False
            
            for completion in completions:
                if not isinstance(completion, dict):
                    logger.error(f"Completion must be a dictionary, got {type(completion)}")
                    return False
                
                if "content" not in completion:
                    logger.error("Completion missing 'content' key")
                    return False
                
                if not isinstance(completion["content"], str):
                    logger.error(f"Completion content must be a string, got {type(completion['content'])}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating input: {str(e)}")
            return False

    def normalize_reward(self, raw_reward: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalize reward to specified range.
        
        Args:
            raw_reward: Raw reward value to normalize
            min_val: Minimum value for normalized reward
            max_val: Maximum value for normalized reward
            
        Returns:
            Normalized reward value
        """
        try:
            normalized = max(min_val, min(max_val, raw_reward))
            logger.debug(f"Normalized reward from {raw_reward} to {normalized}")
            return normalized
        except Exception as e:
            logger.error(f"Error normalizing reward: {str(e)}")
            return 0.0

    def log_reward_calculation(self, completion: Dict[str, Any], reward: float) -> None:
        """Log detailed information about reward calculation.
        
        Args:
            completion: The completion dictionary being evaluated
            reward: The calculated reward value
        """
        try:
            content_preview = completion.get("content", "")[:100] + "..."
            logger.debug(f"Reward calculation for content: {content_preview}")
            logger.debug(f"Calculated reward: {reward}")
        except Exception as e:
            logger.error(f"Error logging reward calculation: {str(e)}") 