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
        """Initialize the reward handler."""
        # Deep copy to prevent mutations
        base_config = config.copy() if config else self.DEFAULT_CONFIG.copy()
        self.config = self._normalize_config(base_config)
        logger.debug(f"Initialized RewardHandler with config: {self.config}")

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize configuration, ensuring weights sum to 1.0."""
        if "weights" in config:
            weights = config["weights"]
            total = sum(weights.values())
            logger.debug(f"Normalizing weights: {weights}, total: {total}")
            
            # Always normalize to ensure sum is 1.0
            if abs(total - 1.0) > 1e-6:
                # Store original proportions for logging
                original = weights.copy()
                config["weights"] = {k: v/total for k, v in weights.items()}
                logger.debug(f"Normalized weights from {original} to {config['weights']}")
                
                # Verify normalization
                new_total = sum(config["weights"].values())
                logger.debug(f"New total after normalization: {new_total}")
                assert abs(new_total - 1.0) < 1e-6, f"Normalization failed: {new_total}"
        
        return config

    def calculate_rewards(self, prompts: List[Dict[str, Any]], completions: List[Dict[str, Any]], **kwargs) -> List[float]:
        """Calculate rewards for completions.
        
        Args:
            prompts: List of prompt dictionaries containing the input context
            completions: List of completion dictionaries containing model responses
            **kwargs: Additional arguments for reward calculation
            
        Returns:
            List of float rewards
        """
        try:
            logger.info("=== Starting Reward Calculation ===")
            logger.info(f"Number of prompts: {len(prompts)}")
            logger.info(f"Number of completions: {len(completions)}")
            logger.debug(f"Additional kwargs: {kwargs}")
            
            # Input validation
            if not self._validate_inputs(prompts, completions):
                logger.error("Input validation failed")
                return [0.0] * len(completions)
            
            # Extract expected answers from prompts
            expected_answers = []
            for prompt in prompts:
                try:
                    # Log prompt structure for debugging
                    logger.debug(f"Prompt structure: {list(prompt.keys())}")
                    
                    # Extract answer from the last message if it's a list of messages
                    messages = prompt.get('prompt', [])
                    if isinstance(messages, list) and messages:
                        user_message = messages[-1].get('content', '')
                        # Extract answer from the Arabic text (assuming format includes #### separator)
                        answer = self._extract_answer_from_text(user_message)
                        expected_answers.append(answer)
                        logger.debug(f"Extracted expected answer: {answer}")
                    else:
                        logger.warning(f"Unexpected prompt format: {prompt}")
                        expected_answers.append("")
                except Exception as e:
                    logger.error(f"Error extracting answer from prompt: {str(e)}")
                    expected_answers.append("")
            
            # Calculate individual rewards
            rewards = []
            for completion, expected_answer in zip(completions, expected_answers):
                try:
                    # Get completion content
                    content = completion.get('content', '')
                    logger.debug(f"Processing completion: {content[:100]}...")
                    
                    # Calculate component rewards
                    xml_reward = self._calculate_xml_reward(content)
                    format_reward = self._calculate_format_reward(content)
                    correctness_reward = self._calculate_correctness_reward(content, expected_answer)
                    
                    # Log component rewards
                    logger.debug(f"Component rewards - XML: {xml_reward:.3f}, Format: {format_reward:.3f}, Correctness: {correctness_reward:.3f}")
                    
                    # Combine rewards using weights
                    weights = self.config["weights"]
                    final_reward = (
                        xml_reward * weights["xml_structure"] +
                        format_reward * weights["format"] +
                        correctness_reward * weights["correctness"]
                    )
                    
                    # Normalize and append
                    normalized_reward = max(0.0, min(1.0, final_reward))
                    rewards.append(normalized_reward)
                    logger.debug(f"Final reward: {normalized_reward:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error calculating reward for completion: {str(e)}")
                    rewards.append(0.0)
            
            logger.info(f"Calculated {len(rewards)} rewards")
            logger.info(f"Average reward: {sum(rewards)/len(rewards):.3f}")
            return rewards
            
        except Exception as e:
            logger.error(f"Error in reward calculation: {str(e)}")
            return [0.0] * len(completions)

    def _validate_inputs(self, prompts: List[Dict[str, Any]], completions: List[Dict[str, Any]]) -> bool:
        """Validate input format and structure."""
        try:
            # Basic validation
            if not isinstance(prompts, list) or not isinstance(completions, list):
                logger.error("Inputs must be lists")
                return False
            
            if len(prompts) != len(completions):
                logger.error(f"Mismatched lengths: prompts={len(prompts)}, completions={len(completions)}")
                return False
            
            if not prompts or not completions:
                logger.error("Empty inputs")
                return False
            
            # Validate prompt structure
            for prompt in prompts:
                if not isinstance(prompt, dict):
                    logger.error(f"Invalid prompt type: {type(prompt)}")
                    return False
                if 'prompt' not in prompt:
                    logger.error("Missing 'prompt' key in prompt")
                    return False
            
            # Validate completion structure
            for completion in completions:
                if not isinstance(completion, dict):
                    logger.error(f"Invalid completion type: {type(completion)}")
                    return False
                if 'content' not in completion:
                    logger.error("Missing 'content' key in completion")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Error in input validation: {str(e)}")
            return False

    def _extract_answer_from_text(self, text: str) -> str:
        """Extract numerical answer from Arabic text."""
        try:
            # Split on #### and take the last part
            parts = text.split('####')
            if len(parts) < 2:
                logger.warning("No answer delimiter (####) found")
                return ""
            
            numerical_answer = parts[-1].strip()
            logger.debug(f"Raw extracted answer: {numerical_answer}")
            
            # Convert Arabic numerals to English
            arabic_to_english = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
            numerical_answer = numerical_answer.translate(arabic_to_english)
            
            # Extract only digits
            numerical_answer = ''.join(c for c in numerical_answer if c.isdigit())
            logger.debug(f"Processed numerical answer: {numerical_answer}")
            
            return numerical_answer
        except Exception as e:
            logger.error(f"Error extracting answer: {str(e)}")
            return ""

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
                logger.debug(f"Processing XML content: {text}")
                
                # More stringent scoring for incomplete structures
                has_thinking_open = "<تفكير>\n" in text
                has_thinking_close = "\n</تفكير>\n" in text
                has_answer_open = "\n<الجواب>\n" in text
                has_answer_close = "\n</الجواب>" in text
                
                # Only award points for complete tag pairs
                if has_thinking_open and has_thinking_close:
                    reward += 0.5
                if has_answer_open and has_answer_close:
                    reward += 0.5
                    
                # Apply penalties
                if text.count("<تفكير>") > 1 or text.count("</تفكير>") > 1:
                    reward -= self.config["penalties"]["multiple_tags"]
                
                extra = text.split("\n</الجواب>")[-1]
                reward -= len(extra) * self.config["penalties"]["extra_content"]
                
                logger.debug(f"XML reward components: thinking={has_thinking_open and has_thinking_close}, "
                            f"answer={has_answer_open and has_answer_close}, final={reward}")
                
                rewards.append(max(0.0, min(1.0, reward)))
            except Exception as e:
                logger.error(f"Error calculating XML reward: {str(e)}")
                rewards.append(0.0)
                
        return rewards

    def calculate_format_reward(self, completions: List[Dict[str, str]]) -> List[float]:
        """Calculate reward for format adherence."""
        logger.debug(f"Input completions type: {type(completions)}")
        logger.debug(f"First completion: {completions[0] if completions else None}")
        
        pattern = r"<تفكير>\n.*?\n</تفكير>\n<الجواب>\n.*?\n</الجواب>"
        rewards = []
        
        for completion in completions:
            try:
                # Handle nested list structure in batch processing
                if isinstance(completion, list):
                    completion = completion[0]
                
                text = completion["content"]
                logger.debug(f"Processing format for text: {text}")
                reward = 1.0 if re.match(pattern, text, re.DOTALL) else 0.0
                logger.debug(f"Format reward: {reward} for pattern match")
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