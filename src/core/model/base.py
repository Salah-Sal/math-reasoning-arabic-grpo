from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

class BaseLanguageModel(ABC):
    """Base class for language model implementations."""
    
    def __init__(self):
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        logger.info("Initializing BaseLanguageModel")

    @abstractmethod
    def load_model(self, 
                  model_name: str,
                  max_seq_length: int,
                  load_in_4bit: bool = True,
                  **kwargs) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load the model and tokenizer.
        
        Args:
            model_name: Name or path of the model to load
            max_seq_length: Maximum sequence length for the model
            load_in_4bit: Whether to load the model in 4-bit quantization
            **kwargs: Additional arguments for model loading
            
        Returns:
            Tuple of (model, tokenizer)
        """
        pass

    @abstractmethod
    def setup_peft(self,
                  target_modules: list,
                  lora_rank: int = 16,
                  lora_alpha: int = 16,
                  **kwargs) -> PreTrainedModel:
        """Set up PEFT (Parameter Efficient Fine-Tuning) for the model.
        
        Args:
            target_modules: List of module names to apply LoRA
            lora_rank: Rank of LoRA matrices
            lora_alpha: Alpha parameter for LoRA
            **kwargs: Additional PEFT configuration parameters
            
        Returns:
            Model with PEFT configuration applied
        """
        pass

    @abstractmethod
    def save_model(self,
                  output_dir: str,
                  save_method: str = "merged_4bit") -> None:
        """Save the model and its configuration.
        
        Args:
            output_dir: Directory to save the model
            save_method: Method to use for saving ("lora", "merged_4bit", etc.)
        """
        pass

    def validate_gpu_setup(self) -> Dict[str, Any]:
        """Validate GPU setup and return system information.
        
        Returns:
            Dictionary containing GPU information and memory stats
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Training will be slow on CPU.")
            return {"cuda_available": False}
        
        try:
            gpu_info = {
                "cuda_available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated": f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB",
                "memory_reserved": f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB",
            }
            logger.info(f"GPU Setup: {gpu_info}")
            return gpu_info
        except Exception as e:
            logger.error(f"Error validating GPU setup: {str(e)}")
            return {"cuda_available": True, "error": str(e)}

    def clear_gpu_memory(self) -> None:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("GPU memory cache cleared")
            except Exception as e:
                logger.error(f"Error clearing GPU memory: {str(e)}") 