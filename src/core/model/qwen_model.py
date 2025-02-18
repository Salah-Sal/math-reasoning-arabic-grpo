from typing import Tuple, Optional, Dict, Any
import os
from src.core.model.base import BaseLanguageModel
from src.infrastructure.logging import get_logger
from transformers import PreTrainedModel, PreTrainedTokenizer
from unsloth import FastLanguageModel, PatchFastRL

logger = get_logger(__name__)

class QwenModel(BaseLanguageModel):
    """Qwen model implementation with unsloth optimization."""
    
    def __init__(self):
        super().__init__()
        logger.info("Initializing QwenModel")
        self.validate_gpu_setup()

    def load_model(self,
                  model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                  max_seq_length: int = 384,
                  load_in_4bit: bool = True,
                  fast_inference: bool = False,
                  gpu_memory_utilization: float = 0.7,
                  **kwargs) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load the Qwen model using unsloth optimization.
        
        Args:
            model_name: Name or path of the Qwen model
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to load in 4-bit quantization
            fast_inference: Whether to enable fast inference
            gpu_memory_utilization: GPU memory utilization target
            **kwargs: Additional arguments for model loading
        
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            logger.info(f"Loading model {model_name} with max_seq_length={max_seq_length}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                fast_inference=fast_inference,
                gpu_memory_utilization=gpu_memory_utilization,
                **kwargs
            )
            logger.info("Model loaded successfully")
            return self.model, self.tokenizer
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def setup_peft(self,
                  target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj"],
                  lora_rank: int = 16,
                  lora_alpha: int = 16,
                  use_gradient_checkpointing: str = "unsloth",
                  random_state: int = 3407,
                  **kwargs) -> PreTrainedModel:
        """Set up PEFT for the Qwen model.
        
        Args:
            target_modules: List of module names for LoRA
            lora_rank: Rank of LoRA matrices
            lora_alpha: Alpha parameter for LoRA
            use_gradient_checkpointing: Gradient checkpointing method
            random_state: Random seed for reproducibility
            **kwargs: Additional PEFT configuration
        
        Returns:
            Model with PEFT configuration
        """
        if self.model is None:
            raise ValueError("Model must be loaded before setting up PEFT")
        
        try:
            logger.info("Setting up PEFT configuration")
            logger.info(f"Initial model type: {type(self.model)}")
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=lora_rank,
                target_modules=target_modules,
                lora_alpha=lora_alpha,
                use_gradient_checkpointing=use_gradient_checkpointing,
                random_state=random_state,
                **kwargs
            )
            logger.info(f"Post-PEFT model type: {type(self.model)}")
            logger.info(f"Model attributes: {dir(self.model)}")
            logger.info(f"Model base attributes: {dir(self.model.base_model)}")
            return self.model
        except Exception as e:
            logger.error(f"PEFT setup failed: {str(e)}")
            raise

    def save_model(self,
                  output_dir: str,
                  save_method: str = "merged_4bit") -> None:
        """Save the Qwen model.
        
        Args:
            output_dir: Directory to save the model
            save_method: Method to use for saving ("lora", "merged_4bit")
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving")
        
        try:
            logger.info(f"Saving model of type: {type(self.model)}")
            logger.info(f"Model hierarchy: {type(self.model)} -> {type(getattr(self.model, 'base_model', None))} -> {type(getattr(getattr(self.model, 'base_model', None), 'model', None))}")
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model to {output_dir} using method {save_method}")
            
            if save_method == "lora":
                self.model.save_lora(output_dir)
                logger.info("LoRA weights saved successfully")
            elif save_method == "merged_4bit":
                self.model.save_pretrained_merged(
                    output_dir,
                    self.tokenizer,
                    save_method="merged_4bit"
                )
                logger.info("Merged model saved successfully")
            else:
                raise ValueError(f"Unsupported save method: {save_method}")
        except Exception as e:
            logger.error(f"Save failed: {str(e)}")
            raise 