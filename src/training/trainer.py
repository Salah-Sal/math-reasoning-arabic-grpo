from pathlib import Path
from typing import Optional
import torch
from unsloth import FastLanguageModel
from src.infrastructure.config import ProjectConfig
from src.infrastructure.logging import get_logger
from src.data.dataset import ArabicMathDataset
import traceback

logger = get_logger(__name__)

class Trainer:
    """Trainer class for the Arabic Math Reasoning model."""
    
    def __init__(
        self,
        config: ProjectConfig,
        poc_mode: bool = False,
        output_dir: Optional[Path] = None
    ):
        """Initialize the trainer.
        
        Args:
            config: Training configuration
            poc_mode: Whether to run in proof-of-concept mode
            output_dir: Directory for saving outputs
        """
        self.config = config
        self._validate_config(self.config)
        
        self.poc_mode = poc_mode
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initializing trainer in {'POC' if poc_mode else 'full'} mode")
        self._initialize_model()
    
    def _validate_config(self, config: ProjectConfig):
        """Validate trainer configuration."""
        if not 0 <= config.model.gpu_memory_utilization <= 1:
            raise ValueError("GPU memory utilization must be between 0 and 1")
        
        if config.model.max_seq_length <= 0:
            raise ValueError("Maximum sequence length must be positive")
        
        if config.training.per_device_train_batch_size <= 0:
            raise ValueError("Batch size must be positive")
    
    def _initialize_model(self):
        """Initialize the model and tokenizer using vLLM for memory management."""
        try:
            logger.info("Starting model initialization...")
            logger.info(f"Model name: {self.config.model.model_name}")
            logger.info(f"Model config: {self.config.model.model_dump()}")
            
            logger.info("Creating FastLanguageModel instance with vLLM...")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model.model_name,
                max_seq_length=self.config.model.max_seq_length,
                load_in_4bit=self.config.model.load_in_4bit,
                fast_inference=self.config.model.fast_inference,
                max_lora_rank=self.config.model.max_lora_rank,
                gpu_memory_utilization=self.config.model.gpu_memory_utilization
            )
            
            logger.info(f"Model type: {type(self.model)}")
            logger.info(f"Model config type: {type(self.model.config)}")
            logger.info("Model initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise
    
    def prepare_dataset(self, dataset: ArabicMathDataset):
        """Prepare dataset for training.
        
        Args:
            dataset: The dataset to prepare
            
        Returns:
            Prepared dataset, possibly subsampled for POC mode
        """
        if self.poc_mode:
            # For POC, limit dataset size
            full_length = len(dataset)
            poc_size = min(100, full_length)
            logger.info(f"POC mode: Using {poc_size} examples from {full_length}")
            
            # Convert to HuggingFace dataset and take first poc_size examples
            hf_dataset = dataset.to_huggingface_dataset()
            return hf_dataset.select(range(poc_size))
        
        return dataset.to_huggingface_dataset() 