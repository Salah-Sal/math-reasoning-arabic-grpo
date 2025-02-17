from pathlib import Path
from typing import Optional
import torch
from unsloth import FastLanguageModel
from src.infrastructure.config import ProjectConfig
from src.infrastructure.logging import get_logger
from src.data.dataset import ArabicMathDataset
from src.utils.memory import clear_memory

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
        # Clear memory before initialization
        clear_memory()
        
        # Validate configuration
        self._validate_config(config)
        
        self.config = config
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
        """Initialize the model and tokenizer."""
        try:
            # Clear memory before loading model
            clear_memory()
            
            # Check available GPU memory
            free_memory, total_memory = torch.cuda.mem_get_info()
            total_gb = total_memory / 1024**3
            free_gb = free_memory / 1024**3
            
            logger.info(f"GPU Memory: {free_gb:.2f}GB free of {total_gb:.2f}GB total")
            
            # Adjust memory utilization based on available memory
            if total_gb < 10:  # For GPUs with less than 10GB
                self.config.model.gpu_memory_utilization = min(
                    0.8,  # Maximum utilization
                    self.config.model.gpu_memory_utilization
                )
                logger.info(f"Adjusted GPU memory utilization to {self.config.model.gpu_memory_utilization}")

            logger.info(f"Loading model: {self.config.model.model_name}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model.model_name,
                max_seq_length=self.config.model.max_seq_length,
                load_in_4bit=self.config.model.load_in_4bit,
                fast_inference=self.config.model.fast_inference,
                max_lora_rank=self.config.model.max_lora_rank,
                gpu_memory_utilization=self.config.model.gpu_memory_utilization
            )
            
            # Configure PEFT
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.config.model.max_lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_alpha=16,
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )
            
            logger.info("Model initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
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