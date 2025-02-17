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
        
        # Force garbage collection and cache clearing
        torch.cuda.empty_cache()
        
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
            
            # Adjust settings for GPUs with less memory without mutating the config
            adjusted_config = self._adjust_config_based_on_gpu(self.config, total_gb)
            
            logger.info(f"Loading model: {adjusted_config.model.model_name}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=adjusted_config.model.model_name,
                max_seq_length=adjusted_config.model.max_seq_length,
                load_in_4bit=adjusted_config.model.load_in_4bit,
                fast_inference=adjusted_config.model.fast_inference,
                max_lora_rank=adjusted_config.model.max_lora_rank,
                gpu_memory_utilization=adjusted_config.model.gpu_memory_utilization
            )
            
            # Configure PEFT with optimized settings
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=adjusted_config.model.max_lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_alpha=16,
                use_gradient_checkpointing="unsloth",  # Enable Unsloth optimizations
                random_state=3407,
            )
            
            logger.info("Model initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
    
    def _adjust_config_based_on_gpu(self, config: ProjectConfig, total_gb: float) -> ProjectConfig:
        """Adjust configuration settings based on available GPU memory."""
        adjusted_model_config = config.model.copy()
        adjusted_training_config = config.training.copy()

        if total_gb < 10:
            adjusted_model_config.max_seq_length = 256
            adjusted_model_config.gpu_memory_utilization = 0.6
            adjusted_training_config.per_device_train_batch_size = 1
            adjusted_training_config.gradient_accumulation_steps = 1
            adjusted_training_config.num_generations = 4
            logger.info("Adjusted settings for GPUs with less memory")

        return ProjectConfig(
            model=adjusted_model_config,
            training=adjusted_training_config,
            data=config.data,
            seed=config.seed,
            device=config.device
        )
    
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