import torch
from pathlib import Path
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOConfig as TRLConfig, GRPOTrainer
from src.training.monitoring import TrainingMonitor
from src.data.dataset import ArabicMathDataset
from src.training.config import GRPOConfig
from src.infrastructure.logging import get_logger
from src.training.callbacks.checkpoint import ModelCheckpointCallback
from src.training.callbacks.early_stopping import EarlyStoppingCallback
from typing import Union

logger = get_logger(__name__)

# Apply the Unsloth patch for GRPO
PatchFastRL("GRPO", FastLanguageModel)

def train_model(config_path: Union[str, Path]) -> None:
    """Train the model using the specified configuration.
    
    Args:
        config_path: Path to the configuration file
    """
    try:
        # Initialize monitoring
        monitor = TrainingMonitor(log_dir=Path("logs/training"))
        monitor.log_system_info()
        
        # Load configuration with detailed logging
        logger.info(f"Loading configuration from {config_path}")
        config = GRPOConfig.from_yaml(config_path)
        
        # Log full configuration structure
        logger.info("=== Configuration Structure ===")
        logger.info(f"Available top-level keys: {config.model_dump().keys()}")
        logger.info(f"Training settings: {config.training.model_dump()}")
        logger.info(f"Memory settings: {config.memory.model_dump()}")
        logger.info(f"Path settings: {config.paths.model_dump()}")
        logger.info("=============================")
        
        # Log monitor methods
        logger.info("=== Monitor Methods ===")
        logger.info(f"Available monitor methods: {[method for method in dir(monitor) if not method.startswith('_')]}")
        logger.info("=====================")
        
        # Verify path resolution
        logger.info("=== Path Resolution ===")
        logger.info(f"Config path resolved to: {Path(config_path).resolve()}")
        logger.info(f"Cache dir resolved to: {config.paths.cache_dir.resolve() if config.paths.cache_dir else 'None'}")
        logger.info("=====================")
        
        # Initialize model and PEFT with config verification
        logger.info("=== Model Configuration ===")
        model_config = {
            'model_name': config.model.model_name if hasattr(config, 'model') else 'Not found in config',
            'trust_remote_code': True,
            'cache_dir': str(config.paths.cache_dir) if config.paths.cache_dir else None
        }
        logger.info(f"Attempting to initialize model with config: {model_config}")
        
        model = FastLanguageModel.from_pretrained(**model_config)
        model = PatchFastRL(model)
        
        monitor.log_model_info(model)
        
        # Load dataset
        logger.info(f"Loading dataset from {config.paths.data_path}")
        dataset = ArabicMathDataset(
            data_dir=config.paths.data_path,
            cache_dir=config.paths.cache_dir
        )
        monitor.log_dataset_info(dataset)
        
        # Sample a batch for monitoring
        batch = dataset.sample_batch(config.training.batch_size)
        monitor.log_batch_processing(batch)
        
        # Initialize callbacks
        callbacks = [
            monitor,
            ModelCheckpointCallback(
                checkpoint_dir=config.paths.checkpoint_dir,
                save_steps=config.training.save_steps,
                max_checkpoints=config.memory.max_checkpoints,
                save_best=True,
                save_final=True
            )
        ]
        
        # Add early stopping if enabled
        if config.early_stopping.enabled:
            callbacks.append(
                EarlyStoppingCallback(
                    patience=config.early_stopping.patience,
                    min_improvement=config.early_stopping.min_improvement,
                    min_steps=config.early_stopping.min_steps
                )
            )
        
        # Initialize trainer with TRL config
        training_args = TRLConfig(
            learning_rate=config.training.learning_rate,
            max_steps=config.training.max_steps,
            per_device_train_batch_size=config.training.per_device_train_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            max_prompt_length=config.training.max_prompt_length,
            max_completion_length=config.training.max_completion_length,
            logging_steps=config.training.logging_steps,
            output_dir=str(config.paths.output_dir),
            report_to=config.training.report_to
        )
        
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=callbacks
        )
        
        # Start training
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        # Use monitor's existing memory logging method
        monitor.log_model_info(None)  # This will log memory stats even if model is None
        raise
    else:
        logger.info("Training completed successfully")
        monitor.log_model_info(None)  # This will log final memory stats

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_path>")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    train_model(config_path) 