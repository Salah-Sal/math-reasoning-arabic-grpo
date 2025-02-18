import torch
from pathlib import Path
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOConfig, GRPOTrainer
from src.training.monitoring import TrainingMonitor
from src.data.dataset import ArabicMathDataset
from src.infrastructure.config import GRPOConfig as ProjectConfig
from src.infrastructure.logging import get_logger
from src.training.callbacks.checkpoint import ModelCheckpointCallback
from src.training.callbacks.early_stopping import EarlyStoppingCallback
from typing import Union

logger = get_logger(__name__)

def train_model(config_path: Union[str, Path]) -> None:
    """Train the model using the specified configuration.
    
    Args:
        config_path: Path to the configuration file
    """
    try:
        # Initialize monitoring
        monitor = TrainingMonitor(log_dir=Path("logs/training"))
        monitor.log_system_info()
        
        # Load configuration
        config = GRPOConfig.from_yaml(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        
        # Initialize model and PEFT
        model = FastLanguageModel.from_pretrained(
            model_name=config.training.model_name,
            trust_remote_code=True,
            cache_dir=config.paths.cache_dir
        )
        model = PatchFastRL(model)
        
        monitor.log_model_info(model)
        
        # Load dataset
        dataset = ArabicMathDataset(
            data_path=config.paths.data_path,
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
        
        # Initialize trainer
        trainer = GRPOTrainer(
            model=model,
            dataset=dataset,
            config=config,
            callbacks=callbacks
        )
        
        # Start training
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        monitor.log_final_memory_stats()
        raise
    else:
        logger.info("Training completed successfully")
        monitor.log_final_memory_stats()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_path>")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    train_model(config_path) 