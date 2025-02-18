import torch
from pathlib import Path
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOConfig, GRPOTrainer
from src.training.monitoring import TrainingMonitor
from src.data.dataset import ArabicMathDataset
from src.infrastructure.config import GRPOConfig as ProjectConfig
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

def train_model(config_path: Path):
    """Main training function with comprehensive monitoring."""
    
    # Initialize monitoring
    monitor = TrainingMonitor(Path("logs/training"))
    monitor.log_system_info()
    
    try:
        # Load configuration
        config = ProjectConfig.from_yaml(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        
        # Initialize model with monitoring
        logger.info("Initializing model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model.model_name,
            max_seq_length=config.model.max_seq_length,
            load_in_4bit=config.model.load_in_4bit,
            fast_inference=config.model.fast_inference,
            gpu_memory_utilization=config.memory.gpu_memory_utilization
        )
        monitor.log_model_info(model)
        
        # Setup PEFT
        logger.info("Setting up PEFT...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.model.max_lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=config.model.max_lora_rank,
            use_gradient_checkpointing=config.memory.use_gradient_checkpointing,
            random_state=config.seed
        )
        monitor.log_model_info(model)  # Log after PEFT setup
        
        # Load dataset with monitoring
        logger.info("Loading dataset...")
        dataset = ArabicMathDataset(
            data_dir=config.data.data_dir,
            system_prompt=config.training.system_prompt
        ).to_huggingface_dataset()
        monitor.log_dataset_info(dataset)
        
        # Sample batch for monitoring
        if len(dataset) > 0:
            sample_batch = {k: dataset[0][k] for k in dataset[0].keys()}
            monitor.log_batch_processing(sample_batch)
            monitor.log_arabic_processing(
                str(sample_batch['prompt']),
                tokenizer
            )
        
        # Initialize trainer
        training_args = GRPOConfig(
            learning_rate=config.training.learning_rate,
            max_steps=config.training.max_steps,
            per_device_train_batch_size=config.training.per_device_train_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            max_prompt_length=config.training.max_prompt_length,
            max_completion_length=config.training.max_completion_length,
            logging_steps=config.training.logging_steps,
            output_dir=str(config.paths.output_dir),
            report_to="none"
        )
        
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=dataset,
            callbacks=[monitor]  # Add monitoring callback
        )
        
        # Train with monitoring
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise
    finally:
        # Log final memory stats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            monitor.log_model_info(model)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_path>")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    train_model(config_path) 