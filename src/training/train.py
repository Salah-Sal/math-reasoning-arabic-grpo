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

# Remove global patch - we'll handle it at instance level
# PatchFastRL("GRPO", FastLanguageModel)

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
        training_config = GRPOConfig.from_yaml(config_path)
        
        # Log full configuration structure
        logger.info("=== Configuration Structure ===")
        logger.info(f"Available top-level keys: {training_config.model_dump().keys()}")
        logger.info(f"Training settings: {training_config.training.model_dump()}")
        logger.info(f"Memory settings: {training_config.memory.model_dump()}")
        logger.info(f"Path settings: {training_config.paths.model_dump()}")
        logger.info("=============================")
        
        # Log monitor methods
        logger.info("=== Monitor Methods ===")
        logger.info(f"Available monitor methods: {[method for method in dir(monitor) if not method.startswith('_')]}")
        logger.info("=====================")
        
        # Verify path resolution
        logger.info("=== Path Resolution ===")
        logger.info(f"Config path resolved to: {Path(config_path).resolve()}")
        logger.info(f"Cache dir resolved to: {training_config.paths.cache_dir.resolve() if training_config.paths.cache_dir else 'None'}")
        logger.info("=====================")
        
        # Enhanced version and dependency logging
        logger.info("=== Environment Verification ===")
        try:
            import unsloth
            import inspect
            import torch
            
            # Verify Unsloth and CUDA setup
            logger.info(f"Unsloth package location: {unsloth.__file__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            logger.info(f"Current CUDA device: {torch.cuda.current_device() if torch.cuda.is_available() else 'None'}")
            logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated() if torch.cuda.is_available() else 0}")
            
            # Verify FastLanguageModel state
            logger.info("=== FastLanguageModel Verification ===")
            flm_attrs = dir(FastLanguageModel)
            logger.info(f"FastLanguageModel methods: {[m for m in flm_attrs if not m.startswith('_')]}")
            logger.info(f"GRPO-related attributes: {[m for m in flm_attrs if 'grpo' in m.lower()]}")
            
            # Verify PatchFastRL
            logger.info("=== PatchFastRL Verification ===")
            patch_source = inspect.getsource(PatchFastRL)
            logger.info(f"PatchFastRL source location: {inspect.getfile(PatchFastRL)}")
            logger.info(f"PatchFastRL parameters: {inspect.signature(PatchFastRL).parameters}")
            
        except Exception as e:
            logger.error(f"Environment verification failed: {str(e)}")
            logger.error("Stack trace:", exc_info=True)

        # Initialize model with enhanced state verification
        logger.info("=== Model Initialization ===")
        try:
            # Base model config with explicit settings
            model_config = {
                'model_name': training_config.model.name,
                'trust_remote_code': True,
                'cache_dir': str(training_config.paths.cache_dir) if training_config.paths.cache_dir else None,
                'load_in_4bit': training_config.model.load_in_4bit,
                'max_seq_length': training_config.model.max_seq_length,
                'gpu_memory_utilization': training_config.model.gpu_memory_utilization,
                'use_flash_attention': training_config.model.use_flash_attention
            }
            
            # Log pre-loading state
            logger.info("=== Pre-Loading State ===")
            logger.info(f"Model configuration: {model_config}")
            logger.info(f"Memory allocated: {torch.cuda.memory_allocated() if torch.cuda.is_available() else 0}")
            
            try:
                # Load base model with state verification
                logger.info("Loading base model...")
                result = FastLanguageModel.from_pretrained(**model_config)
                
                if isinstance(result, tuple):
                    model, tokenizer = result
                    logger.info("Model and tokenizer loaded successfully")
                else:
                    model = result
                    tokenizer = None
                    logger.info("Model loaded successfully (no tokenizer)")
                
                # Verify model state
                logger.info("=== Model State Verification ===")
                logger.info(f"Model type: {type(model)}")
                logger.info(f"Model device: {next(model.parameters()).device}")
                logger.info(f"Model dtype: {next(model.parameters()).dtype}")
                logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
                logger.info(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
                
                # Prepare model for GRPO
                logger.info("=== GRPO Preparation ===")
                try:
                    # First verify model is on correct device
                    if torch.cuda.is_available():
                        model = model.cuda()
                        logger.info("Model moved to CUDA")
                    
                    # Apply GRPO patch with state verification
                    logger.info("Attempting GRPO patching...")
                    
                    # First attempt: Global patch
                    logger.info("Attempting global GRPO patch")
                    PatchFastRL("GRPO", FastLanguageModel)
                    logger.info("Global patch applied")
                    
                    # Second attempt: Instance patch
                    logger.info("Attempting instance-level GRPO patch")
                    patched_model = PatchFastRL(model)
                    
                    if patched_model is not None:
                        model = patched_model
                        logger.info("Instance-level patch successful")
                    else:
                        logger.warning("Instance-level patch returned None, using globally patched model")
                    
                    # Verify final model state
                    logger.info("=== Final Model State ===")
                    logger.info(f"Final model type: {type(model)}")
                    logger.info(f"GRPO methods: {[m for m in dir(model) if 'grpo' in m.lower()]}")
                    logger.info(f"Base model type: {type(getattr(model, 'base_model', None))}")
                    logger.info(f"Available methods: {[m for m in dir(model) if not m.startswith('_')]}")
                    
                    return model, tokenizer
                    
                except Exception as e:
                    logger.error(f"GRPO patching failed: {str(e)}")
                    logger.error("Stack trace:", exc_info=True)
                    raise
                
            except Exception as e:
                logger.error(f"Model loading failed: {str(e)}")
                logger.error("Stack trace:", exc_info=True)
                raise
                
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            raise

        # Initialize trainer with verified model
        logger.info("=== Trainer Initialization ===")
        training_args = TRLConfig(
            learning_rate=training_config.training.learning_rate,
            max_steps=training_config.training.max_steps,
            per_device_train_batch_size=training_config.training.per_device_train_batch_size,
            gradient_accumulation_steps=training_config.training.gradient_accumulation_steps,
            max_prompt_length=training_config.training.max_prompt_length,
            max_completion_length=training_config.training.max_completion_length,
            logging_steps=training_config.training.logging_steps,
            output_dir=str(training_config.paths.output_dir),
            report_to=training_config.training.report_to
        )
        
        logger.info(f"Training arguments: {training_args}")
        
        # Load dataset with detailed logging
        logger.info(f"Loading dataset from {training_config.paths.data_path}")
        dataset = ArabicMathDataset(
            data_dir=training_config.paths.data_path,
            cache_dir=training_config.paths.cache_dir
        )
        monitor.log_dataset_info(dataset)
        
        # Sample a batch for monitoring with enhanced error handling
        logger.info("Sampling batch for monitoring")
        try:
            batch = dataset.sample_batch(training_config.training.per_device_train_batch_size)
            if not batch.get('examples'):
                logger.warning("No examples in sampled batch")
            else:
                logger.info(f"Successfully sampled batch of {batch['size']} examples")
            monitor.log_batch_processing(batch)
        except Exception as e:
            logger.error(f"Error sampling batch: {str(e)}")
            logger.warning("Continuing without batch monitoring")
        
        # Initialize callbacks with enhanced logging
        logger.info("=== Initializing Callbacks ===")
        
        # Prepare checkpoint callback with proper config access
        checkpoint_config = {
            'checkpoint_dir': training_config.paths.checkpoint_dir,
            'save_steps': training_config.training.save_steps,
            'max_checkpoints': training_config.training.max_checkpoints,  # Changed from memory to training
            'save_best': True,
            'save_final': True
        }
        logger.info(f"Checkpoint configuration: {checkpoint_config}")
        
        callbacks = [
            monitor,
            ModelCheckpointCallback(**checkpoint_config)
        ]
        
        # Add early stopping if enabled
        if training_config.early_stopping.enabled:
            early_stopping_config = {
                'patience': training_config.early_stopping.patience,
                'min_improvement': training_config.early_stopping.min_improvement,
                'min_steps': training_config.early_stopping.min_steps
            }
            logger.info(f"Early stopping configuration: {early_stopping_config}")
            callbacks.append(EarlyStoppingCallback(**early_stopping_config))
        
        logger.info(f"Initialized {len(callbacks)} callbacks")
        logger.info("=============================")
        
        # Create reward function wrapper for GRPO
        def reward_function(prompts, completions, **kwargs):
            logger.info("=== GRPO Reward Function Called ===")
            logger.info(f"Prompts: {len(prompts)}, Completions: {len(completions)}")
            logger.info(f"Additional kwargs: {kwargs}")
            try:
                rewards = reward_handler.calculate_rewards(prompts=prompts, completions=completions, **kwargs)
                logger.info(f"Calculated rewards: min={min(rewards):.3f}, max={max(rewards):.3f}, avg={sum(rewards)/len(rewards):.3f}")
                return rewards
            except Exception as e:
                logger.error(f"Error in reward function: {str(e)}", exc_info=True)
                return [0.0] * len(completions)

        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=callbacks,
            reward_funcs=[reward_function]  # Use wrapped function
        )
        logger.info("GRPO Trainer initialized successfully")
        
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