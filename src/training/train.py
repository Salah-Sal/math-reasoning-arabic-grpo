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
            from transformers import AutoConfig
            
            # Verify Unsloth and model setup
            logger.info(f"Unsloth package location: {unsloth.__file__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            
            # Verify Qwen2 configuration
            logger.info("=== Qwen2 Configuration Verification ===")
            try:
                base_config = AutoConfig.from_pretrained(training_config.model.name)
                logger.info(f"Base model config type: {type(base_config)}")
                logger.info(f"Base model config attributes: {dir(base_config)}")
                logger.info(f"Flash attention support: {getattr(base_config, 'use_flash_attention', None)}")
            except Exception as e:
                logger.error(f"Error loading base config: {str(e)}")
            
            # Verify Unsloth model handling
            logger.info("=== Unsloth Model Handling ===")
            logger.info(f"Available model types: {[m for m in dir(unsloth.models) if not m.startswith('_')]}")
            logger.info(f"Qwen2 specific handlers: {[m for m in dir(unsloth.models) if 'qwen' in m.lower()]}")
            
        except Exception as e:
            logger.error(f"Environment verification failed: {str(e)}")
            logger.error("Stack trace:", exc_info=True)

        # Initialize model with enhanced state verification
        logger.info("=== Model Initialization ===")
        try:
            # Verify Unsloth configuration
            logger.info("=== Unsloth Configuration Verification ===")
            import unsloth
            logger.info(f"Unsloth version: {unsloth.__version__}")
            logger.info(f"Available optimizations: {[m for m in dir(unsloth) if 'fast' in m.lower()]}")
            logger.info(f"Available model types: {[m for m in dir(unsloth.models) if not m.startswith('_')]}")

            # Verify model configuration
            from transformers import AutoConfig
            logger.info("=== Model Configuration Verification ===")
            base_config = AutoConfig.from_pretrained(
                training_config.model.name,
                trust_remote_code=True
            )
            logger.info(f"Model architecture: {base_config.architectures if hasattr(base_config, 'architectures') else 'Unknown'}")
            logger.info(f"Model attributes: {dir(base_config)}")
            logger.info(f"Available attention methods: {[attr for attr in dir(base_config) if 'attention' in attr.lower()]}")

            # Get base configuration without optimizations
            base_model_config = {
                'model_name': training_config.model.name,
                'trust_remote_code': True,
                'load_in_4bit': training_config.model.load_in_4bit,
                'max_seq_length': training_config.model.max_seq_length,
                'gpu_memory_utilization': training_config.model.gpu_memory_utilization
            }
            logger.info(f"Base model configuration: {base_model_config}")

            # Initialize model first without optimizations
            logger.info("Loading base model...")
            try:
                result = FastLanguageModel.from_pretrained(
                    **base_model_config
                )
                
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
                logger.info(f"Model config type: {type(model.config)}")
                logger.info(f"Model device: {next(model.parameters()).device}")
                logger.info(f"Model dtype: {next(model.parameters()).dtype}")
                
                # Check for Unsloth optimizations
                logger.info("=== Optimization Verification ===")
                logger.info(f"Available model methods: {[m for m in dir(model) if not m.startswith('_')]}")
                logger.info(f"Attention implementation: {type(model.get_decoder().layers[0].self_attn) if hasattr(model, 'get_decoder') else 'Unknown'}")
                
                # Move to device before applying optimizations
                if torch.cuda.is_available():
                    model = model.cuda()
                    logger.info(f"Model moved to CUDA: {next(model.parameters()).device}")

                # Apply GRPO patch with verification
                logger.info("=== GRPO Patch Verification ===")
                try:
                    # Check pre-patch state
                    pre_patch_methods = set(dir(model))
                    logger.info(f"Pre-patch training methods: {[m for m in pre_patch_methods if 'train' in m.lower()]}")
                    
                    # Apply patch
                    PatchFastRL("GRPO", FastLanguageModel)
                    logger.info("GRPO patch applied")
                    
                    # Verify patch effects
                    post_patch_methods = set(dir(model))
                    new_methods = post_patch_methods - pre_patch_methods
                    logger.info(f"New methods after patch: {new_methods}")
                    
                except Exception as e:
                    logger.error(f"GRPO patching failed: {str(e)}")
                    raise

                return model, tokenizer

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