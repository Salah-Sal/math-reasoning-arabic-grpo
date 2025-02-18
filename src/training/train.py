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
import os
import inspect
from packaging import version
import transformers

logger = get_logger(__name__)

# Remove global patch - we'll handle it at instance level
# PatchFastRL("GRPO", FastLanguageModel)

def get_unsloth_version() -> str:
    """Safely get Unsloth version from multiple sources."""
    try:
        import unsloth
        import pkg_resources
        
        # Try different methods to get version
        version_sources = []
        
        # Method 1: Direct attribute
        if hasattr(unsloth, 'VERSION'):
            version_sources.append(('VERSION attribute', unsloth.VERSION))
        
        # Method 2: Package metadata
        try:
            dist = pkg_resources.get_distribution('unsloth')
            version_sources.append(('pkg_resources', dist.version))
        except Exception:
            pass
        
        # Method 3: Module dictionary
        version_attrs = {k: v for k, v in unsloth.__dict__.items() if 'version' in k.lower()}
        for key, value in version_attrs.items():
            version_sources.append((f'module dict ({key})', value))
        
        # Method 4: File path version indicator
        if hasattr(unsloth, '__file__'):
            file_path = unsloth.__file__
            if 'site-packages' in file_path:
                version_from_path = file_path.split('site-packages/')[1].split('/')[0]
                if 'unsloth' in version_from_path:
                    version_sources.append(('file path', version_from_path))
        
        # Log all found versions
        logger.debug(f"Found version sources: {version_sources}")
        
        # Return first valid version found
        for source, version in version_sources:
            if version:
                logger.info(f"Using version from {source}: {version}")
                return str(version)
        
        return "Unknown"
    except Exception as e:
        logger.warning(f"Error getting Unsloth version: {str(e)}")
        return "Unknown"

def verify_compatibility():
    """Verify version compatibility of key dependencies."""
    min_transformers = "4.48.0"
    min_unsloth = "2025.2.12"
    
    # Check transformers version
    current_transformers = transformers.__version__
    if version.parse(current_transformers) < version.parse(min_transformers):
        raise ValueError(f"Transformers version {current_transformers} is too old. Minimum required: {min_transformers}")
    
    logger.info(f"Transformers version verified: {current_transformers}")
    return True

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
        
        # Verify compatibility first
        verify_compatibility()
        
        # Set up environment
        os.environ['TRUST_REMOTE_CODE'] = '1'
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Update base configuration
        base_model_config = {
            'model_name': training_config.model.name,  # Add model name
            'device_map': 'auto',
            'low_cpu_mem_usage': True
        }
        
        # Initialize model with enhanced error handling
        logger.info("=== Model Loading Process ===")
        try:
            # Log Unsloth configuration
            logger.info("=== Unsloth Configuration Verification ===")
            from unsloth.models.loader import dispatch_model
            
            logger.info(f"Model name: {base_model_config['model_name']}")
            logger.info(f"Detected model type: {'qwen' if 'qwen' in base_model_config['model_name'].lower() else 'unknown'}")
            
            # Inspect Unsloth's model handling
            logger.info("=== Model Dispatch Information ===")
            dispatch_info = {
                'available_handlers': [m for m in dir(dispatch_model) if not m.startswith('_')],
                'model_types': getattr(dispatch_model, 'MODEL_TYPES', 'Not found'),
                'default_dtype': getattr(dispatch_model, 'default_dtype', 'Not found')
            }
            logger.info(f"Dispatch info: {dispatch_info}")
            
            # Get model-specific configuration
            logger.info("=== Model-Specific Configuration ===")
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(
                base_model_config['model_name'],
                trust_remote_code=True
            )
            logger.info(f"Model architecture: {model_config.architectures if hasattr(model_config, 'architectures') else 'Unknown'}")
            logger.info(f"Model config type: {type(model_config).__name__}")
            
            # Check if bfloat16 is supported
            bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            logger.info(f"bfloat16 support: {bf16_supported}")
            
            # Set dtype based on support
            if bf16_supported:
                base_model_config['torch_dtype'] = torch.bfloat16
                logger.info("Using bfloat16 for model")
            else:
                base_model_config['torch_dtype'] = torch.float16
                logger.info("Falling back to float16")
            
            # Initialize model
            logger.info(f"Final model config: {base_model_config}")
            result = FastLanguageModel.from_pretrained(**base_model_config)
            
            if result is None:
                raise ValueError("Model initialization failed - explicit None check")
            
            if isinstance(result, tuple):
                if len(result) != 2:
                    raise ValueError(f"Expected (model, tokenizer) tuple, got tuple of length {len(result)}")
                model, tokenizer = result
            else:
                model = result
                tokenizer = None
            
            if model is None:
                raise ValueError("Model is None after unpacking")
            
            # Verify model initialization
            logger.info(f"Model initialized: {type(model)}")
            logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
            
            # Handle device placement
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{torch.cuda.current_device()}')
                model = model.to(device)
                logger.info(f"Model moved to device: {device}")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            logger.error("=== Error Context ===")
            logger.error(f"Base config: {base_model_config}")
            logger.error(f"CUDA state: {torch.cuda.is_available()}")
            logger.error(f"Memory state: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
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