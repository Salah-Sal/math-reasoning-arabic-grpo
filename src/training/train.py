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
from typing import Union, Dict, Any
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

def inspect_model_loading_chain():
    """Inspect the model loading chain and available configurations."""
    try:
        import unsloth
        logger.info("=== Model Loading Chain Inspection ===")
        
        # Check available model classes
        logger.info("Available model classes:")
        logger.info(f"FastLanguageModel: {inspect.getmembers(unsloth.FastLanguageModel)}")
        logger.info(f"Qwen2 module exists: {hasattr(unsloth.models, 'qwen2')}")
        if hasattr(unsloth.models, 'qwen2'):
            logger.info(f"Qwen2 module contents: {dir(unsloth.models.qwen2)}")
        
        # Check adapter routing
        logger.info("Adapter routing:")
        logger.info(f"Direct Qwen2 loading available: {hasattr(unsloth.models.qwen2, 'from_pretrained')}")
        logger.info(f"Using Llama adapter: {hasattr(unsloth.models.llama, 'from_pretrained')}")
        
        return True
    except Exception as e:
        logger.error(f"Error inspecting model chain: {str(e)}")
        return False

def clean_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Remove problematic parameters from model config."""
    # Expanded list of parameters to remove
    remove_params = [
        'use_fast_tokenizer', 
        'use_fast',
        'tokenizer',  # Remove tokenizer from model config
        'use_flash_attention',  # Remove potentially conflicting parameters
        'use_memory_efficient_attention'
    ]
    
    cleaned = {k: v for k, v in config.items() if k not in remove_params}
    logger.info(f"Removed parameters: {[k for k in config if k not in cleaned]}")
    return cleaned

def verify_qwen_compatibility(model_name: str) -> bool:
    """Verify Qwen2 model compatibility."""
    try:
        from transformers import AutoConfig
        logger.info(f"Verifying compatibility for model: {model_name}")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        logger.info(f"Qwen2 config type: {type(config)}")
        logger.info(f"Available config attributes: {dir(config)}")
        return True
    except Exception as e:
        logger.error(f"Qwen2 compatibility check failed: {str(e)}")
        return False

def load_model_with_fallback(model_config: Dict[str, Any], tokenizer) -> Any:
    """Try loading model with fallback options."""
    logger.info("=== Attempting Model Load with Fallback ===")
    
    try:
        # First attempt: Direct loading
        logger.info("Attempt 1: Direct FastLanguageModel loading")
        clean_config = clean_model_config(model_config)
        logger.info(f"Clean config for attempt 1: {clean_config}")
        
        result = FastLanguageModel.from_pretrained(
            **clean_config
        )
        
        if result is not None:
            logger.info("Direct loading successful")
            return result
            
    except Exception as e:
        logger.error(f"Direct loading failed: {str(e)}")
        
        try:
            # Second attempt: Load with transformers then optimize
            logger.info("Attempt 2: Loading with transformers then applying Unsloth")
            from transformers import AutoModelForCausalLM
            
            base_model = AutoModelForCausalLM.from_pretrained(
                model_config['model_name'],
                trust_remote_code=True,
                load_in_4bit=model_config.get('load_in_4bit', True)
            )
            
            logger.info("Base model loaded, applying Unsloth optimization")
            # Apply Unsloth optimization after loading
            optimized_model = FastLanguageModel.optimize_model(
                base_model,
                gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.7),
                use_gradient_checkpointing=model_config.get('use_gradient_checkpointing', True)
            )
            
            return (optimized_model, tokenizer)
            
        except Exception as e2:
            logger.error(f"Fallback loading failed: {str(e2)}")
            raise

def train_model(config_path: Union[str, Path]) -> None:
    """Train the model using the specified configuration.
    
    Args:
        config_path: Path to the configuration file
    """
    try:
        # Initialize monitoring
        monitor = TrainingMonitor(log_dir=Path("logs/training"))
        monitor.log_system_info()
        
        # Enhanced version logging
        logger.info("=== Version Verification ===")
        import torch
        import transformers
        import unsloth
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"Transformers: {transformers.__version__}")
        logger.info(f"Unsloth: {get_unsloth_version()}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        
        # Environment setup logging
        logger.info("=== Environment Setup ===")
        os.environ['TRUST_REMOTE_CODE'] = '1'
        logger.info(f"Environment variables:")
        logger.info(f"TRUST_REMOTE_CODE: {os.getenv('TRUST_REMOTE_CODE')}")
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")
        
        # Memory state logging
        logger.info("=== Initial Memory State ===")
        if torch.cuda.is_available():
            logger.info(f"CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
            logger.info(f"CUDA Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f}MB")
            logger.info(f"Max Memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f}MB")
        
        # Load configuration with detailed logging
        logger.info(f"Loading configuration from {config_path}")
        training_config = GRPOConfig.from_yaml(config_path)
        
        # Initialize base configurations with separation
        logger.info("=== Configuration Chain Logging ===")
        base_model_config = {
            'model_name': training_config.model.name,
            'max_seq_length': training_config.model.max_seq_length,
            'load_in_4bit': training_config.model.load_in_4bit,
            'device_map': 'auto',
            'trust_remote_code': True
        }
        
        # Separate tokenizer configuration
        tokenizer_config = {
            'trust_remote_code': True,
            'use_fast': True
        }
        logger.info(f"Tokenizer config: {tokenizer_config}")
        
        # Clean model configuration
        model_config = clean_model_config(base_model_config)
        logger.info(f"Initial cleaned model config: {model_config}")
        
        # Memory configuration
        logger.info("=== Memory Configuration ===")
        memory_config = training_config.memory.model_dump()
        model_config.update({
            'gpu_memory_utilization': memory_config.get('gpu_memory_utilization', 0.7),
            'use_gradient_checkpointing': memory_config.get('use_gradient_checkpointing', True)
        })
        logger.info(f"Updated model configuration with memory settings: {model_config}")
        
        # Verify Qwen compatibility
        if not verify_qwen_compatibility(model_config['model_name']):
            logger.warning("Qwen compatibility check failed - proceeding with caution")
        
        # Clear CUDA cache before model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
            logger.info(f"Pre-load CUDA Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
        
        # Add model chain inspection
        inspect_model_loading_chain()
        
        # Initialize model with fallback
        logger.info("=== Model Loading Process ===")
        try:
            # Load tokenizer first
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_config['model_name'],
                **tokenizer_config
            )
            logger.info(f"Tokenizer loaded successfully: {type(tokenizer)}")
            
            # Try loading model with fallback
            result = load_model_with_fallback(model_config, tokenizer)
            
            # Verify loaded model
            if result is not None:
                model = result[0] if isinstance(result, tuple) else result
                logger.info(f"Model loaded successfully. Type: {type(model)}")
                logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
                
                return (model, tokenizer)
            else:
                raise ValueError("Model loading returned None")
                
        except Exception as e:
            logger.error("=== Model Loading Error Context ===")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Model loading chain trace:", exc_info=True)
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