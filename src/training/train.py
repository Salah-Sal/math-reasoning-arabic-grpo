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
            import sys
            import pkg_resources
            
            # Verify Unsloth installation
            logger.info("=== Unsloth Installation Verification ===")
            logger.info(f"Unsloth package location: {unsloth.__file__}")
            logger.info(f"Unsloth package path: {sys.modules['unsloth'].__path__ if hasattr(sys.modules['unsloth'], '__path__') else 'N/A'}")
            
            # Try multiple version detection methods
            version_info = {
                'pkg_resources': pkg_resources.working_set.by_key.get('unsloth'),
                'module_version': getattr(unsloth, 'VERSION', None),
                'module_dict': {k: v for k, v in unsloth.__dict__.items() if 'version' in k.lower()},
                'file_path': unsloth.__file__
            }
            logger.info(f"Version detection attempts: {version_info}")
            
            # Verify module structure
            logger.info("=== Unsloth Module Structure ===")
            logger.info(f"Available top-level attributes: {[attr for attr in dir(unsloth) if not attr.startswith('_')]}")
            logger.info(f"Available FastLanguageModel attributes: {[attr for attr in dir(unsloth.FastLanguageModel) if not attr.startswith('_')]}")
            
            # Verify patching status
            logger.info("=== Patching Status ===")
            patch_status = {
                'FastLanguageModel_patched': hasattr(unsloth.FastLanguageModel, 'is_patched'),
                'GRPO_available': 'GRPO' in [attr for attr in dir(unsloth) if not attr.startswith('_')],
                'patch_functions': [attr for attr in dir(unsloth) if 'patch' in attr.lower()]
            }
            logger.info(f"Patch verification: {patch_status}")
            
            # Verify model handlers
            logger.info("=== Model Handler Verification ===")
            try:
                handler_info = {
                    'available_models': [m for m in dir(unsloth.models) if not m.startswith('_')],
                    'qwen_handlers': [m for m in dir(unsloth.models) if 'qwen' in m.lower()],
                    'model_classes': inspect.getmembers(unsloth.models, inspect.isclass)
                }
                logger.info(f"Handler information: {handler_info}")
            except Exception as e:
                logger.error(f"Error checking model handlers: {str(e)}")

            # Verify CUDA setup
            logger.info("=== CUDA Verification ===")
            cuda_info = {
                'cuda_available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
                'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None
            }
            logger.info(f"CUDA information: {cuda_info}")
            
        except Exception as e:
            logger.error(f"Environment verification failed: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            raise

        # Initialize model with enhanced state verification
        logger.info("=== Model Initialization ===")
        try:
            # Verify Unsloth configuration
            logger.info("=== Unsloth Configuration Verification ===")
            
            # Get and log version information
            logger.info("=== Version Information ===")
            version = get_unsloth_version()
            logger.info(f"Detected Unsloth version: {version}")
            
            # Log available features
            logger.info("=== Available Features ===")
            features = {
                'fast_language_model': hasattr(unsloth, 'FastLanguageModel'),
                'patch_fast_rl': hasattr(unsloth, 'PatchFastRL'),
                'models': hasattr(unsloth, 'models'),
                'optimizations': [m for m in dir(unsloth) if 'fast' in m.lower()]
            }
            logger.info(f"Available features: {features}")

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
                # Log pre-loading state
                logger.info("=== Pre-Loading State ===")
                logger.info(f"CUDA memory before loading: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
                logger.info(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}")
                
                # Add trust_remote_code handling
                logger.info("Handling remote code execution...")
                os.environ['TRUST_REMOTE_CODE'] = '1'  # Explicitly set trust
                
                # Enhanced model loading with device management
                logger.info("=== Model Loading Process ===")
                try:
                    result = FastLanguageModel.from_pretrained(
                        **base_model_config
                    )
                    logger.info("Initial model loading call completed")
                except Exception as e:
                    logger.error(f"Error in FastLanguageModel.from_pretrained: {str(e)}")
                    raise
                
                # Verify result immediately
                logger.info("=== Model Loading Verification ===")
                logger.info(f"Result type: {type(result)}")
                logger.info(f"Result structure: {result if isinstance(result, tuple) else 'Not tuple'}")
                
                if result is None:
                    raise ValueError("Model loading returned None")
                
                # Unpack result with verification
                if isinstance(result, tuple):
                    model, tokenizer = result
                    logger.info("Unpacked model and tokenizer from tuple")
                else:
                    model = result
                    tokenizer = None
                    logger.info("Got model without tokenizer")
                
                # Verify model object
                if model is None:
                    raise ValueError("Model is None after unpacking")
                
                logger.info("=== Model Object Verification ===")
                logger.info(f"Model type: {type(model)}")
                logger.info(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
                
                # Verify model parameters exist
                try:
                    param_count = sum(p.numel() for p in model.parameters())
                    logger.info(f"Model parameter count: {param_count}")
                except Exception as e:
                    logger.error(f"Error accessing model parameters: {str(e)}")
                    raise ValueError("Model parameters not accessible")
                
                # Device management with verification
                if torch.cuda.is_available():
                    logger.info("=== CUDA Device Management ===")
                    current_device = torch.cuda.current_device()
                    logger.info(f"Current CUDA device: {current_device}")
                    logger.info(f"Memory before move: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
                    
                    try:
                        model = model.cuda()
                        logger.info("Model moved to CUDA")
                        device = next(model.parameters()).device
                        logger.info(f"Model device verified: {device}")
                    except Exception as e:
                        logger.error(f"Error moving model to CUDA: {str(e)}")
                        raise

                # Apply GRPO patch with enhanced verification
                logger.info("=== GRPO Patch Verification ===")
                try:
                    # Verify model is ready for patching
                    if not hasattr(model, 'parameters'):
                        raise ValueError("Model not properly initialized for patching")
                    
                    # Store pre-patch state
                    pre_patch_methods = set(dir(model))
                    pre_patch_params = sum(p.numel() for p in model.parameters())
                    logger.info(f"Pre-patch parameter count: {pre_patch_params}")
                    logger.info(f"Pre-patch training methods: {[m for m in pre_patch_methods if 'train' in m.lower()]}")
                    
                    # Apply patch with verification
                    logger.info("Applying GRPO patch...")
                    try:
                        PatchFastRL("GRPO", FastLanguageModel)
                        logger.info("GRPO patch call completed")
                    except Exception as e:
                        logger.error(f"Error in PatchFastRL call: {str(e)}")
                        raise
                    
                    # Verify patch effects
                    post_patch_methods = set(dir(model))
                    post_patch_params = sum(p.numel() for p in model.parameters())
                    new_methods = post_patch_methods - pre_patch_methods
                    
                    logger.info("=== Patch Effect Verification ===")
                    logger.info(f"New methods added: {new_methods}")
                    logger.info(f"Parameter count change: {post_patch_params - pre_patch_params}")
                    logger.info(f"Training methods after patch: {[m for m in post_patch_methods if 'train' in m.lower()]}")
                    
                    # Verify GRPO-specific attributes
                    grpo_attributes = {
                        'has_grpo_methods': any('grpo' in m.lower() for m in dir(model)),
                        'is_patched': hasattr(model, 'is_patched'),
                        'has_reward_model': hasattr(model, 'compute_reward')
                    }
                    logger.info(f"GRPO attributes verification: {grpo_attributes}")
                    
                except Exception as e:
                    logger.error(f"GRPO patching failed: {str(e)}")
                    logger.error("=== Patching Error Context ===")
                    logger.error(f"Model state: {type(model)}")
                    logger.error(f"Available methods: {[m for m in dir(model) if not m.startswith('_')]}")
                    raise

                return model, tokenizer

            except Exception as e:
                logger.error(f"Model loading failed: {str(e)}")
                logger.error("Stack trace:", exc_info=True)
                
                # Additional error context
                logger.error("=== Error Context ===")
                if torch.cuda.is_available():
                    logger.error(f"CUDA memory at error: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
                logger.error(f"Base config used: {base_model_config}")
                
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