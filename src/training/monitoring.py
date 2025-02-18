import logging
import torch
from typing import Any, Dict, Optional
from pathlib import Path
import psutil
import os
import sys
import platform
from datetime import datetime
from transformers import PreTrainedTokenizer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from datasets import Dataset
from src.training.callbacks.base import BaseCallback

logger = logging.getLogger(__name__)

class TrainingMonitor(TrainerCallback, BaseCallback):
    """Comprehensive monitoring system for training process."""
    
    def __init__(self, log_dir: Path):
        BaseCallback.__init__(self)
        TrainerCallback.__init__(self)
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.file_handler = logging.FileHandler(
            self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self.file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(self.file_handler)
        self.logger.setLevel(logging.INFO)
        
        # Initialize metrics
        self.start_time = None
        self.best_reward = float('-inf')
        self.training_steps = 0
        
        logger.info("Initialized TrainingMonitor with HuggingFace Trainer compatibility")
    
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl) -> TrainerControl:
        """Called at the end of trainer initialization."""
        logger.info("=== Trainer Initialization Complete ===")
        logger.info(f"Training Arguments: {args}")
        logger.info(f"Initial State: {state}")
        self.log_system_info()
        return control
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl) -> TrainerControl:
        """Called at the beginning of training."""
        self.start_time = datetime.now()
        logger.info(f"Training started at {self.start_time}")
        logger.info(f"Training Arguments: {args}")
        return control
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl) -> TrainerControl:
        """Called at the end of training."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        logger.info("=== Training Summary ===")
        logger.info(f"Training completed at: {end_time}")
        logger.info(f"Total duration: {duration}")
        logger.info(f"Total steps: {state.global_step}")
        logger.info(f"Best reward achieved: {self.best_reward}")
        return control
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl) -> TrainerControl:
        """Called at the beginning of each step."""
        if state.global_step % args.logging_steps == 0:
            self._log_memory_stats()
        return control
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl) -> TrainerControl:
        """Called at the end of each step."""
        self.training_steps += 1
        if self.training_steps % args.logging_steps == 0:
            logger.info(f"Step {self.training_steps}: loss={state.log_history[-1].get('loss', 'N/A')}")
        return control
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl) -> TrainerControl:
        """Called during evaluation."""
        logger.info("=== Evaluation ===")
        if state.log_history:
            metrics = state.log_history[-1]
            logger.info(f"Evaluation metrics: {metrics}")
        return control
    
    def _log_memory_stats(self):
        """Log current memory statistics."""
        if torch.cuda.is_available():
            gpu_memory = {
                'allocated': f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB",
                'reserved': f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB",
                'max_allocated': f"{torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
            }
            logger.info(f"GPU Memory Stats: {gpu_memory}")
        
        cpu_memory = {
            'used': f"{psutil.Process().memory_info().rss / 1024**2:.2f} MB",
            'percent': f"{psutil.virtual_memory().percent}%"
        }
        logger.info(f"CPU Memory Stats: {cpu_memory}")
    
    def log_system_info(self):
        """Log system information including hardware and software details."""
        logger.info("=== System Information ===")
        logger.info(f"OS: {platform.system()} {platform.release()}")
        logger.info(f"Python Version: {sys.version}")
        logger.info(f"PyTorch Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        logger.info(f"CPU Count: {psutil.cpu_count()}")
        logger.info(f"Memory Available: {psutil.virtual_memory().available / 1e9:.2f} GB")
        logger.info("========================")
    
    def log_model_info(self, model):
        """Log model information including architecture and memory usage."""
        self.logger.info("=== Model Information ===")
        
        if model is None:
            self.logger.error("Cannot log model info: model is None")
            self.logger.info("========================")
            return
        
        self.logger.info(f"Model Type: {type(model).__name__}")
        
        try:
            # Get total parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.logger.info(f"Total Parameters: {total_params:,}")
            self.logger.info(f"Trainable Parameters: {trainable_params:,}")
            
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                memory_reserved = torch.cuda.memory_reserved() / 1e9
                self.logger.info(f"GPU Memory Allocated: {memory_allocated:.2f} GB")
                self.logger.info(f"GPU Memory Reserved: {memory_reserved:.2f} GB")
        except Exception as e:
            self.logger.error(f"Error getting model information: {str(e)}")
        
        self.logger.info("========================")
    
    def log_dataset_info(self, dataset):
        """Log dataset statistics and information."""
        self.logger.info("=== Dataset Information ===")
        
        try:
            # Basic dataset info
            self.logger.info(f"Dataset Size: {len(dataset)}")
            
            # Try to get HuggingFace Dataset features
            try:
                self.logger.info(f"Features: {dataset.features}")
                self.logger.info(f"Column Names: {dataset.column_names}")
            except AttributeError:
                self.logger.info("Dataset does not have HuggingFace features/columns")
            
            # Sample and log first example
            try:
                first_example = dataset[0]
                self.logger.info("First example structure:")
                for key, value in first_example.items():
                    preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    self.logger.info(f"  {key}: {preview}")
            except Exception as e:
                self.logger.warning(f"Could not log first example: {str(e)}")
            
            # Try to get basic statistics if possible
            if hasattr(dataset, 'column_names'):
                for column in dataset.column_names:
                    try:
                        if isinstance(dataset[0][column], (int, float)):
                            values = [item[column] for item in dataset]
                            self.logger.info(f"{column} stats:")
                            self.logger.info(f"  Mean: {sum(values)/len(values):.2f}")
                            self.logger.info(f"  Min: {min(values)}")
                            self.logger.info(f"  Max: {max(values)}")
                    except Exception as e:
                        self.logger.debug(f"Could not compute stats for column {column}: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error logging dataset info: {str(e)}")
        
        self.logger.info("=========================")
    
    def log_batch_processing(self, batch: Dict[str, Any]):
        """Log information about batch processing."""
        self.logger.info("=== Batch Processing ===")
        self.logger.info(f"Batch Keys: {list(batch.keys())}")
        
        for key, value in batch.items():
            if isinstance(value, (str, int, float)):
                self.logger.info(f"{key} (type: {type(value).__name__}): {value}")
            elif hasattr(value, 'shape'):
                self.logger.info(f"{key} shape: {value.shape}")
        
        self.logger.info("========================")
    
    def log_arabic_processing(self, text: str, tokenizer: PreTrainedTokenizer):
        """Log Arabic text processing details."""
        self.logger.info("=== Arabic Processing ===")
        self.logger.info(f"Original text: {text}")
        
        # Tokenization info
        tokens = tokenizer.tokenize(text)
        self.logger.info(f"Token count: {len(tokens)}")
        self.logger.info(f"Tokens: {tokens}")
        
        # Special character handling
        special_chars = [char for char in text if ord(char) > 127]
        self.logger.info(f"Special characters: {special_chars}")
        
        self.logger.info("========================") 