import logging
import torch
from typing import Any, Dict, Optional
from pathlib import Path
import psutil
import os
import sys
import platform
from datetime import datetime
from transformers import PreTrainedTokenizer
from datasets import Dataset
from src.training.callbacks.base import BaseCallback

logger = logging.getLogger(__name__)

class TrainingMonitor(BaseCallback):
    """Comprehensive monitoring system for training process."""
    
    def __init__(self, log_dir: Path):
        super().__init__()
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
    
    def log_system_info(self):
        """Log system information including hardware and software details."""
        self.logger.info("=== System Information ===")
        self.logger.info(f"OS: {platform.system()} {platform.release()}")
        self.logger.info(f"Python Version: {sys.version}")
        self.logger.info(f"PyTorch Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        self.logger.info(f"CPU Count: {psutil.cpu_count()}")
        self.logger.info(f"Memory Available: {psutil.virtual_memory().available / 1e9:.2f} GB")
        self.logger.info("========================")
    
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
    
    def log_dataset_info(self, dataset: Dataset):
        """Log dataset statistics and information."""
        self.logger.info("=== Dataset Information ===")
        self.logger.info(f"Dataset Size: {len(dataset)}")
        self.logger.info(f"Features: {dataset.features}")
        self.logger.info(f"Column Names: {dataset.column_names}")
        
        # Log sample counts and basic statistics
        if len(dataset) > 0:
            for column in dataset.column_names:
                if isinstance(dataset[0][column], (int, float)):
                    values = [item[column] for item in dataset]
                    self.logger.info(f"{column} stats:")
                    self.logger.info(f"  Mean: {sum(values)/len(values):.2f}")
                    self.logger.info(f"  Min: {min(values)}")
                    self.logger.info(f"  Max: {max(values)}")
        
        self.logger.info("========================")
    
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
    
    def _on_training_start(self):
        """Log information at the start of training."""
        self.start_time = datetime.now()
        self.logger.info(f"Training started at {self.start_time}")
        
        if self.trainer:
            self.logger.info("=== Training Configuration ===")
            self.logger.info(f"Learning rate: {self.trainer.args.learning_rate}")
            self.logger.info(f"Batch size: {self.trainer.args.per_device_train_batch_size}")
            self.logger.info(f"Max steps: {self.trainer.args.max_steps}")
            self.logger.info("========================")
    
    def _on_step_end(self, args: Optional[Dict[str, Any]] = None):
        """Log training progress and metrics."""
        self.training_steps += 1
        
        if args and 'metrics' in args:
            metrics = args['metrics']
            self.logger.info(f"Step {self.training_steps} metrics:")
            for key, value in metrics.items():
                self.logger.info(f"  {key}: {value}")
            
            # Track best reward
            if 'reward' in metrics:
                current_reward = metrics['reward']
                if current_reward > self.best_reward:
                    self.best_reward = current_reward
                    self.logger.info(f"New best reward: {self.best_reward}")
        
        # Log memory usage periodically
        if self.training_steps % 100 == 0 and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            self.logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
    
    def _on_training_end(self):
        """Log final training statistics."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.logger.info("=== Training Summary ===")
        self.logger.info(f"Training completed at: {end_time}")
        self.logger.info(f"Total duration: {duration}")
        self.logger.info(f"Total steps: {self.training_steps}")
        self.logger.info(f"Best reward achieved: {self.best_reward}")
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            self.logger.info(f"Final GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
        
        self.logger.info("========================") 