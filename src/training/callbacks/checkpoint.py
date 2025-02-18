import shutil
import os
from pathlib import Path
from typing import List, Optional
from src.training.callbacks.base import BaseCallback
from src.infrastructure.logging import get_logger
from transformers import TrainingArguments, TrainerState, TrainerControl

logger = get_logger(__name__)

class ModelCheckpointCallback(TrainerCallback, BaseCallback):
    """Callback for saving model checkpoints during training.
    
    This callback handles saving model checkpoints at regular intervals,
    maintaining the best model based on rewards, and managing checkpoint rotation.
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        save_steps: int = 500,
        max_checkpoints: Optional[int] = None,
        save_best: bool = True,
        save_final: bool = True,
        order: int = 0
    ):
        """Initialize the checkpoint callback.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_steps: Number of steps between checkpoints
            max_checkpoints: Maximum number of checkpoints to keep (None for unlimited)
            save_best: Whether to save best model based on reward
            save_final: Whether to save final model after training
            order: Callback execution order
        """
        BaseCallback.__init__(self)
        TrainerCallback.__init__(self)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_steps = save_steps
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.save_final = save_final
        self.order = order
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track saved checkpoints
        self._saved_checkpoints: List[Path] = []
        self._best_reward: float = float('-inf')
        self._trainer = None
        self._model = None
        
        logger.info(
            f"Initialized checkpoint callback:\n"
            f"  Directory: {self.checkpoint_dir}\n"
            f"  Save frequency: {self.save_steps} steps\n"
            f"  Max checkpoints: {max_checkpoints or 'unlimited'}"
        )
    
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> TrainerControl:
        """Called at the end of trainer initialization."""
        logger.info("=== Checkpoint Callback Initialization ===")
        logger.info(f"Args: {args}")
        logger.info(f"State: {state}")
        logger.info(f"Additional kwargs: {kwargs}")
        
        # Store model reference if provided
        if 'model' in kwargs:
            self._model = kwargs['model']
            logger.info(f"Model type stored: {type(self._model)}")
            
            # Log PEFT configuration if available
            if hasattr(self._model, 'peft_config'):
                logger.info(f"PEFT config: {self._model.peft_config}")
        
        # Initialize checkpoint tracking
        self._saved_checkpoints = []
        self._best_reward = float('-inf')
        
        # Verify directory structure
        logger.info(f"Checkpoint directory exists: {self.checkpoint_dir.exists()}")
        logger.info(f"Checkpoint directory is writable: {os.access(self.checkpoint_dir, os.W_OK)}")
        
        return control

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> TrainerControl:
        """Called at the beginning of training."""
        logger.info("=== Starting Checkpoint Tracking ===")
        logger.info(f"Initial state: {state}")
        
        # Log GRPO-specific configuration if available
        if hasattr(args, 'grpo_config'):
            logger.info(f"GRPO config: {args.grpo_config}")
        
        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> TrainerControl:
        """Save checkpoint at regular intervals and check for best model."""
        current_step = state.global_step
        
        # Log step information
        if current_step % self.save_steps == 0:
            logger.info(f"=== Checkpoint Step {current_step} ===")
            
            # Check for GRPO-specific metrics
            current_reward = kwargs.get('reward', None)
            if current_reward is not None:
                logger.info(f"Current reward: {current_reward}")
                
                # Update best reward if applicable
                if current_reward > self._best_reward:
                    self._best_reward = current_reward
                    logger.info(f"New best reward: {self._best_reward}")
                    
                    if self.save_best:
                        best_path = self.checkpoint_dir / "checkpoint-best"
                        self._save_checkpoint(best_path, is_best=True)
            
            # Regular checkpoint saving
            checkpoint_path = self.checkpoint_dir / f"checkpoint-{current_step}"
            self._save_checkpoint(checkpoint_path)
            
            # Log checkpoint information
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            logger.info(f"Total checkpoints: {len(self._saved_checkpoints)}")
        
        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> TrainerControl:
        """Save final checkpoint at end of training."""
        logger.info("=== Training End Checkpoint ===")
        
        if self.save_final:
            final_path = self.checkpoint_dir / "checkpoint-final"
            self._save_checkpoint(final_path)
            logger.info(f"Saved final checkpoint: {final_path}")
        
        # Log final statistics
        logger.info(f"Total checkpoints saved: {len(self._saved_checkpoints)}")
        logger.info(f"Best reward achieved: {self._best_reward}")
        
        # Log GRPO-specific final metrics if available
        if 'final_metrics' in kwargs:
            logger.info(f"Final GRPO metrics: {kwargs['final_metrics']}")
        
        return control

    def _save_checkpoint(self, path: Path, is_best: bool = False) -> None:
        """Save a model checkpoint with enhanced logging."""
        try:
            logger.info(f"Saving checkpoint to {path}")
            logger.info(f"Is best model: {is_best}")
            
            # Verify model state before saving
            if self._model is None:
                raise ValueError("No model available for saving")
            
            # Save model and tokenizer
            self._model.save_pretrained(path)
            
            # Track checkpoint
            if "checkpoint-" in path.name:
                if is_best:
                    self._saved_checkpoints = [p for p in self._saved_checkpoints 
                                             if p.name != "checkpoint-best"]
                    self._saved_checkpoints.append(path)
                elif not path.name.endswith("final"):
                    self._saved_checkpoints.append(path)
                    
                    # Rotate checkpoints if needed
                    if self.max_checkpoints and len(self._saved_checkpoints) > self.max_checkpoints:
                        self._cleanup_old_checkpoints()
            
            logger.info(f"Successfully saved checkpoint to {path}")
            logger.info(f"Current checkpoint count: {len(self._saved_checkpoints)}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint to {path}: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            raise
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove oldest checkpoints when max_checkpoints is exceeded."""
        try:
            # Sort checkpoints by step number
            self._saved_checkpoints.sort(
                key=lambda p: int(p.name.split('-')[-1])
            )
            
            # Remove oldest checkpoints
            while len(self._saved_checkpoints) > self.max_checkpoints:
                checkpoint_to_remove = self._saved_checkpoints.pop(0)
                if checkpoint_to_remove.exists():
                    shutil.rmtree(checkpoint_to_remove)
                    logger.info(f"Removed old checkpoint: {checkpoint_to_remove}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up checkpoints: {str(e)}")
    
    def _check_save_best(self) -> None:
        """Check and save best model if current reward is best so far."""
        current_reward = self.trainer.state.current_reward
        
        if current_reward > self._best_reward:
            self._best_reward = current_reward
            best_path = self.checkpoint_dir / "checkpoint-best"
            self._save_checkpoint(best_path, is_best=True)
            logger.info(
                f"New best model with reward {current_reward:.3f} "
                f"(previous best: {self._best_reward:.3f})"
            )
    
    def _on_step_end(self) -> None:
        """Save checkpoint at regular intervals and check for best model."""
        current_step = self.trainer.state.current_step
        
        # Save at regular intervals
        if current_step % self.save_steps == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint-{current_step}"
            self._save_checkpoint(checkpoint_path)
        
        # Check for best model (only if save_best is True)
        if self.save_best and self.trainer.state.current_reward > self._best_reward:
            self._check_save_best()
    
    def _on_training_end(self) -> None:
        """Save final checkpoint at end of training."""
        if self.save_final:
            final_path = self.checkpoint_dir / "checkpoint-final"
            self._save_checkpoint(final_path)
            logger.info("Saved final checkpoint") 