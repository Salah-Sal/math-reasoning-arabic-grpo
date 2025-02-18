import shutil
from pathlib import Path
from typing import List, Optional
from src.training.callbacks.base import BaseCallback
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

class ModelCheckpointCallback(BaseCallback):
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
        super().__init__(order=order)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_steps = save_steps
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.save_final = save_final
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track saved checkpoints
        self._saved_checkpoints: List[Path] = []
        self._best_reward: float = float('-inf')
        
        logger.info(
            f"Initialized checkpoint callback:\n"
            f"  Directory: {self.checkpoint_dir}\n"
            f"  Save frequency: {self.save_steps} steps\n"
            f"  Max checkpoints: {max_checkpoints or 'unlimited'}"
        )
    
    def _save_checkpoint(self, path: Path, is_best: bool = False) -> None:
        """Save a model checkpoint.
        
        Args:
            path: Path to save the checkpoint
            is_best: Whether this is the best model so far
        """
        try:
            # Save model and tokenizer
            self.trainer.model.save_pretrained(path)
            logger.info(f"Saved {'best ' if is_best else ''}checkpoint to {path}")
            
            # Add to tracked checkpoints if not best/final
            if not is_best and "checkpoint-" in path.name:
                self._saved_checkpoints.append(path)
                
                # Rotate checkpoints if needed
                if self.max_checkpoints and len(self._saved_checkpoints) > self.max_checkpoints:
                    self._cleanup_old_checkpoints()
                    
        except Exception as e:
            logger.error(f"Error saving checkpoint to {path}: {str(e)}")
    
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
        
        # Check for best model
        if self.save_best:
            self._check_save_best()
    
    def _on_training_end(self) -> None:
        """Save final checkpoint at end of training."""
        if self.save_final:
            final_path = self.checkpoint_dir / "checkpoint-final"
            self._save_checkpoint(final_path)
            logger.info("Saved final checkpoint") 