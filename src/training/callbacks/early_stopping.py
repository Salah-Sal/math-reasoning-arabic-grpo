from typing import Optional
from src.training.callbacks.base import BaseCallback
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

class EarlyStoppingCallback(BaseCallback):
    """Callback for early stopping based on reward improvement.
    
    This callback monitors the reward during training and stops training
    if no improvement is seen for a specified number of steps.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_improvement: float = 0.01,
        min_steps: int = 0,
        order: int = 0
    ):
        """Initialize early stopping callback.
        
        Args:
            patience: Number of steps to wait for improvement before stopping
            min_improvement: Minimum improvement in reward to reset patience
            min_steps: Minimum number of steps before allowing early stopping
            order: Callback execution order
        """
        super().__init__(order=order)
        
        # Validate parameters
        if patience < 0:
            raise ValueError("Patience must be non-negative")
        if min_improvement < 0:
            raise ValueError("Minimum improvement must be non-negative")
        if min_steps < 0:
            raise ValueError("Minimum steps must be non-negative")
        
        self.patience = patience
        self.min_improvement = min_improvement
        self.min_steps = min_steps
        
        # Initialize tracking variables
        self._best_reward = float('-inf')
        self._steps_without_improvement = 0
        
        logger.info(
            f"Initialized early stopping callback:\n"
            f"  Patience: {patience} steps\n"
            f"  Min improvement: {min_improvement}\n"
            f"  Min steps: {min_steps}"
        )
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self._best_reward = float('-inf')
        self._steps_without_improvement = 0
        logger.info("Reset early stopping state")
    
    def _check_improvement(self, current_reward: float) -> bool:
        """Check if current reward shows sufficient improvement.
        
        Args:
            current_reward: Current reward value
            
        Returns:
            True if reward shows sufficient improvement
        """
        improvement = current_reward - self._best_reward
        return improvement >= self.min_improvement
    
    def _on_step_end(self) -> None:
        """Check for early stopping conditions at end of step."""
        if not self.is_active:
            return
            
        current_step = self.trainer.state.current_step
        current_reward = self.trainer.state.current_reward
        
        # Don't stop before minimum steps
        if current_step < self.min_steps:
            logger.info(f"Step {current_step}: Minimum steps not reached")
            return
        
        # Check for improvement
        if self._check_improvement(current_reward):
            self._best_reward = current_reward
            self._steps_without_improvement = 0
            logger.info(
                f"Step {current_step}: New best reward {current_reward:.4f}"
            )
        else:
            self._steps_without_improvement += 1
            logger.info(
                f"Step {current_step}: No improvement for {self._steps_without_improvement} steps"
                f" (current: {current_reward:.4f}, best: {self._best_reward:.4f})"
            )
        
        # Check stopping condition
        if self._steps_without_improvement >= self.patience:
            logger.info(
                f"Stopping early at step {current_step}:\n"
                f"  No improvement for {self.patience} steps\n"
                f"  Best reward: {self._best_reward:.4f}\n"
                f"  Current reward: {current_reward:.4f}"
            )
            self.trainer.stop_training()
    
    def _on_training_start(self) -> None:
        """Reset state at start of training."""
        self.reset()
        logger.info("Reset early stopping state at training start")
    
    def _on_training_end(self) -> None:
        """Clean up at end of training."""
        self.reset()
        logger.info("Reset early stopping state at training end") 