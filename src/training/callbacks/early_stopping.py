from typing import Optional
from src.training.callbacks.base import BaseCallback
from src.infrastructure.logging import get_logger
from transformers import (
    TrainingArguments,
    TrainerState,
    TrainerControl,
    TrainerCallback,
    __version__ as transformers_version
)

logger = get_logger(__name__)

class EarlyStoppingCallback(TrainerCallback, BaseCallback):
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
        logger.info("Initializing EarlyStoppingCallback")
        logger.info(f"Callback bases: {self.__class__.__bases__}")
        
        # Initialize both parent classes
        try:
            BaseCallback.__init__(self)
            TrainerCallback.__init__(self)
            logger.info("Successfully initialized parent classes")
        except Exception as e:
            logger.error(f"Error initializing parent classes: {str(e)}")
            raise
        
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
        self.order = order
        
        # Initialize tracking variables
        self._best_reward = float('-inf')
        self._steps_without_improvement = 0
        self._model = None
        self._trainer = None
        
        logger.info(
            f"Initialized early stopping callback:\n"
            f"  Patience: {patience} steps\n"
            f"  Min improvement: {min_improvement}\n"
            f"  Min steps: {min_steps}"
        )
    
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> TrainerControl:
        """Called at the end of trainer initialization."""
        logger.info("=== Early Stopping Initialization ===")
        logger.info(f"Args: {args}")
        logger.info(f"State: {state}")
        logger.info(f"Additional kwargs: {kwargs}")
        
        # Store model reference if provided
        if 'model' in kwargs:
            self._model = kwargs['model']
            logger.info(f"Model type stored: {type(self._model)}")
        
        # Reset tracking variables
        self.reset()
        
        # Log GRPO-specific configuration if available
        if hasattr(args, 'grpo_config'):
            logger.info(f"GRPO config: {args.grpo_config}")
        
        return control

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> TrainerControl:
        """Called at the beginning of training."""
        logger.info("=== Starting Early Stopping Monitor ===")
        logger.info(f"Initial state: {state}")
        self.reset()
        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> TrainerControl:
        """Check for early stopping conditions at end of step."""
        if not self.is_active:
            return control
            
        current_step = state.global_step
        
        # Don't stop before minimum steps
        if current_step < self.min_steps:
            logger.debug(f"Step {current_step}: Minimum steps not reached")
            return control
        
        # Get current reward from GRPO-specific metrics
        current_reward = kwargs.get('reward', None)
        if current_reward is not None:
            logger.info(f"Step {current_step} - Current reward: {current_reward}")
            
            # Check for improvement
            if self._check_improvement(current_reward):
                self._best_reward = current_reward
                self._steps_without_improvement = 0
                logger.info(f"New best reward: {current_reward:.4f}")
            else:
                self._steps_without_improvement += 1
                logger.info(
                    f"No improvement for {self._steps_without_improvement} steps "
                    f"(current: {current_reward:.4f}, best: {self._best_reward:.4f})"
                )
            
            # Check stopping condition
            if self._steps_without_improvement >= self.patience:
                logger.info(
                    f"Stopping early at step {current_step}:\n"
                    f"  No improvement for {self.patience} steps\n"
                    f"  Best reward: {self._best_reward:.4f}\n"
                    f"  Current reward: {current_reward:.4f}"
                )
                control.should_training_stop = True
        
        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> TrainerControl:
        """Called at the end of training."""
        logger.info("=== Early Stopping Summary ===")
        logger.info(f"Best reward achieved: {self._best_reward:.4f}")
        logger.info(f"Steps without improvement: {self._steps_without_improvement}")
        logger.info(f"Total steps: {state.global_step}")
        
        # Log GRPO-specific final metrics if available
        if 'final_metrics' in kwargs:
            logger.info(f"Final GRPO metrics: {kwargs['final_metrics']}")
        
        return control

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
        logger.debug(f"Improvement check: {improvement} (threshold: {self.min_improvement})")
        return improvement >= self.min_improvement 