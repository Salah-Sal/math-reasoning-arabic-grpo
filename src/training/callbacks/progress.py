import time
from typing import Optional
from src.training.callbacks.base import BaseCallback
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

class TrainingProgressCallback(BaseCallback):
    """Callback for monitoring and logging training progress.
    
    This callback tracks training progress, estimates time remaining,
    and logs various metrics during training.
    """
    
    def __init__(self, logging_steps: int = 10, order: int = 0):
        """Initialize the progress monitor.
        
        Args:
            logging_steps: Number of steps between progress updates
            order: Callback execution order
        """
        super().__init__(order=order)
        self.logging_steps = logging_steps
        self._start_time: Optional[float] = None
        self._last_log_time: Optional[float] = None
    
    def _get_progress_percentage(self) -> float:
        """Calculate training progress percentage.
        
        Returns:
            Progress as a percentage
        """
        current = self.trainer.state.current_step
        total = self.trainer.state.max_steps
        return (current / total) * 100 if total > 0 else 0.0
    
    def _estimate_total_time(self) -> float:
        """Estimate total training time based on current progress.
        
        Returns:
            Estimated total time in seconds
        """
        if not self._start_time or self.trainer.state.current_step == 0:
            return 0.0
            
        elapsed_time = time.time() - self._start_time
        progress = self.trainer.state.current_step / self.trainer.state.max_steps
        
        if progress > 0:
            return elapsed_time / progress
        return 0.0
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to a human-readable string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _format_metrics(self) -> str:
        """Format current training metrics.
        
        Returns:
            Formatted metrics string
        """
        state = self.trainer.state
        return (
            f"loss: {state.loss:.3f} | "
            f"reward: {state.current_reward:.3f} | "
            f"best_reward: {state.best_reward:.3f}"
        )
    
    def _on_training_start(self) -> None:
        """Log information at training start."""
        self._start_time = time.time()
        self._last_log_time = self._start_time
        
        logger.info(
            f"Starting training for {self.trainer.state.max_steps} steps\n"
            f"Logging every {self.logging_steps} steps"
        )
    
    def _on_step_end(self) -> None:
        """Log progress at the end of each step."""
        current_step = self.trainer.state.current_step
        
        # Only log at specified intervals
        if current_step % self.logging_steps != 0:
            return
            
        current_time = time.time()
        progress = self._get_progress_percentage()
        
        # Calculate time statistics
        elapsed = current_time - self._start_time
        time_per_step = elapsed / current_step if current_step > 0 else 0
        steps_remaining = self.trainer.state.max_steps - current_step
        estimated_remaining = time_per_step * steps_remaining
        
        # Update last log time
        self._last_log_time = current_time
        
        # Log progress
        logger.info(
            f"Step [{current_step}/{self.trainer.state.max_steps}] "
            f"({progress:.1f}%) | "
            f"{self._format_metrics()} | "
            f"Elapsed: {self._format_time(elapsed)} | "
            f"Remaining: {self._format_time(estimated_remaining)}"
        )
    
    def _on_training_end(self) -> None:
        """Log summary at training end."""
        if not self._start_time:
            return
            
        total_time = time.time() - self._start_time
        completed_steps = self.trainer.state.current_step
        total_steps = self.trainer.state.max_steps
        
        early_stop = completed_steps < total_steps
        completion_status = "EARLY STOP" if early_stop else "COMPLETED"
        
        logger.info(
            f"\nTraining {completion_status}\n"
            f"Steps: {completed_steps}/{total_steps} "
            f"({(completed_steps/total_steps)*100:.1f}%)\n"
            f"Best Reward: {self.trainer.state.best_reward:.3f}\n"
            f"Total Time: {self._format_time(total_time)}\n"
            f"Average Time per Step: {self._format_time(total_time/completed_steps)}"
        ) 