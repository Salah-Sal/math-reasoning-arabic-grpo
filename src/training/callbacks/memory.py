import gc
import torch
from typing import Dict, Optional
from src.training.callbacks.base import BaseCallback
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

class MemoryMonitorCallback(BaseCallback):
    """Callback for monitoring GPU memory usage during training.
    
    This callback tracks GPU memory usage and can trigger warnings or take action
    when memory usage exceeds specified thresholds.
    """
    
    def __init__(
        self,
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.9,
        logging_steps: int = 10,
        order: int = 0
    ):
        """Initialize the memory monitor.
        
        Args:
            warning_threshold: Memory usage ratio to trigger warnings
            critical_threshold: Memory usage ratio to trigger critical actions
            logging_steps: Number of steps between memory checks
            order: Callback execution order
        """
        super().__init__(order=order)
        
        # Validate thresholds
        if not 0 < warning_threshold < critical_threshold <= 1.0:
            raise ValueError(
                f"Invalid thresholds: warning ({warning_threshold}) must be less than "
                f"critical ({critical_threshold}) and both must be between 0 and 1"
            )
        
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.logging_steps = logging_steps
        self._last_memory_check: Optional[Dict] = None
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - memory monitoring will be disabled")
    
    def _check_memory(self) -> Dict:
        """Check current GPU memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        if not torch.cuda.is_available():
            return {}
            
        try:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            max_allocated = torch.cuda.max_memory_allocated()
            
            memory_stats = {
                'allocated': allocated,
                'reserved': reserved,
                'max_allocated': max_allocated,
                'utilization': allocated / max_allocated if max_allocated > 0 else 0
            }
            
            return memory_stats
        except Exception as e:
            logger.error(f"Error checking GPU memory: {str(e)}")
            return {}
    
    def _handle_memory_status(self, memory_stats: Dict) -> None:
        """Handle memory status and take appropriate action.
        
        Args:
            memory_stats: Dictionary of memory statistics
        """
        if not memory_stats:
            return
            
        utilization = memory_stats['utilization']
        
        if utilization >= self.critical_threshold:
            logger.warning(
                f"CRITICAL: GPU memory utilization at {utilization:.1%}. "
                "Attempting emergency cleanup..."
            )
            self._emergency_cleanup()
        elif utilization >= self.warning_threshold:
            logger.warning(
                f"WARNING: High GPU memory utilization at {utilization:.1%}"
            )
    
    def _emergency_cleanup(self) -> None:
        """Perform emergency memory cleanup."""
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            # Force garbage collection
            gc.collect()
            
            logger.info("Completed emergency memory cleanup")
    
    def _on_step_begin(self) -> None:
        """Check memory at the start of each step."""
        if not torch.cuda.is_available():
            return
            
        current_step = self.trainer.state.current_step
        
        # Only check memory at specified intervals
        if current_step % self.logging_steps == 0:
            memory_stats = self._check_memory()
            self._last_memory_check = memory_stats
            self._handle_memory_status(memory_stats)
            
            # Log detailed memory information
            if memory_stats:
                logger.info(
                    f"Step {current_step} memory stats:\n"
                    f"  Allocated: {memory_stats['allocated'] / 1024**2:.1f}MB\n"
                    f"  Reserved:  {memory_stats['reserved'] / 1024**2:.1f}MB\n"
                    f"  Max:       {memory_stats['max_allocated'] / 1024**2:.1f}MB\n"
                    f"  Usage:     {memory_stats['utilization']:.1%}"
                )
    
    def _on_training_end(self) -> None:
        """Clean up memory when training ends."""
        if torch.cuda.is_available():
            logger.info("Cleaning up GPU memory after training")
            torch.cuda.empty_cache()
            gc.collect() 