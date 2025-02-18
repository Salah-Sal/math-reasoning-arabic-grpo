from typing import Optional, Any
from abc import ABC
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

class BaseCallback(ABC):
    """Base class for training callbacks.
    
    Callbacks can be used to monitor and modify training behavior by implementing
    hooks that are called at specific points during training.
    """
    
    def __init__(self, order: int = 0):
        """Initialize callback.
        
        Args:
            order: Execution order for multiple callbacks (lower = earlier)
        """
        if order < 0:
            raise ValueError("Callback order must be non-negative")
        
        self.order = order
        self._trainer = None
        self._is_active = True
        logger.debug(f"Initialized {self.__class__.__name__} with order {order}")
    
    @property
    def trainer(self) -> Any:
        """Get the bound trainer instance."""
        if self._trainer is None:
            raise RuntimeError("Callback not bound to any trainer")
        return self._trainer
    
    @property
    def is_active(self) -> bool:
        """Check if callback is active."""
        return self._is_active
    
    def activate(self) -> None:
        """Activate the callback."""
        self._is_active = True
        logger.debug(f"Activated {self.__class__.__name__}")
    
    def deactivate(self) -> None:
        """Deactivate the callback."""
        self._is_active = False
        logger.debug(f"Deactivated {self.__class__.__name__}")
    
    def bind_trainer(self, trainer: Any) -> None:
        """Bind callback to a trainer instance.
        
        Args:
            trainer: The trainer instance to bind to
        """
        self._trainer = trainer
        logger.debug(f"Bound {self.__class__.__name__} to trainer")
    
    def __lt__(self, other: 'BaseCallback') -> bool:
        """Compare callbacks for sorting by order."""
        return self.order < other.order
    
    def _call_event(self, event_name: str) -> None:
        """Safely call an event method with error handling.
        
        Args:
            event_name: Name of the event method to call
        """
        if not self.is_active:
            return
            
        try:
            event_method = getattr(self, event_name)
            event_method()
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}.{event_name}: {str(e)}")
    
    def on_training_start(self) -> None:
        """Called when training starts."""
        if self.is_active:
            try:
                self._on_training_start()
            except Exception as e:
                logger.error(f"Error in {self.__class__.__name__}.on_training_start: {str(e)}")
    
    def on_training_end(self) -> None:
        """Called when training ends."""
        if self.is_active:
            try:
                self._on_training_end()
            except Exception as e:
                logger.error(f"Error in {self.__class__.__name__}.on_training_end: {str(e)}")
    
    def on_step_begin(self) -> None:
        """Called at the beginning of each training step."""
        if self.is_active:
            try:
                self._on_step_begin()
            except Exception as e:
                logger.error(f"Error in {self.__class__.__name__}.on_step_begin: {str(e)}")
    
    def on_step_end(self) -> None:
        """Called at the end of each training step."""
        if self.is_active:
            try:
                self._on_step_end()
            except Exception as e:
                logger.error(f"Error in {self.__class__.__name__}.on_step_end: {str(e)}")
    
    def on_evaluate(self) -> None:
        """Called during model evaluation."""
        if self.is_active:
            try:
                self._on_evaluate()
            except Exception as e:
                logger.error(f"Error in {self.__class__.__name__}.on_evaluate: {str(e)}")
    
    def on_save(self) -> None:
        """Called when model is being saved."""
        if self.is_active:
            try:
                self._on_save()
            except Exception as e:
                logger.error(f"Error in {self.__class__.__name__}.on_save: {str(e)}")
    
    # Protected event methods for subclasses to override
    def _on_training_start(self) -> None:
        """Implementation of training start event."""
        pass
    
    def _on_training_end(self) -> None:
        """Implementation of training end event."""
        pass
    
    def _on_step_begin(self) -> None:
        """Implementation of step begin event."""
        pass
    
    def _on_step_end(self) -> None:
        """Implementation of step end event."""
        pass
    
    def _on_evaluate(self) -> None:
        """Implementation of evaluation event."""
        pass
    
    def _on_save(self) -> None:
        """Implementation of save event."""
        pass 