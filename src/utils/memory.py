import gc
import torch
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

def clear_memory():
    """Clear GPU memory and garbage collect."""
    try:
        # Force Python garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Force garbage collection of CUDA tensors
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj):
                        del obj
                except:
                    pass
            gc.collect()
            
            # Log memory status
            free_memory, total_memory = torch.cuda.mem_get_info()
            free_gb = free_memory / 1024**3
            total_gb = total_memory / 1024**3
            logger.info(f"Memory cleared. Free: {free_gb:.2f}GB / Total: {total_gb:.2f}GB")
            
    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}") 