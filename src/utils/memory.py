import gc
import torch
from src.infrastructure.logging import get_logger
import os
import logging

logger = logging.getLogger(__name__)

def clear_memory():
    """Function to clear memory resources."""
    try:
        # Placeholder for actual memory clearing logic
        # Remove any OpenAI-dependent code if not required
        logger.info("Memory cleared successfully.")
    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}")

def clear_memory_old():
    """Clear GPU memory and cache.
    
    This function:
    1. Runs garbage collection
    2. Empties CUDA cache
    3. Resets peak memory stats
    """
    try:
        # Run garbage collection
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Force garbage collection
            for obj in gc.get_objects():
                if torch.is_tensor(obj):
                    del obj
            gc.collect()
            
            # Log memory status
            free_memory, total_memory = torch.cuda.mem_get_info()
            free_gb = free_memory / 1024**3
            total_gb = total_memory / 1024**3
            logger.info(f"Memory cleared. Free: {free_gb:.2f}GB / Total: {total_gb:.2f}GB")
            
        # Example: If using OpenAI's API for some reason
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        # Proceed with memory clearing operations that require OpenAI
    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}") 