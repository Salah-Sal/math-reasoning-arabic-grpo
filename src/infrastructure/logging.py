import logging
import logging.config
import yaml
from pathlib import Path
from typing import Optional

def setup_logging(
    config_path: Optional[Path] = None,
    default_level: int = logging.INFO
) -> None:
    """Initialize logging configuration"""
    if config_path is None:
        config_path = Path("configs/logging_config.yaml")
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config['default'])
    except Exception as e:
        print(f"Error loading logging configuration: {str(e)}")
        print("Using basic configuration instead.")
        logging.basicConfig(
            level=default_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name"""
    return logging.getLogger(f"math_reasoning.{name}") 