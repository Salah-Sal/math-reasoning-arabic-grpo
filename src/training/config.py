from pathlib import Path
from typing import Optional, Dict, Any, Literal, List, Union
from pydantic import BaseModel, Field, validator, ConfigDict
import yaml
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

class TrainingSettings(BaseModel):
    """Training hyperparameters and settings."""
    learning_rate: float = Field(
        default=3e-6,
        gt=0.0,
        description="Learning rate for training"
    )
    max_steps: int = Field(
        default=1000,
        gt=0,
        description="Maximum number of training steps"
    )
    per_device_train_batch_size: int = Field(
        default=8,
        gt=0,
        description="Batch size per device"
    )
    gradient_accumulation_steps: int = Field(
        default=4,
        gt=0,
        description="Number of steps to accumulate gradients"
    )
    max_prompt_length: int = Field(
        default=256,
        gt=0,
        description="Maximum length of input prompts"
    )
    max_completion_length: int = Field(
        default=128,
        gt=0,
        description="Maximum length of completions"
    )
    logging_steps: int = Field(
        default=10,
        gt=0,
        description="Number of steps between logging"
    )
    save_steps: int = Field(
        default=100,
        gt=0,
        description="Number of steps between saving checkpoints"
    )
    warmup_steps: int = Field(
        default=100,
        ge=0,
        description="Number of warmup steps"
    )
    max_checkpoints: int = Field(
        default=3,
        gt=0,
        description="Maximum number of checkpoints to keep"
    )
    report_to: str = Field(
        default="none",
        description="Where to report results (none/wandb/tensorboard)"
    )

    @property
    def batch_size(self) -> int:
        """Alias for per_device_train_batch_size to maintain compatibility."""
        return self.per_device_train_batch_size
    
    def model_dump(self) -> Dict[str, Any]:
        """Enhanced model dump with logging."""
        data = super().model_dump()
        logger.info("Training settings:")
        for key, value in data.items():
            logger.info(f"  {key}: {value}")
        return data

class MemorySettings(BaseModel):
    """Memory management settings."""
    gpu_memory_utilization: float = Field(
        default=0.7,
        gt=0.0,
        le=1.0,
        description="Target GPU memory utilization"
    )
    use_gradient_checkpointing: Union[bool, Literal["unsloth"]] = Field(
        default=True,
        description="Whether to use gradient checkpointing. Can be boolean or 'unsloth' for Unsloth-specific implementation"
    )
    optimize_memory_use: bool = Field(
        default=True,
        description="Whether to optimize memory usage"
    )
    max_memory_MB: int = Field(
        default=8000,
        gt=0,
        description="Maximum memory usage in MB"
    )
    dtype: Literal["float16", "bfloat16"] = Field(
        default="float16",
        description="Data type for training"
    )

    @validator('use_gradient_checkpointing')
    def validate_gradient_checkpointing(cls, v):
        """Validate gradient checkpointing setting."""
        if isinstance(v, str) and v.lower() == "unsloth":
            logger.info("Using Unsloth-specific gradient checkpointing")
            return "unsloth"
        elif isinstance(v, bool):
            return v
        else:
            raise ValueError(f"use_gradient_checkpointing must be boolean or 'unsloth', got {type(v)} with value {v}")

class ModelSettings(BaseModel):
    """Model configuration settings."""
    name: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        description="Name or path of the model to use",
        alias="model_name"
    )
    max_seq_length: int = Field(
        default=384,
        description="Maximum sequence length"
    )
    load_in_4bit: bool = Field(
        default=True,
        description="Whether to load model in 4-bit quantization"
    )
    fast_inference: bool = Field(
        default=False,
        description="Whether to use fast inference"
    )
    gpu_memory_utilization: float = Field(
        default=0.7,
        description="GPU memory utilization target"
    )
    use_flash_attention: bool = Field(
        default=True,
        description="Whether to enable flash attention (if supported)"
    )
    flash_attention_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "cross_attention": True,
            "flash_rotary": True,
            "flash_normalization": True,
            "device": "cuda"
        },
        description="Flash attention configuration parameters"
    )

    model_config = ConfigDict(
        extra='allow',  # Allow extra fields
        validate_assignment=True,  # Validate during assignment
        populate_by_name=True,  # Allow population by field name or alias
        alias_generator=None  # No automatic alias generation
    )

    @property
    def model_name(self) -> str:
        """Alias for name to maintain backward compatibility."""
        return self.name

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelSettings":
        """Create instance from dictionary with logging."""
        logger.info("=== Creating ModelSettings from dict ===")
        logger.info(f"Input data: {data}")
        logger.info(f"Available fields: {cls.__fields__.keys()}")
        
        # Handle field name mapping
        if 'model_name' in data and 'name' not in data:
            logger.info("Converting 'model_name' to 'name'")
            data['name'] = data.pop('model_name')
        
        # Log field mapping
        logger.info("=== Field Mapping ===")
        for key in data:
            field_info = cls.__fields__.get(key)
            if field_info:
                logger.info(f"Field '{key}': alias={field_info.alias}, type={field_info.annotation}")
            else:
                logger.warning(f"Extra field '{key}' not in model definition")
        
        # Log any extra fields not in model
        extra_fields = set(data.keys()) - set(cls.__fields__.keys())
        if extra_fields:
            logger.warning(f"Extra fields found in data: {extra_fields}")
        
        # Log LoRA-specific parameters
        lora_params = {k: v for k, v in data.items() 
                      if k in ['lora_rank', 'lora_alpha', 'target_modules', 'lora_dropout']}
        logger.info(f"LoRA parameters found: {lora_params}")
        
        try:
            instance = cls(**data)
            logger.info("=== Created Instance ===")
            logger.info(f"Fields: {instance.model_dump().keys()}")
            logger.info(f"Model name via property: {instance.model_name}")
            logger.info(f"Model name via field: {instance.name}")
            return instance
        except Exception as e:
            logger.error(f"Error creating ModelSettings instance: {str(e)}", exc_info=True)
            raise

    def model_dump(self) -> Dict[str, Any]:
        """Enhanced model dump with logging."""
        data = super().model_dump()
        logger.info("=== Model Settings Dump ===")
        for key, value in data.items():
            logger.info(f"  {key}: {value}")
        return data

    def __init__(self, **data):
        super().__init__(**data)
        logger.info("=== ModelSettings Initialization ===")
        logger.info(f"Received fields: {list(data.keys())}")
        logger.info(f"Flash attention enabled: {self.use_flash_attention}")
        if self.use_flash_attention:
            logger.info(f"Flash attention config: {self.flash_attention_config}")
        logger.info(f"Initialized fields: {self.model_dump().keys()}")
    
    def get_field_value(self, field_name: str, default: Any = None) -> Any:
        """Safely get field value with logging."""
        try:
            value = getattr(self, field_name, default)
            logger.debug(f"Accessing field '{field_name}': {value} (default: {default})")
            return value
        except Exception as e:
            logger.warning(f"Error accessing field '{field_name}': {str(e)}")
            return default

class RewardWeights(BaseModel):
    """Reward function weights."""
    xml_structure: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for XML structure reward"
    )
    format: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for format reward"
    )
    correctness: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for correctness reward"
    )

    @validator('*')
    def validate_weights_sum(cls, v, values):
        """Validate that weights sum to 1.0."""
        if len(values) == 2:  # When processing the last value
            total = sum(values.values()) + v
            if not abs(total - 1.0) < 1e-6:
                raise ValueError(f"Reward weights must sum to 1.0, got {total}")
        return v

class RewardSettings(BaseModel):
    """Reward calculation settings."""
    weights: RewardWeights = Field(
        default_factory=RewardWeights,
        description="Reward function weights"
    )
    cache_size: int = Field(
        default=1000,
        gt=0,
        description="Size of reward calculation cache"
    )
    normalize_rewards: bool = Field(
        default=True,
        description="Whether to normalize rewards"
    )

class PathSettings(BaseModel):
    """Path configuration settings."""
    data_path: Path = Field(
        default=Path("/home/Sal3/ml_data/translations"),
        description="Directory containing the dataset files"
    )
    output_dir: Path = Field(
        default=Path("outputs/grpo_training"),
        description="Directory for training outputs"
    )
    checkpoint_dir: Path = Field(
        default=Path("checkpoints"),
        description="Directory for logs"
    )
    log_dir: Path = Field(
        default=Path("logs"),
        description="Directory for logs"
    )
    cache_dir: Optional[Path] = Field(
        default=Path(".cache"),
        description="Directory for caching"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @validator('*')
    def create_directories(cls, v: Path) -> Path:
        """Ensure directories exist."""
        if v is not None:
            logger.info(f"Validating path: {v}")
            try:
                v.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directory exists or created: {v}")
            except Exception as e:
                logger.warning(f"Could not create directory {v}: {e}")
        return v

class EarlyStoppingSettings(BaseModel):
    """Early stopping configuration."""
    enabled: bool = Field(
        default=True,
        description="Whether to enable early stopping"
    )
    patience: int = Field(
        default=5,
        ge=0,
        description="Number of steps to wait for improvement before stopping"
    )
    min_improvement: float = Field(
        default=0.01,
        gt=0.0,
        description="Minimum improvement in reward to reset patience"
    )
    min_steps: int = Field(
        default=100,
        ge=0,
        description="Minimum number of steps before allowing early stopping"
    )

class GRPOConfig(BaseModel):
    """Main GRPO training configuration."""
    model: ModelSettings = Field(
        default_factory=ModelSettings,
        description="Model settings"
    )
    training: TrainingSettings = Field(
        default_factory=TrainingSettings,
        description="Training settings"
    )
    memory: MemorySettings = Field(
        default_factory=MemorySettings,
        description="Memory management settings"
    )
    reward: RewardSettings = Field(
        default_factory=RewardSettings,
        description="Reward calculation settings"
    )
    paths: PathSettings = Field(
        default_factory=PathSettings,
        description="Path settings"
    )
    early_stopping: EarlyStoppingSettings = Field(
        default_factory=EarlyStoppingSettings,
        description="Early stopping settings"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "GRPOConfig":
        """Load configuration from YAML file with enhanced logging."""
        try:
            logger.info(f"Loading configuration from {config_path}")
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Log raw configuration
            logger.info("=== Raw Configuration ===")
            logger.info(f"Loaded data: {config_dict}")
            
            # Memory settings specific logging
            if 'memory' in config_dict:
                logger.info("=== Memory Configuration ===")
                logger.info(f"Memory settings: {config_dict['memory']}")
                logger.info(f"Gradient checkpointing type: {type(config_dict['memory'].get('use_gradient_checkpointing'))}")
                logger.info(f"Gradient checkpointing value: {config_dict['memory'].get('use_gradient_checkpointing')}")
            
            # Handle Unsloth-specific configuration
            if 'memory' in config_dict and 'use_gradient_checkpointing' in config_dict['memory']:
                if config_dict['memory']['use_gradient_checkpointing'] == "unsloth":
                    logger.info("Converting 'unsloth' to True for gradient checkpointing")
                    config_dict['memory']['use_gradient_checkpointing'] = True
            
            # Log field validation
            logger.info("=== Field Validation ===")
            for section, values in config_dict.items():
                if isinstance(values, dict):
                    logger.info(f"Section '{section}' fields: {list(values.keys())}")
                    
            try:
                config = cls(**config_dict)
                logger.info("Configuration validated successfully")
                return config
            except Exception as e:
                logger.error(f"Configuration validation failed: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    def save_to_yaml(self, config_path: Path) -> None:
        """Save configuration to YAML file."""
        try:
            # Convert to dict and handle Path objects
            config_dict = self.model_dump()
            config_dict["paths"] = {
                k: str(v) for k, v in config_dict["paths"].items()
                if v is not None
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f)
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {str(e)}")
            raise 