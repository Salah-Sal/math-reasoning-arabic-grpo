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
        description="Whether to use gradient checkpointing or Unsloth's implementation"
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

    model_config = ConfigDict(
        extra='allow',  # Allow extra fields
        validate_assignment=True
    )

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
        description="Whether to use flash attention"
    )
    # Add LoRA configuration fields
    lora_rank: int = Field(
        default=16,
        description="Rank of LoRA matrices"
    )
    lora_alpha: int = Field(
        default=16,
        description="Alpha parameter for LoRA"
    )
    target_modules: List[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj"],
        description="Target modules for LoRA"
    )
    lora_dropout: float = Field(
        default=0.05,
        description="Dropout probability for LoRA layers"
    )

    model_config = ConfigDict(
        extra='allow',  # Allow extra fields
        validate_assignment=True,
        populate_by_name=True
    )

    @validator('use_gradient_checkpointing', pre=True)
    def validate_gradient_checkpointing(cls, v):
        """Validate gradient checkpointing setting."""
        if isinstance(v, str) and v.lower() == 'unsloth':
            return 'unsloth'
        return bool(v)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelSettings":
        """Create instance from dictionary with enhanced logging."""
        logger.info("=== Creating ModelSettings from dict ===")
        logger.info(f"Input data: {data}")
        
        # Log validation steps
        logger.info("=== Validation Steps ===")
        logger.info(f"Checking required fields: {cls.__fields__.keys()}")
        logger.info(f"Extra fields in input: {set(data.keys()) - set(cls.__fields__.keys())}")
        
        try:
            instance = cls(**data)
            logger.info("=== Validated Settings ===")
            logger.info(f"Model name: {instance.name}")
            logger.info(f"Flash attention: {instance.use_flash_attention}")
            logger.info(f"LoRA config: rank={instance.lora_rank}, alpha={instance.lora_alpha}")
            return instance
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            raise

    def model_dump(self) -> Dict[str, Any]:
        """Enhanced model dump with logging."""
        data = super().model_dump()
        logger.info("=== Model Settings Dump ===")
        for key, value in data.items():
            logger.info(f"  {key}: {value}")
        return data

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

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow'
    )

    @classmethod
    def from_yaml(cls, config_path: Path) -> "GRPOConfig":
        """Load configuration from YAML file with enhanced logging and validation."""
        try:
            logger.info(f"Loading configuration from {config_path}")
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Log raw configuration
            logger.info("=== Raw Configuration ===")
            logger.info(f"Loaded sections: {list(config_dict.keys())}")
            
            # Validate and convert paths
            if 'paths' in config_dict:
                logger.info("Converting path strings to Path objects")
                for key, value in config_dict['paths'].items():
                    if value:
                        config_dict['paths'][key] = Path(value)
            
            # Validate memory settings
            if 'memory' in config_dict:
                logger.info("Validating memory settings")
                memory_config = config_dict['memory']
                logger.info(f"Memory configuration: {memory_config}")
                
                # Handle gradient checkpointing
                if 'use_gradient_checkpointing' in memory_config:
                    value = memory_config['use_gradient_checkpointing']
                    logger.info(f"Gradient checkpointing value: {value}")
            
            # Create config instance
            config = cls(**config_dict)
            
            # Log final configuration
            logger.info("=== Validated Configuration ===")
            logger.info(f"Model settings: {config.model.model_dump()}")
            logger.info(f"Memory settings: {config.memory.model_dump()}")
            
            return config
        except Exception as e:
            logger.error(f"Configuration loading failed: {str(e)}", exc_info=True)
            raise 