from pathlib import Path
from typing import Optional, Dict, Any, Literal
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

class MemorySettings(BaseModel):
    """Memory management settings."""
    gpu_memory_utilization: float = Field(
        default=0.7,
        gt=0.0,
        le=1.0,
        description="Target GPU memory utilization"
    )
    use_gradient_checkpointing: bool = Field(
        default=True,
        description="Whether to use gradient checkpointing"
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
    output_dir: Path = Field(
        default=Path("outputs/grpo_training"),
        description="Directory for training outputs"
    )
    checkpoint_dir: Path = Field(
        default=Path("checkpoints"),
        description="Directory for model checkpoints"
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
            v.mkdir(parents=True, exist_ok=True)
        return v

class GRPOConfig(BaseModel):
    """Main GRPO training configuration."""
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "GRPOConfig":
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return cls(**config_dict)
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
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