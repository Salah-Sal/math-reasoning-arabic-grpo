from pydantic import BaseModel, Field
from typing import Optional, List
from pathlib import Path


class ModelConfig(BaseModel):
    """Configuration for the model settings"""
    model_name: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        description="Name of the base model to use"
    )
    max_seq_length: int = Field(default=384, description="Maximum sequence length")
    load_in_4bit: bool = Field(default=True, description="Whether to load in 4-bit quantization")
    fast_inference: bool = Field(default=False, description="Enable fast inference mode")
    max_lora_rank: int = Field(default=16, description="Maximum LoRA rank")
    gpu_memory_utilization: float = Field(
        default=0.7,
        description="GPU memory utilization target",
        ge=0.0,
        le=1.0
    )


class TrainingConfig(BaseModel):
    """Configuration for training parameters"""
    learning_rate: float = Field(default=5e-6, description="Learning rate for training")
    per_device_train_batch_size: int = Field(default=8, description="Batch size per device")
    gradient_accumulation_steps: int = Field(default=8, description="Number of gradient accumulation steps")
    max_prompt_length: int = Field(default=256, description="Maximum prompt length")
    max_completion_length: int = Field(default=128, description="Maximum completion length")
    logging_steps: int = Field(default=10, description="Number of steps between logging")
    save_steps: int = Field(default=500, description="Number of steps between model saves")


class DataConfig(BaseModel):
    """Configuration for data handling"""
    data_dir: Path = Field(
        default=Path("/home/Sal3/ml_data/translations"),
        description="Directory containing the dataset"
    )
    output_dir: Path = Field(
        default=Path("./outputs"),
        description="Directory for saving outputs"
    )
    cache_dir: Optional[Path] = Field(
        default=None,
        description="Directory for caching processed data"
    )


class ProjectConfig(BaseModel):
    """Main configuration class combining all sub-configurations"""
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    seed: int = Field(default=3407, description="Random seed for reproducibility")
    device: str = Field(default="cuda", description="Device to use for training")

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def load_from_yaml(cls, config_path: Path) -> "ProjectConfig":
        """Load configuration from a YAML file"""
        try:
            import yaml
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
        except Exception as e:
            raise ValueError(f"Error loading configuration from {config_path}: {str(e)}")

    def save_to_yaml(self, config_path: Path) -> None:
        """Save configuration to a YAML file"""
        try:
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(self.dict(), f)
        except Exception as e:
            raise ValueError(f"Error saving configuration to {config_path}: {str(e)}") 