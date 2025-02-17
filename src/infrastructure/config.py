from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from pathlib import Path


class ModelConfig(BaseModel):
    """Configuration for the model settings"""
    model_name: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        description="Name of the base model to use"
    )
    max_seq_length: int = Field(default=256, description="Maximum sequence length")
    load_in_4bit: bool = Field(default=True, description="Whether to load in 4-bit quantization")
    fast_inference: bool = Field(default=True, description="Enable fast inference mode")
    max_lora_rank: int = Field(default=8, description="Maximum LoRA rank")
    gpu_memory_utilization: float = Field(
        default=0.6,
        description="GPU memory utilization target",
        ge=0.0,
        le=1.0
    )
    
    model_config = ConfigDict(frozen=False)  # Allow modifications


class TrainingConfig(BaseModel):
    """Configuration for training parameters"""
    learning_rate: float = Field(default=5e-6, description="Learning rate for training")
    per_device_train_batch_size: int = Field(default=4, description="Batch size per device")
    gradient_accumulation_steps: int = Field(default=4, description="Number of gradient accumulation steps")
    max_prompt_length: int = Field(default=256, description="Maximum prompt length")
    max_completion_length: int = Field(default=128, description="Maximum completion length")
    logging_steps: int = Field(default=10, description="Number of steps between logging")
    save_steps: int = Field(default=500, description="Number of steps between model saves")
    
    model_config = ConfigDict(frozen=True)


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
    
    model_config = ConfigDict(frozen=True)


class ProjectConfig(BaseModel):
    """Main configuration class combining all sub-configurations"""
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    seed: int = Field(default=3407, description="Random seed for reproducibility")
    device: str = Field(default="cuda", description="Device to use for training")

    model_config = ConfigDict(frozen=True)

    @classmethod
    def load_from_yaml(cls, config_path: Path) -> "ProjectConfig":
        """Load configuration from a YAML file"""
        try:
            import yaml
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                
            # Convert string paths to Path objects
            if 'data' in config_dict:
                data_config = config_dict['data']
                if 'data_dir' in data_config:
                    data_config['data_dir'] = Path(data_config['data_dir'])
                if 'output_dir' in data_config:
                    data_config['output_dir'] = Path(data_config['output_dir'])
                if 'cache_dir' in data_config and data_config['cache_dir']:
                    data_config['cache_dir'] = Path(data_config['cache_dir'])
                    
            return cls(**config_dict)
        except Exception as e:
            raise ValueError(f"Error loading configuration from {config_path}: {str(e)}")

    def save_to_yaml(self, config_path: Path) -> None:
        """Save configuration to a YAML file"""
        try:
            import yaml
            
            # Convert to dict and handle Path objects
            config_dict = self.model_dump()
            if 'data' in config_dict:
                data_config = config_dict['data']
                if 'data_dir' in data_config:
                    data_config['data_dir'] = str(data_config['data_dir'])
                if 'output_dir' in data_config:
                    data_config['output_dir'] = str(data_config['output_dir'])
                if 'cache_dir' in data_config and data_config['cache_dir']:
                    data_config['cache_dir'] = str(data_config['cache_dir'])
            
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f)
        except Exception as e:
            raise ValueError(f"Error saving configuration to {config_path}: {str(e)}") 