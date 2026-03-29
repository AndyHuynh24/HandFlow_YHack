"""Pydantic configuration models loaded from config/config.yaml."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

class MediaPipeConfig(BaseModel): 
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.2
    max_num_hands: int = 2
    model_complexity: int = 0  # 0=Lite (Fastest), 1=Full (Default is 1)

class ValidationConfig(BaseModel):
    check_nan: bool = True
    check_inf: bool = True
    check_outliers: bool = True
    check_sequence_length: bool = True

    # Thresholds
    outlier_std_threshold: float = 4.0    

class FeatureConfig:
    pass

class DataConfig(BaseModel): 
    right_raw_path: str 
    left_raw_path: str
    processed_path: str 
    keypoint_dim: int 
    sequence_length: int 
    validator: ValidationConfig = ValidationConfig()

class ModelConfig(BaseModel): 
    output_dir: str
    logs_dir: str
    architecture: str
    gestures: list[str]
    input_dim: int
    hidden_units: int
    num_layers: int
    dropout: float
    num_classes: int
    model_path: str

class ReduceLearningRateConfig(BaseModel):
    enabled: bool
    factor: float
    patience: int
    min_lr: float

class TrainingConfig(BaseModel): 
    batch_size: int
    epochs: int
    learning_rate: float
    
    reduce_lr: ReduceLearningRateConfig
    early_stopping_patience: int
    validation_split: float
    optimizer: str
    loss: str

class WandBConfig(BaseModel):
    """Weights & Biases configuration."""
    enabled: bool = False
    project: str = "ml-experiments"
    entity: str | None = None
    log_model: bool = True


class TensorBoardConfig(BaseModel):
    enabled: bool = True
    log_dir: str = "logs/tensorboard"
    histogram_freq: int = 1


class TrackingConfig(BaseModel):
    """Experiment tracking configuration."""
    enabled: bool = True
    wandb: WandBConfig = WandBConfig()
    tensorboard: TensorBoardConfig = TensorBoardConfig()


class AugmentationConfig(BaseModel):
    """Configuration for realistic hand gesture augmentation."""

    # Master switch
    enabled: bool = True

    # Number of augmented variants per sample
    n_variants: int = 6

    max_frame_jitter: float = 0.015        # caps per-frame noise
    motion_noise_scale: float = 0.5        # reduce noise when hand is static
    min_timewarp_frames: int = 6         # disables time warp for very short seq
    landmark_dropout_mode: str = "whole" # "whole" | "fingertips"
    max_depth_delta: float = 0.08          # clamp z distortion

    # -------------------------------------------------
    # Temporal & sensor-level augmentations 
    # -------------------------------------------------

    # Gaussian noise (sensor jitter)
    noise_prob: float = 0.6
    noise_std: float = 0.015  # reduced to avoid landmark drift

    # Time warping (speed variation, length preserved)
    time_warp_prob: float = 0.4
    time_warp_factor: float = 0.15  

    # Frame dropout (occlusion / missed detection)
    dropout_prob: float = 0.25
    dropout_rate: float = 0.1  

    # -------------------------------------------------
    # Global geometric transforms 
    # -------------------------------------------------

    # Uniform scaling (camera distance variation)
    scale_prob: float = 0.35
    scale_range: tuple = (0.9, 1.1)

    # In-plane rotation (camera tilt / wrist rotation)
    rotation_prob: float = 0.25
    rotation_range: tuple = (8,8)  # degrees

    # -------------------------------------------------
    # Z-axis depth variation (MediaPipe-realistic)
    # -------------------------------------------------

    # Global depth scaling (hand closer / farther)
    z_scale_prob: float = 0.4
    z_scale_range: tuple = (0.9, 1.1)

    # Small camera depth shift
    z_shift_prob: float = 0.4
    z_shift_range: tuple = (-0.05, 0.05)

    z_proportional_prob: float = 0.25
    z_proportional_range: tuple = (0.85, 1.15)
    
    z_finger_length_prob: float = 0.25
    z_finger_length_range: tuple = (0.9, 1.1)

    # Z-axis sensor noise
    z_noise_prob: float = 0.5
    z_noise_std: float = 0.01

    hand_tilt_prob: float = 0.2
    hand_tilt_range: tuple = (-12, 12)  # degrees

class Config(BaseModel):
    mediapipe: MediaPipeConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    augmentation: AugmentationConfig
    tracking: TrackingConfig = TrackingConfig()

def load_config(config_path: str | Path) -> Config:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated Config object.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    return Config(**raw_config)


def save_config(config: Config, output_path: str | Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration object to save.
        output_path: Path to save the YAML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)


