from dataclasses import dataclass
from typing import Optional
from hydra.core.config_store import ConfigStore

@dataclass
class WandbConfig:
    """Wandb-specific configuration"""

    enabled: bool = True
    entity: str = "???"  # fill your wandb entity
    project: str = "???"

@dataclass
class DataConfig:
    """Data loading configuration"""

    batch_size: int = 1
    num_samples: int = 100
    cle_data_path: str = ""
    tgt_data_path: str = ""
    output: str = ""


@dataclass
class OptimConfig:
    """Optimization parameters"""

    alpha: float = 1.0
    epsilon: int = 8
    steps: int = 300


@dataclass
class ModelConfig:
    """Model-specific parameters"""

    input_res: int = 336
    use_source_crop: bool = True
    use_target_crop: bool = True
    ensemble: bool = True
    device: str = "cuda:0"  # Can be "cpu", "cuda:0", "cuda:1", etc.
    backbone: list = (
        "L336",
        "B16",
        "B32",
        "Laion",
    )  # List of models to use: L336, B16, B32, Laion
    window_ratio: float = 0.75
    area_scale_range: Optional[tuple] = (0.5, 0.9)
    stride: int = 1
    interval: int = 1

@dataclass
class MainConfig:
    """Main configuration combining all sub-configs"""

    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig()
    model: ModelConfig = ModelConfig()
    wandb: WandbConfig = WandbConfig()

# register config for different setting
@dataclass
class Ensemble3ModelsConfig(MainConfig):
    """Configuration for anyattack.py"""

    data: DataConfig = DataConfig(batch_size=1)
    model: ModelConfig = ModelConfig(
        use_source_crop=True, use_target_crop=True, backbone=["B16", "B32", "Laion"]
    )

# Register configs with Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=MainConfig)
cs.store(name="OGJA", node=Ensemble3ModelsConfig)
