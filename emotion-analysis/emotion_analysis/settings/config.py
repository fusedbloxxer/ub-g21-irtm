from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from yaml import safe_load

from ..data.types import SubTask
from ..model.finetune import FineTune


@dataclass
class EmotionAnalysisConfig:
    # Task
    subtask: SubTask

    # Training
    seed: int
    batch_size: int
    num_workers: int
    prefetch_factor: int

    # Model settings
    learning_rate: float
    finetune: FineTune

    # Pretrained Model
    model_repo: str

    # Tokenization
    max_conv_len: int
    max_uttr_len: int

    # Paths
    log_dir: Path
    data_dir: Path
    ckpt_dir: Path
    cache_dir: Path
    root_dir : Path

    # JAX settings
    gpu_memory: Literal['preallocate', 'on-demand']

    # Logging
    tracking_uri: str

    @classmethod
    def from_yaml(cls, filepath: Path) -> 'EmotionAnalysisConfig':
        # Read setting from file on disk
        with open(filepath, 'r') as yaml_config:
            yaml_config = safe_load(yaml_config)

        # Parse and extract settings from file
        config = {}
        config['seed'] = yaml_config['seed']
        config['batch_size'] = yaml_config['batch_size']
        config['num_workers'] = yaml_config['num_workers']
        config['prefetch_factor'] = yaml_config['prefetch_factor']
        config['root_dir'] = Path(yaml_config['rootdir'])
        config['log_dir'] = config['root_dir'] / 'log'
        config['data_dir'] = config['root_dir'] /'data'
        config['ckpt_dir'] = config['root_dir'] / 'ckpt'
        config['cache_dir'] = config['root_dir'] / 'cache'
        config['gpu_memory'] = yaml_config['gpu_memory']
        config['tracking_uri'] = f"file://{config['log_dir'].absolute()}/mlflow"
        config['subtask'] = yaml_config['subtask']
        config['max_conv_len'] = yaml_config['max_conv_len']
        config['max_uttr_len'] = yaml_config['max_uttr_len']
        config['model_repo'] = yaml_config['model_repo']
        config['learning_rate'] = yaml_config['learning_rate']
        config['finetune'] = yaml_config['finetune']

        # Wrap settings
        return cls(**config)
