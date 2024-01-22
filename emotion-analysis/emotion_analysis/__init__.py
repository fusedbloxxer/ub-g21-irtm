import argparse
import logging
import os
import pathlib as pb
import random
import sys
from logging import Formatter, StreamHandler
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml

from .settings.config import EmotionAnalysisConfig

# Setup script logging
formatter = Formatter(fmt='[%(levelname)s]:%(message)s')
logger = logging.getLogger('emotion_analysis')
handler = StreamHandler()
handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

# Read project-wide settings from YAML file
config = EmotionAnalysisConfig.from_yaml(Path('..', 'config.yaml'))

# Setup JAX memory
match config.gpu_memory:
    case 'preallocate':
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "true"
        import jax; jax.numpy.zeros((1, 1))
    case 'on-demand':
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = "platform"
        import jax
    case _:
        raise ValueError('invalid gpu_memory option: {}'.format(config.gpu_memory))

# Change HuggingFace cache directory
os.environ['HF_HOME'] = str(config.cache_dir / 'huggingface')

# Setup MLFlow logging
mlflow.set_tracking_uri(uri=config.tracking_uri)

# Environment:
print('JAX Backend: ', jax.default_backend())
print('JAX Version: ', jax.__version__)
print('Python: ', sys.version)
print('System: ', os.uname())

# Set seed to reproduce results
random.seed(config.seed)
np.random.seed(config.seed)
gen_torch = torch.Generator().manual_seed(config.seed)
