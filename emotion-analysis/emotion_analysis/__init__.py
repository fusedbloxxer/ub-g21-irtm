import os
import sys
import argparse
import logging
from logging import StreamHandler, Formatter
import pathlib as pb
import mlflow

# Setup script logging
formatter = Formatter(fmt='[%(levelname)s]:%(message)s')
logger = logging.getLogger('emotion_analysis')
handler = StreamHandler()
handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

# Allocate memory on-demand to avoid OOM and debug easily
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = "platform"

# Allow JAX to allocate as much memory as possible for faster processing
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "true"

# Setup environment
SEED = 42
ROOT_DIR = pb.Path('..')
LOG_DIR = ROOT_DIR / 'log'
CKPT_DIR = ROOT_DIR / 'ckpt'
DATA_DIR = ROOT_DIR / 'data'
CACHE_DIR = ROOT_DIR / 'cache'

# Change HuggingFace cache directory
os.environ['HF_HOME'] = str(CACHE_DIR / 'huggingface')

# Setup MLFlow logging
mlflow.set_tracking_uri(uri=f'file://{(LOG_DIR / "mlflow").absolute()}')

# DeepLearning Backend
import jax

# Environment:
print('JAX Backend: ', jax.default_backend())
print('JAX Version: ', jax.__version__)
print('Python: ', sys.version)
print('System: ', os.uname())

# Force JAX to use GPU memory
jax.numpy.zeros((1, 1))
