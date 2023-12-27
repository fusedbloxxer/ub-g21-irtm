import jax
import flax
import jaxlib
import pandas as pd
import pathlib as pb


# Setup environment
ROOT_DIR = pb.Path('..')
DATA_DIR = ROOT_DIR / 'data'


print('JAX Backend: ', jax.default_backend())
