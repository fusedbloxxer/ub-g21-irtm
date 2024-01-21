from functools import partial
from dataclasses import dataclass
from torch.utils.data import DataLoader
from .. import config, gen_torch


DefaultDataLoader = partial(
    DataLoader,
    generator=gen_torch,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    prefetch_factor=config.prefetch_factor,
)
