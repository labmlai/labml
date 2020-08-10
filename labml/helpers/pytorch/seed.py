import numpy as np
import torch

from labml.configs import BaseConfigs, option


class SeedConfigs(BaseConfigs):
    seed: int = 5

    set_seed = 'set_seed'

    def __init__(self):
        super().__init__(_primary='set_seed')


@option(SeedConfigs.set_seed)
def set_seed(c: SeedConfigs):
    torch.manual_seed(c.seed)
    np.random.seed(c.seed)
    return c.seed
