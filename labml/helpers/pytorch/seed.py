import torch
import numpy as np

from labml.configs import BaseConfigs


class SeedConfigs(BaseConfigs):
    seed: int = 5

    set_seed = 'set_seed'


@SeedConfigs.calc(SeedConfigs.set_seed)
def set_seed(c: SeedConfigs):
    torch.manual_seed(c.seed)
    np.random.seed(c.seed)
