import numpy as np
import torch

from labml.configs import BaseConfigs, option


class SetSeed:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)


class SeedConfigs(BaseConfigs):
    seed: int = 5

    set = '_set_seed'


@option(SeedConfigs.set)
def _set_seed(c: SeedConfigs):
    return SetSeed(c.seed)
