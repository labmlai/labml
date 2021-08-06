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
    r"""
    This is a configurable module for setting the seeds.
    It will set seeds with ``torch.manual_seed`` and ``np.random.seed``.

    You need to call ``set`` method to set seeds
    (`example <https://github.com/labmlai/labml/blob/master/samples/pytorch/mnist/e_labml_helpers.py>`_).

    Arguments:
        seed (int): Seed integer. Defaults to ``5``.
    """
    seed: int = 5

    set = '_set_seed'


@option(SeedConfigs.set)
def _set_seed(c: SeedConfigs):
    return SetSeed(c.seed)
