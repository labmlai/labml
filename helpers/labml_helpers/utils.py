from typing import List, Tuple, Union

import torch


def detach(s: Union[Tuple, List, torch.Tensor]):
    if s is None:
        return None
    elif isinstance(s, torch.Tensor):
        return s.detach()
    elif isinstance(s, tuple):
        return tuple(detach(e) for e in s)
    else:
        return list(detach(e) for e in s)
