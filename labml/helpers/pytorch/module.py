from typing import Any

import torch.nn


class Module(torch.nn.Module):
    r"""
    Wraps ``torch.nn.Module`` to overload ``__call__`` instead of
    ``forward`` for better type checking.
    """

    def _forward_unimplemented(self, *input: Any) -> None:
        # To stop PyTorch from giving abstract methods warning
        pass

    def __init_subclass__(cls, **kwargs):
        if cls.__dict__.get('__call__', None) is None:
            return

        setattr(cls, 'forward', cls.__dict__['__call__'])
        delattr(cls, '__call__')

    @property
    def device(self):
        params = self.parameters()
        try:
            sample_param = next(params)
            return sample_param.device
        except StopIteration:
            raise RuntimeError(f"Unable to determine"
                               f" device of {self.__class__.__name__}") from None


if __name__ == '__main__':
    m = Module()
    print(m.device)
