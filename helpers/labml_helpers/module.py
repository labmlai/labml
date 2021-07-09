from typing import Any, TypeVar, Iterator, Iterable, Generic

import torch.nn


class Module(torch.nn.Module):
    r"""
    Wraps ``torch.nn.Module`` to overload ``__call__`` instead of
    ``forward`` for better type checking.
    
    `PyTorch Github issue for clarification <https://github.com/pytorch/pytorch/issues/44605>`_
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


M = TypeVar('M', bound=torch.nn.Module)
T = TypeVar('T')


class TypedModuleList(torch.nn.ModuleList, Generic[M]):
    def __getitem__(self, idx: int) -> M:
        return super().__getitem__(idx)

    def __setitem__(self, idx: int, module: M) -> None:
        return super().__setitem__(idx, module)

    def __iter__(self) -> Iterator[M]:
        return super().__iter__()

    def __iadd__(self: T, modules: Iterable[M]) -> T:
        return super().__iadd__(modules)

    def insert(self, index: int, module: M) -> None:
        super().insert(index, module)

    def append(self: T, module: M) -> T:
        return super().append(module)

    def extend(self: T, modules: Iterable[M]) -> T:
        return super().extend(modules)

    def forward(self):
        raise NotImplementedError()


if __name__ == '__main__':
    m = Module()
    print(m.device)
