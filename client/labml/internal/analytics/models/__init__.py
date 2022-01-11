from typing import Callable, Dict, List, TYPE_CHECKING

from labml.internal.util.strings import is_pattern_match

if TYPE_CHECKING:
    from torch import nn


class ForwardHook:
    def __init__(self, name: str, save_callback: Callable):
        self.save_callback = save_callback
        self.name = name

    def __call__(self, module, i, o):
        self.save_callback(self.name, i, o)


class BackwardHook:
    def __init__(self, name: str, save_callback: Callable):
        self.save_callback = save_callback
        self.name = name

    def __call__(self, module, i, o):
        self.save_callback(self.name, i, o)


class ModelProbe:
    def __init__(self, model: 'nn.Module', name: str = 'model'):
        self.model = model
        for n, module in model.named_modules():
            module: 'nn.Module'
            if n == '':
                n = name
            forward_hook = ForwardHook(n, self.add_forward_tensor)
            module.register_forward_hook(forward_hook)

            backward_hook = ForwardHook(n, self.add_backward_tensor)
            module.register_full_backward_hook(backward_hook)

        self._forward_output = {}
        self._forward_input = {}

        self._backward_output = {}
        self._backward_input = {}

        self._parameters = {}
        for k, v in model.named_parameters():
            self._parameters[k] = v

    def add_forward_tensor(self, name: str, inp: any, outp: any):
        self._forward_input[name] = inp
        self._forward_output[name] = outp

    def add_backward_tensor(self, name: str, inp: any, outp: any):
        self._backward_input[name] = inp
        self._backward_output[name] = outp

    @property
    def parameters(self):
        return ValueCollection(self._parameters, sorted(list(self._parameters.keys())))

    @property
    def forward_input(self):
        return ValueCollection(self._forward_input, sorted(list(self._forward_input.keys())))

    @property
    def forward_output(self):
        return ValueCollection(self._forward_output, sorted(list(self._forward_output.keys())))

    @property
    def backward_input(self):
        return ValueCollection(self._backward_input, sorted(list(self._backward_input.keys())))

    @property
    def backward_output(self):
        return ValueCollection(self._backward_output, sorted(list(self._backward_output.keys())))


class ValueCollection:
    def __init__(self, values: Dict[str, any], keys: List[str]):
        self._keys = keys
        self._values = values

    @staticmethod
    def _expand_value(prefix: str, value: any, separator: str = '.'):
        if isinstance(value, tuple) or isinstance(value, list):
            return sum([
                ValueCollection._expand_value(f'{prefix}{separator}{i}', v) for i, v in enumerate(value)],
                [])
        if isinstance(value, dict):
            return sum([ValueCollection._expand_value(f'{prefix}{separator}{k}', v) for k, v in value.items()], [])

        return [prefix]

    def get_value(self, key: str):
        return self._values[key]

    def get_list(self):
        return [self.get_value(f) for f in self._keys]

    def get_dict(self):
        return {f: self.get_value(f) for f in self._keys}

    def deep(self):
        if isinstance(self, DeepValueCollection):
            return self

        keys = sum([
            ValueCollection._expand_value(f'{k}', self._values[k], '/') for k in self._keys],
            [])

        return DeepValueCollection(self._values, sorted(keys))

    def __str__(self):
        return str(self._keys)

    def __repr__(self):
        return repr(self._keys)

    def __getitem__(self, item: str):
        keys = [k for k in self._keys if is_pattern_match(k, item)]

        return self.__class__(self._values, keys)

    def __len__(self):
        return len(self._keys)

    def keys(self):
        return self._keys


class DeepValueCollection(ValueCollection):
    def get_value(self, key: str):
        parts = key.split('/')

        if len(parts) == 1:
            return self._values[parts[0]]

        assert len(parts) == 2
        value = self._values[parts[0]]
        key = parts[1]

        parts = key.split('.')
        for p in parts:
            if isinstance(value, tuple) or isinstance(value, list):
                value = value[int(p)]
            else:
                value = value[p]

        return value
