import ast
import inspect
import textwrap
import warnings
from typing import Callable

import typing


class Configs:
    batch_size: int
    x: ast.NodeTransformer


def dep():
    warnings.warn("inspect.getargspec() is deprecated since Python 3.0, "
                  "use inspect.signature() or inspect.getfullargspec()",
                  DeprecationWarning, stacklevel=2)

    return None


def model(configs: Configs):
    print(2)

    def temp():
        print('x')

    temp()
    z = max(2, configs.batch_size * 2)

    return z, configs, configs.x.visit()


class ConfigFunctionProcessor(ast.NodeVisitor):
    def __init__(self, func: Callable):
        if type(func) == type:
            func = typing.cast(object, func).__init__
            spec: inspect.Signature = inspect.signature(func)
            params = spec.parameters
            assert len(params) == 2
            param: inspect.Parameter = params[list(params.keys())[1]]
            source = textwrap.dedent(inspect.getsource(func))

        else:
            spec: inspect.Signature = inspect.signature(func)
            params = spec.parameters
            assert len(params) == 1
            param: inspect.Parameter = params[list(params.keys())[0]]
            source = inspect.getsource(func)

        assert (param.kind == param.POSITIONAL_ONLY or
                param.kind == param.POSITIONAL_OR_KEYWORD)

        self.arg_name = param.name

        self.required = set()
        self.is_referenced = False

        parsed = ast.parse(source)
        self.visit(parsed)

    def visit_Attribute(self, node: ast.Attribute):
        while not isinstance(node.value, ast.Name):
            node = node.value

        if node.value.id != self.arg_name:
            return

        self.required.add(node.attr)

    # Only visits if not captured before
    def visit_Name(self, node: ast.Name):
        if node.id == self.arg_name:
            self.is_referenced = True
            print(f"Referenced {node.id} in {node.lineno}:{node.col_offset}")


x = inspect.signature(model)
y = inspect.getfullargspec(model)
print(x)

c = ConfigFunctionProcessor(model)
print(c.required, c.is_referenced)


class Model:
    def __init__(self, configs: Configs):
        print(2)

        def temp():
            print('x')

        temp()
        z = max(2, configs.batch_size * 2)

        self.ret = (z, configs, configs.x.visit())

    def method(self):
        pass


class CallModel:
    def __init__(self):
        self.ret = 2

    def __call__(self, configs: Configs):
        print(2)

        def temp():
            print('x')

        temp()
        z = max(2, configs.batch_size * 2)

        return z, configs, configs.x.visit()


c = ConfigFunctionProcessor(Model)
print(c.required, c.is_referenced)

call_model = CallModel()
x = inspect.signature(call_model)
y = inspect.getfullargspec(call_model)
print(x)

# src = inspect.getsource(call_model)
# p = ast.parse(src)


x = inspect.signature(Model)
y = inspect.getfullargspec(Model)
print(x)

src = inspect.getsource(Model)
p = ast.parse(src)

print(p)