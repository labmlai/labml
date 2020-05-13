import ast
import inspect
import textwrap
from typing import Callable, cast


class DependencyParser(ast.NodeVisitor):
    def __init__(self, func: Callable):
        if type(func) == type:
            func = cast(object, func).__init__
            spec: inspect.Signature = inspect.signature(func)
            params = spec.parameters
            assert len(params) == 2
            param: inspect.Parameter = params[list(params.keys())[1]]
            source = textwrap.dedent(inspect.getsource(func))

        else:
            spec: inspect.Signature = inspect.signature(func)
            params = spec.parameters
            assert len(params) >= 1
            param: inspect.Parameter = params[list(params.keys())[0]]
            source = inspect.getsource(func)

        assert (param.kind == param.POSITIONAL_ONLY or
                param.kind == param.POSITIONAL_OR_KEYWORD)

        self.arg_name = param.name

        self.required = set()
        self.is_referenced = False

        source = textwrap.dedent(source)
        parsed = ast.parse(source)
        self.visit(parsed)

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.value, ast.Name):
            if node.value.id == self.arg_name:
                self.required.add(node.attr)
        else:
            for child in ast.iter_child_nodes(node):
                self.visit(child)

    def visit_Name(self, node: ast.Name):
        if node.id == self.arg_name:
            self.is_referenced = True
            print(f"Referenced {node.id} in {node.lineno}:{node.col_offset}")