import inspect
import tokenize
from io import BytesIO
from typing import Dict, NamedTuple

import torch


class GuardError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class GuardCallError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class Definition(NamedTuple):
    file: str
    line: int


class DefinedValue(NamedTuple):
    name: str
    value: int
    file: str
    line: int


def frame_info_to_definition(frame_info: inspect.FrameInfo):
    return Definition(frame_info.filename, frame_info.lineno)


class GuardSameSize:
    OPERATORS = {'+', '-', '*', '/', '%', '(', ')', '//'}
    IGNORE_TOKENS = {tokenize.ENCODING, tokenize.NEWLINE, tokenize.ENDMARKER}

    def __init__(self, caller_depth):
        self.caller_depth = caller_depth
        self._parse_cache: Dict[str, list[str]] = {}
        self._values: Dict[str, int] = {}
        self._definitions: Dict[str, Definition] = {}

    def parse(self, string):
        if string in self._parse_cache:
            return self._parse_cache[string]

        tokens = tokenize.tokenize(BytesIO(string.encode('utf-8')).readline)
        parsed = []
        for t in tokens:
            if t.type == tokenize.NAME:
                parsed.append(t.string)
            elif t.type == tokenize.OP:
                if t.string not in self.OPERATORS:
                    raise GuardCallError(f"Could not parse query: {string} {t}")
                parsed.append(t.string)
            elif t.type in self.IGNORE_TOKENS:
                continue
            else:
                raise GuardCallError(f"Could not parse query: {string} {t}")

        self._parse_cache[string] = parsed

        return parsed

    def get_single_value(self, identifier, caller: Definition):
        # TODO: use caller for name spaces
        if identifier[0] == '_':
            return None

        if identifier in self._values:
            return self._values[identifier]
        else:
            return None

    def get_value(self, parsed, caller):
        if len(parsed) == 1:
            return self.get_single_value(parsed[0], caller)

        expr = []
        for p in parsed:
            if p in self.OPERATORS:
                expr.append(p)
            else:
                v = self.get_single_value(p, caller)
                if v is None:
                    string = "".join(parsed)
                    raise GuardError(f"Expression {string} with undefined identifier {p}")
                expr.append(v)

        expr = [str(e) for e in expr]
        return eval("".join(expr), None, None)

    def set_value(self, parsed, value, caller):
        assert len(parsed) == 1
        identifier = parsed[0]
        if identifier[0] == '_':
            return

        self._values[identifier] = value
        self._definitions[identifier] = caller

    def get_definition(self, parsed, caller):
        definitions = []

        identifiers = set()
        for p in parsed:
            if p not in self.OPERATORS:
                identifiers.add(p)

        for p in identifiers:
            definitions.append(DefinedValue(p, self.get_value([p], caller),
                                            self._definitions[p].file, self._definitions[p].line))

        return definitions

    def guard_single(self, value: int, string: str, caller):
        if type(value) != int:
            raise GuardError(f"Sizes should be ints: {value}")

        parsed = self.parse(string)
        previous_value = self.get_value(parsed, caller)
        if previous_value is None:
            self.set_value(parsed, value, caller)
        elif previous_value != value:
            raise GuardError(f"{string} mismatch: {value} != {previous_value} defined at "
                             f"{self.get_definition(parsed, caller)}")

    def guard(self, *args):
        if len(args) < 2:
            raise GuardCallError("Guard needs at least two arguments")

        strings = []
        for i in reversed(range(len(args))):
            if type(args[i]) == str:
                strings.append(args[i])
            else:
                break

        strings = list(reversed(strings))
        values = args[:len(args) - len(strings)]
        value = values[0]

        for v in values:
            if type(v) != type(value):
                raise GuardError("All shapes to be guarded must be equal")
            if v != value:
                raise GuardError("All shapes to be guarded must be equal")

        value_type = type(value)
        if value_type == list or value_type == tuple or value_type is torch.Size:
            if len(value) != len(strings):
                raise GuardError("Number of dimensions do not match")
            values = list(value)
        else:
            values = [value]

        stack = inspect.stack()
        caller = frame_info_to_definition(stack[1 + self.caller_depth])
        del stack

        for v, s in zip(values, strings):
            self.guard_single(v, s, caller)

        if len(values) == 1:
            return values[0]
        else:
            return [v for v in values]


_guard = GuardSameSize(1)


def guard(*args):
    return _guard.guard(*args)


def test():
    test_guard = GuardSameSize(0)

    test_guard.guard(5, 'batch_size')
    test_guard.guard((2, 3), 'x', 'y')
    test_guard.guard((5, 2), 'batch_size', 'x')
    test_guard.guard((7, 2), 'batch_size+x', 'x')
    test_guard.guard((6, 2), '(batch_size*x)-y', 'x')


if __name__ == '__main__':
    test()
