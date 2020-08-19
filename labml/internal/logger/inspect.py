from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from ...logger import Text, StyleCode

if TYPE_CHECKING:
    from . import Logger

try:
    import torch
except ImportError:
    torch = None

try:
    import numpy
except ImportError:
    numpy = None


def _format_int(value: int):
    return f"{value:,}"


def _format_float(value: float):
    if abs(value) < 1e-9:
        lg = 0
    else:
        lg = int(np.ceil(np.log10(abs(value)))) + 1

    decimals = 7 - lg
    decimals = max(1, decimals)
    decimals = min(6, decimals)

    fmt = "{v:8,." + str(decimals) + "f}"
    return fmt.format(v=value)


def _format_value(value: any):
    if isinstance(value, int):
        return _format_int(value)
    elif (isinstance(value, np.int) or
          isinstance(value, np.long) or
          isinstance(value, np.uint64)):
        return _format_int(int(value))
    elif isinstance(value, np.float):
        if np.isnan(value):
            return 'NaN'
        return _format_float(float(value))
    elif torch is not None and isinstance(value, torch.Tensor):
        if not value.shape:
            return _format_value(value.item())
        else:
            return None
    else:
        return None


def _format_tensor(s: List[str], limit: int = 1_000, style: StyleCode = Text.value):
    res = []
    length = 0
    for p in s:
        if p in [',', ']', '[', '...']:
            res.append((p, Text.subtle))
        else:
            res.append((p, style))
            length += len(p)

        if length > limit:
            res.append((' ... ', Text.warning))
            break

    return res


def _key_value_pair(key: any, value: any, style: StyleCode = Text.meta):
    f = _format_value(value)
    if f is None:
        f = str(value)
    return [(f'{str(key)}: ', Text.subtle),
            (f, style)]


def _render_tensor(tensor, *, new_line: str = '\n', indent: str = ''):
    if len(tensor) > 5:
        idx = [0, 1, 2, '...', len(tensor) - 1]
    else:
        idx = [i for i in range(len(tensor))]

    res = [indent, '[']
    if new_line == '\n':
        next_indent = ' ' + indent
    else:
        next_indent = indent
    if len(tensor.shape) > 1:
        res.append(new_line)
        for i in idx:
            if i == '...':
                res.append(next_indent)
                res.append('...')
            else:
                res += _render_tensor(tensor[i],
                                      new_line=new_line,
                                      indent=next_indent)
            if i != idx[-1]:
                res.append(', ')
            res.append(new_line)
    else:
        for i in idx:
            if i == '...':
                res.append('...')
            else:
                res += _format_value(tensor[i])
            if i != idx[-1]:
                res.append(', ')

    res.append(indent)
    res.append(']')

    return res


def _get_value_full(value: any):
    if isinstance(value, str):
        if len(value) < 500:
            return [('"', Text.subtle),
                    (value, Text.value),
                    ('" len(', Text.subtle),
                    (_format_int(len(value)), Text.meta),
                    (')', Text.subtle)]
        else:
            return [('"', Text.subtle),
                    (value[:500], Text.value),
                    (' ..." len(', Text.subtle),
                    (_format_int(len(value)), Text.meta),
                    (')', Text.subtle)]

    elif numpy is not None and isinstance(value, numpy.ndarray):
        return [*_key_value_pair('dtype', value.dtype),
                '\n',
                *_key_value_pair('shape', [s for s in value.shape], Text.value),
                '\n',
                *_key_value_pair('min', numpy.min(value)),
                ' ',
                *_key_value_pair('max', numpy.max(value)),
                ' ',
                *_key_value_pair('mean', numpy.mean(value)),
                ' ',
                *_key_value_pair('std', numpy.std(value)),
                '\n',
                *_format_tensor(_render_tensor(value, new_line='\n'))]
    elif torch is not None and isinstance(value, torch.Tensor):
        return [*_key_value_pair('dtype', value.dtype),
                '\n',
                *_key_value_pair('shape', [s for s in value.shape], Text.value),
                '\n',
                *_key_value_pair('min', torch.min(value).item()),
                ' ',
                *_key_value_pair('max', torch.max(value).item()),
                ' ',
                *_key_value_pair('mean', torch.mean(value.to(torch.float)).item()),
                ' ',
                *_key_value_pair('std', torch.std(value.to(torch.float)).item()),
                '\n',
                *_format_tensor(_render_tensor(value, new_line='\n'))]

    s = str(value)
    s = s.replace('\r', '')
    return [s]


def _shrink(s: str, style: StyleCode = Text.value, limit: int = 80):
    s = s.replace('\r', '')
    lines = s.split('\n')

    res = []
    length = 0
    for line in lines:
        if len(res) > 0:
            res.append(('\\n', Text.subtle))
        if len(line) + length < limit:
            res.append((line, style))
            length += len(line)
        else:
            res.append((line[:limit - length], style))
            res.append((' ...', Text.warning))
            break

    return res


def _get_value_line(value: any):
    f = _format_value(value)

    if f is not None:
        return [(f, Text.value)]
    elif isinstance(value, str):
        return [('"', Text.subtle)] + _shrink(value) + [('"', Text.subtle)]
    elif numpy is not None and isinstance(value, numpy.ndarray):
        return [*_format_tensor(_render_tensor(value, new_line=''), limit=80)]
    elif torch is not None and isinstance(value, torch.Tensor):
        return [*_format_tensor(_render_tensor(value, new_line=''), limit=80)]

    s = str(value)
    return _shrink(s)


class Inspect:
    def __init__(self, logger: 'Logger'):
        self.__logger = logger

    def _log_key_value(self, items: List[Tuple[any, any]], is_show_count=True):
        max_key_len = 0
        for k, v in items:
            max_key_len = max(max_key_len, len(str(k)))

        count = len(items)
        if count > 12:
            items = items[:10]
        for k, v in items:
            spaces = " " * (max_key_len - len(str(k)))
            self.__logger.log([(f"{spaces}{k}: ", Text.key)] +
                              _get_value_line(v))
        if count > 12:
            self.__logger.log([("...", Text.meta)])

        if is_show_count:
            self.__logger.log([
                "Total ",
                (str(count), Text.meta),
                " item(s)"])

    def info(self, *args, **kwargs):
        if len(args) == 0:
            self._log_key_value([(k, v) for k, v in kwargs.items()], False)
        elif len(args) == 1:
            assert len(kwargs.keys()) == 0
            arg = args[0]
            if type(arg) == list:
                self._log_key_value([(i, v) for i, v in enumerate(arg)])
            elif type(arg) == dict:
                keys = list(arg.keys())
                keys.sort()
                self._log_key_value([(k, arg[k]) for k in keys])
            else:
                self.__logger.log(_get_value_full(arg))
        else:
            assert len(kwargs.keys()) == 0
            self._log_key_value([(i, v) for i, v in enumerate(args)], False)
