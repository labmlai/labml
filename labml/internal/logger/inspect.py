from typing import TYPE_CHECKING, List, Tuple

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


class Inspect:
    def __init__(self, logger: 'Logger'):
        self.__logger = logger

    def _key_value_pair(self, key: any, value: any, style: StyleCode = Text.meta):
        return [(f'{str(key)}: ', Text.subtle),
                (str(value), style)]

    def _format_tensor(self, s: List[str], limit: int = 1_000, style: StyleCode = Text.value):
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

    def _get_tensor_value(self, tensor):
        if torch is not None and isinstance(tensor, torch.Tensor):
            return str(tensor.item())
        else:
            return str(tensor)

    def _render_tensor(self, tensor, *, new_line: str = '\n', indent: str = ''):
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
                    res += self._render_tensor(tensor[i],
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
                    res += self._get_tensor_value(tensor[i])
                if i != idx[-1]:
                    res.append(', ')

        res.append(indent)
        res.append(']')

        return res

    def _get_value_full(self, value: any):
        if isinstance(value, str):
            return [('"', Text.subtle),
                    (value, Text.value),
                    ('"', Text.subtle)]
        elif numpy is not None and isinstance(value, numpy.ndarray):
            return [*self._key_value_pair('dtype', value.dtype),
                    '\n',
                    *self._key_value_pair('shape', [s for s in value.shape], Text.value),
                    '\n',
                    *self._key_value_pair('min', numpy.min(value)),
                    ' ',
                    *self._key_value_pair('max', numpy.max(value)),
                    ' ',
                    *self._key_value_pair('mean', numpy.mean(value)),
                    ' ',
                    *self._key_value_pair('std', numpy.std(value)),
                    '\n',
                    *self._format_tensor(self._render_tensor(value, new_line='\n'))]
        elif torch is not None and isinstance(value, torch.Tensor):
            return [*self._key_value_pair('dtype', value.dtype),
                    '\n',
                    *self._key_value_pair('shape', [s for s in value.shape], Text.value),
                    '\n',
                    *self._key_value_pair('min', torch.min(value).item()),
                    ' ',
                    *self._key_value_pair('max', torch.max(value).item()),
                    ' ',
                    *self._key_value_pair('mean', torch.mean(value.to(torch.float)).item()),
                    ' ',
                    *self._key_value_pair('std', torch.std(value.to(torch.float)).item()),
                    '\n',
                    *self._format_tensor(self._render_tensor(value, new_line='\n'))]

        s = str(value)
        s = s.replace('\r', '')
        return [s]

    def _shrink(self, s: str, style: StyleCode = Text.value, limit: int = 80):
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

    def _get_value_line(self, value: any):
        if isinstance(value, str):
            return [('"', Text.subtle)] + self._shrink(value) + [('"', Text.subtle)]
        elif numpy is not None and isinstance(value, numpy.ndarray):
            return [*self._format_tensor(self._render_tensor(value, new_line=''), limit=80)]
        elif torch is not None and isinstance(value, torch.Tensor):
            return [*self._format_tensor(self._render_tensor(value, new_line=''), limit=80)]

        s = str(value)
        return self._shrink(s)

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
                              self._get_value_line(v))
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
                self.__logger.log(self._get_value_full(arg))
        else:
            assert len(kwargs.keys()) == 0
            self._log_key_value([(i, v) for i, v in enumerate(args)], False)
