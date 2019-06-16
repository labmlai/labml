"""
# Logger class

*Should be initialized via `Experiment`*

This module contains logging and monotring helpers.

Logger prints to the screen and writes TensorBoard summaries.
"""

from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np

from lab import colors
from lab.colors import ANSICode
from lab.logger.delayed_keyboard_interrupt import DelayedKeyboardInterrupt
from lab.logger.monitor import Monitor
from lab.logger.monitor_iterator import MonitorIterator
from lab.logger.progress import Progress


class Writer:
    def write(self, *, global_step: int,
              queues,
              histograms,
              pairs,
              scalars,
              tf_summaries):
        raise NotImplementedError()


class Logger:
    """
    ## 🖨 Logger class
    """

    def __init__(self, *, is_color=True):
        """
        ### Initializer
        :param is_color: whether to use colours in console output
        """
        self.queues = {}
        self.histograms = {}
        self.pairs: Dict[str, List[Tuple[int, int]]] = {}
        self.scalars = {}
        self.__writers: List[Writer] = []
        self.print_order = []
        self.progress_indicators = []
        self.tf_summaries = []

        self.is_color = is_color

        self.current_line = []
        self.over_write_line = ""

    @staticmethod
    def ansi_code(text: str, color: List[ANSICode] or ANSICode or None):
        """
        ### Add ansi color codes
        """
        if color is None:
            return text
        elif type(color) is list:
            return "".join(color) + f"{text}{colors.Reset}"
        else:
            return f"{color}{text}{colors.Reset}"

    def add_writer(self, writer: Writer):
        self.__writers.append(writer)

    def log(self, message, *,
            color: List[ANSICode] or ANSICode or None = None,
            new_line=True):
        """
        ### Print a message to screen in color
        """

        message = self.ansi_code(message, color)

        self.current_line.append(message)

        if new_line:
            end_char = '\n'
        else:
            end_char = ''

        text = "".join(self.current_line)
        lim = self.__count_text(self.over_write_line,
                                self.__count_text(text))

        if lim < len(self.over_write_line):
            text += self.over_write_line[lim:]
            self.over_write_line = text

        print("\r" + text, end=end_char, flush=True)
        if new_line:
            self.current_line = []
            self.over_write_line = ""

    @staticmethod
    def __count_text(text, limit=None):
        """
        ### Count text length without color codes
        """

        count = 0
        is_text = True
        for i, c in enumerate(text):
            if is_text and c == '\33':
                is_text = False

            if is_text:
                count += 1
            if limit is not None and count == limit:
                return i + 1

            if not is_text and c == 'm':
                is_text = True

        if limit is not None:
            return len(text)
        else:
            return count

    def pop_current_line(self):
        """
        ### Pop last segment from current line
        """
        self.current_line.pop()

    def log_color(self, parts: List[Tuple[str, ANSICode or None]], *,
                  new_line=True):
        """
        ### Print a message with different colors.
        """

        coded = [self.ansi_code(text, color) for text, color in parts]
        self.log("".join(coded), new_line=new_line)

    def add_indicator(self, name: str, *,
                      queue_limit: int = None,
                      is_histogram: bool = True,
                      is_print: bool = True,
                      is_progress: Optional[bool] = None,
                      is_pair: bool = False):
        """
        ### Add an indicator
        """

        if queue_limit is not None:
            self.queues[name] = deque(maxlen=queue_limit)
        elif is_histogram:
            self.histograms[name] = []
        else:
            self.scalars[name] = []

        if is_print:
            self.print_order.append(name)

        if is_progress is None:
            is_progress = is_print
        if is_progress:
            self.progress_indicators.append(name)

        if is_pair:
            assert not is_print and not is_progress and not is_histogram and queue_limit is None
            self.pairs[name] = []

    def _store_list(self, items: List[Dict[str, float]]):
        for item in items:
            self.store(**item)

    def _store_kv(self, k, v):
        if k in self.queues:
            self.queues[k].append(v)
        elif k in self.histograms:
            self.histograms[k].append(v)
        elif k in self.pairs:
            if type(v) == tuple:
                assert len(v) == 2
                self.pairs[k].append((v[0], v[1]))
            else:
                assert type(v) == list
                self.pairs[k] += v
        else:
            self.scalars[k].append(v)

    def _store_kvs(self, **kwargs):
        for k, v in kwargs.items():
            self._store_kv(k, v)

    def has_key(self, k):
        if k in self.queues:
            return len(self.queues[k]) > 0
        elif k in self.histograms:
            return len(self.histograms[k]) > 0
        elif k in self.pairs:
            return len(self.pairs[k]) > 0
        else:
            return len(self.scalars[k]) > 0

    def store(self, *args, **kwargs):
        """
        ### Stores a value in the logger.

        This may be added to a queue, a list or stored as
        a TensorBoard histogram depending on the
        type of the indicator.
        """
        assert len(args) <= 2

        if len(args) == 0:
            self._store_kvs(**kwargs)
        elif len(args) == 1:
            assert not kwargs
            if isinstance(args[0], list):
                self._store_list(args[0])
            else:
                assert isinstance(args[0], bytes)
                self.tf_summaries.append(args[0])
        elif len(args) == 2:
            assert isinstance(args[0], str)
            if isinstance(args[1], list):
                for v in args[1]:
                    self._store_kv(args[0], v)
            else:
                self._store_kv(args[0], args[1])

    def _write_to_screen(self, *, new_line: bool):
        parts = []

        for k in self.print_order:
            if k in self.queues:
                if len(self.queues[k]) == 0:
                    continue
                v = np.mean(self.queues[k])
            elif k in self.histograms:
                if len(self.histograms[k]) == 0:
                    continue
                v = np.mean(self.histograms[k])
            else:
                if len(self.scalars[k]) == 0:
                    continue
                v = np.mean(self.scalars[k])

            parts.append((f" {k}: ", None))
            if self.is_color:
                parts.append((f"{v :8,.2f}", colors.Style.bold))
            else:
                parts.append((f"{v :8,.2f}", None))

        self.log_color(parts, new_line=new_line)

    def get_progress_dict(self, *, global_step: int):
        """
        ### Get progress dictionary

        This is used for adding progress to trial information
        """
        res = dict(global_step=f"{global_step :8,}")

        for k in self.progress_indicators:
            if k in self.queues:
                if len(self.queues[k]) == 0:
                    continue
                v = np.mean(self.queues[k])
            elif k in self.histograms:
                if len(self.histograms[k]) == 0:
                    continue
                v = np.mean(self.histograms[k])
            else:
                if len(self.scalars[k]) == 0:
                    continue
                v = np.mean(self.scalars[k])

            res[k] = f"{v :8,.2f}"

        return res

    def _clear_stores(self):
        for k in self.histograms:
            self.histograms[k] = []
        for k in self.scalars:
            self.scalars[k] = []
        for k in self.pairs:
            self.pairs[k] = []
        self.tf_summaries = []

    def write(self, *, global_step: int, new_line: bool = True):
        """
        ### Output the stored log values to screen and TensorBoard summaries.
        """
        for w in self.__writers:
            w.write(global_step=global_step,
                    queues=self.queues,
                    histograms=self.histograms,
                    pairs=self.pairs,
                    scalars=self.scalars,
                    tf_summaries=self.tf_summaries)
        self._write_to_screen(new_line=new_line)
        self._clear_stores()

    def print_global_step(self, global_step):
        """
        ### Print the global step
        """
        self.log(f"{global_step :8,}:  ",
                 color=colors.BrightColor.orange,
                 new_line=False)

    def clear_line(self, reset: bool):
        """
        ### Clears the current line
        """
        if reset:
            print("\r", end="", flush=True)
            self.over_write_line = "".join(self.current_line)
        else:
            self.over_write_line = ""
            print()

        self.current_line = []

    def iterator(self, *args, **kwargs):
        """
        ### Create a monitored iterator
        """
        kwargs['logger'] = self
        return MonitorIterator(*args, **kwargs)

    def monitor(self, *args, **kwargs):
        """
        ### Create a code section monitor
        """
        kwargs['logger'] = self
        return Monitor(*args, **kwargs)

    def progress(self, *args, **kwargs):
        """
        ### Create a progress monitor
        """
        kwargs['logger'] = self
        return Progress(*args, **kwargs)

    def delayed_keyboard_interrupt(self):
        """
        ### Create a section with a delayed keyboard interrupt
        """
        return DelayedKeyboardInterrupt(self)

    def _log_key_value(self, items: List[Tuple[any, any]]):
        max_key_len = 0
        for k, v in items:
            max_key_len = max(max_key_len, len(str(k)))

        count = 0
        for k, v in items:
            count += 1
            spaces = " " * (max_key_len - len(str(k)))
            self.log_color([(f"{spaces}{k}: ", None),
                            (str(v), colors.Style.bold)])

        self.log_color([
            ("Total ", None),
            (str(count), colors.Style.bold),
            (" item(s)", None)])

    def info(self, *args, **kwargs):
        """
        ### 🎨 Pretty prints a set of values.
        """

        if len(args) == 0:
            self._log_key_value([(k, v) for k, v in kwargs.items()])
        elif len(args) == 1:
            assert len(kwargs.keys()) == 0
            arg = args[0]
            if type(arg) == list:
                self._log_key_value([(i, v) for i, v in enumerate(arg)])
            elif type(arg) == dict:
                self._log_key_value([(k, v) for k, v in arg.items()])
        else:
            assert len(kwargs.keys()) == 0
            self._log_key_value([(i, v) for i, v in enumerate(args)])
