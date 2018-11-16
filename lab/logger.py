"""
# Logger class

*Should be initialized via `Experiment`*

This module contains logging and monotring helpers.

Logger prints to the screen and writes TensorBoard summaries.
"""

import time
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from lab import colors


def _print_color(text: str, color: str or None, *args, **kwargs):
    """
    Helper function for colored console output
    """
    if color is not None:
        print(colors.CodeStart + color + text + colors.CodeStart + colors.Reset,
              *args, **kwargs)
    else:
        print(text, *args, **kwargs)


def _get_histogram(values):
    """
    Get TensorBoard histogram from a numpy array.
    """

    values = np.array(values)
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values ** 2))

    counts, bin_edges = np.histogram(values, bins=20)
    bin_edges = bin_edges[1:]

    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    return hist


class _Monitor:
    """
    ### Monitors a section of code

    *Should be initialized via `Logger`.*

    This is used monitor time taken for the section of code to run.
    It also helps structure the code.

    You can set `is_successful` to `False` if the execution failed.
    """

    def __init__(self, name: str, *, silent: bool = False, timed: bool = True):
        """

        :param name: name of the section of code
        :param silent: whether or not to output to screen
        :param timed: whether to time the section of code
        """
        self.name = name
        self.silent = silent
        self.timed = timed
        self._start_time = 0
        self.is_successful = True

    def __enter__(self):
        if self.silent:
            return self

        print("{}...".format(self.name), end="", flush=True)

        if self.timed:
            self._start_time = time.time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.silent:
            return

        if self.timed:
            time_end = time.time()
            if self.is_successful:
                _print_color("[DONE]", colors.BrightColor.green, end="", flush=False)
            else:
                _print_color("[FAIL]", colors.BrightColor.red, end="", flush=False)
            _print_color("\t{:,.2f}ms".format(1000 * (time_end - self._start_time)),
                         colors.BrightColor.cyan)
        else:
            if self.is_successful:
                _print_color("[DONE]", colors.BrightColor.green)
            else:
                _print_color("[FAIL]", colors.BrightColor.red)


class _MonitorIteratorSection:
    """
    ### Monitors a section of code within an iterator

    *Should only be initialzed via `_MonitorIterator`.*

    It keeps track of moving exponentiol average of
     time spent on the section through out all iterations.
    """
    def __init__(self, parent, name: str):
        self.parent = parent
        self.name = name
        self._start_time = 0

        self.beta = 0.9
        self.beta_pow = 1

        self.estimated_time = 0

    def get_time(self):
        """
        Get the moving exponential average time.
        """
        if self.estimated_time <= 0:
            return self.estimated_time

        return self.estimated_time / (1 - self.beta_pow)

    def add_time(self, elapsed):
        """
        Update moving average time
        """
        self.beta_pow *= self.beta
        self.estimated_time = self.beta * self.estimated_time + (1 - self.beta) * elapsed

    def __enter__(self):
        print("{}: ...".format(self.name) + " " * 8 + "\b" * 8, end="",
              flush=True)
        self._start_time = time.time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.add_time(time.time() - self._start_time)
        print("\b" * 4, end="", flush=True)
        _print_color("{:10,.2f}ms  ".format(1000 * self.get_time()),
                     colors.BrightColor.cyan,
                     end="", flush=False)


class _UnmonitoredIteratorSection:
    """
    ### Unmonitored section in a monitored iterator.

    Used for structuring code.
    """
    def __init__(self, parent, name: str):
        self.parent = parent
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class _MonitorIterator:
    """
    ### Monitor an iterator

    *Should be initialized only via `Logger`.*
    """
    def __init__(self, iterator: range):
        """
        Creates an iterator with a range `iterator`.

        See example for usage.
        """
        self.iterator = iterator
        self.sections = {}
        self._start_time = 0
        self.steps = len(iterator)
        self.counter = 0

    def section(self, name: str):
        """
        Creates a monitored section with given name.
        """
        if name not in self.sections:
            self.sections[name] = _MonitorIteratorSection(self, name)

        return self.sections[name]

    def unmonitored(self, name: str):
        """
        Creates an unmonitored section with given name.
        """
        if name not in self.sections:
            self.sections[name] = _UnmonitoredIteratorSection(self, name)

        return self.sections[name]

    def __iter__(self):
        self.iterator_iter = iter(self.iterator)
        self._start_time = time.time()
        self.counter = 0
        return self

    def __next__(self):
        try:
            next_value = next(self.iterator_iter)
        except StopIteration as e:
            raise e

        self.counter += 1

        return next_value

    def progress(self):
        """
        Show progress
        """
        spent = (time.time() - self._start_time) / 60
        remain = self.steps * spent / self.counter - spent

        _print_color("  {:3d}:{:02d}m/{:3d}:{:02d}m  ".format(int(spent // 60),
                                                              int(spent % 60),
                                                              int(remain // 60),
                                                              int(remain % 60)),
                     colors.BrightColor.purple,
                     end="", flush=True)


class _Progress:
    """
    ### Manually monitor percentage progress.
    """

    def __init__(self, total: float):
        self.total = total
        print(" {:4.0f}%".format(0.0), end="", flush=True)

    def update(self, value):
        """
        Update progress
        """
        percentage = min(1, max(0, value / self.total))
        print("\b" * 5 + "{:4.0f}%".format(percentage * 100), end="", flush=True)

    def clear(self):
        """
        Clear line
        """
        print("\b" * 6 + " " * 6 + "\b" * 6, end="", flush=True)


class Logger:
    """
    ## Logger class
    """
    def __init__(self, *, is_color=True):
        """
        ### Initializer
        :param is_color: whether to use colours in console output
        """
        self.queues = {}
        self.histograms = {}
        self.scalars = {}
        self.writer: tf.summary.FileWriter = None
        self.print_order = []
        self.tf_summaries = []

        self.is_color = is_color

    def log(self, message, *, color: str = None, new_line=True):
        """
        Print a message to screen in color
        """
        end_char = '\n' if new_line else ''

        if color is not None:
            _print_color(message, color, end=end_char, flush=True)
        else:
            print(message,
                  end=end_char, flush=True)

    def log_color(self, parts: List[Tuple[str, str or None]]):
        """
        Print a message with different colors.
        """
        for text, color in parts:
            _print_color(text, color, end="", flush=False)
        _print_color("", None)

    def add_indicator(self, name: str, *,
                      queue_limit: int = None,
                      is_histogram: bool = True,
                      is_print: bool = True):
        """
        Sets an indicator
        """
        if queue_limit is not None:
            self.queues[name] = deque(maxlen=queue_limit)
        elif is_histogram:
            self.histograms[name] = []
        else:
            self.scalars[name] = []

        if is_print:
            self.print_order.append(name)

    def _store_list(self, items: List[Dict[str, float]]):
        for item in items:
            self.store(**item)

    def _store_kv(self, k, v):
        if k in self.queues:
            self.queues[k].append(v)
        elif k in self.histograms:
            self.histograms[k].append(v)
        else:
            self.scalars[k].append(v)

    def _store_kvs(self, **kwargs):
        for k, v in kwargs.items():
            self._store_kv(k, v)

    def store(self, *args, **kwargs):
        """
        Stores a value in the logger.
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

    def _write_to_writer(self, *, global_step: int):
        if self.writer is None:
            return

        summary = tf.Summary()

        for k, v in self.queues.items():
            if len(v) == 0:
                continue
            summary.value.add(tag=k, histo=_get_histogram(v))
            summary.value.add(tag=f"{k}_mean", simple_value=float(np.mean(v)))

        for k, v in self.histograms.items():
            if len(v) == 0:
                continue
            summary.value.add(tag=k, histo=_get_histogram(v))
            summary.value.add(tag=f"{k}_mean", simple_value=float(np.mean(v)))

        for k, v in self.scalars.items():
            if len(v) == 0:
                continue
            summary.value.add(tag=k, simple_value=float(np.mean(v)))

        self.writer.add_summary(summary, global_step=global_step)

        for v in self.tf_summaries:
            self.writer.add_summary(v, global_step=global_step)

    def _write_to_screen(self, *, new_line: bool):
        print_log = ""

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

            if self.is_color:
                print_log += " {}: {}{:8,.2f}{}".format(k,
                                                        colors.CodeStart + colors.Style.bold,
                                                        v,
                                                        colors.CodeStart + colors.Reset)
            else:
                print_log += " {}: {:8,.2f}".format(k, v)

        end_char = "\n" if new_line else ""

        print(print_log, end=end_char, flush=True)

    def _clear_stores(self):
        for k in self.histograms:
            self.histograms[k] = []
        for k in self.scalars:
            self.scalars[k] = []
        self.tf_summaries = []

    def write(self, *, global_step: int, new_line: bool = True):
        """
        Output the stored log values to screen and TensorBoard summaries.
        """
        self._write_to_writer(global_step=global_step)
        self._write_to_screen(new_line=new_line)
        self._clear_stores()

    def print_global_step(self, global_step):
        """
        Outputs the global step
        """
        _print_color("{:8,}".format(global_step),
                     colors.BrightColor.orange,
                     end="")
        print(":  ", end="", flush=True)

    def clear_line(self, reset: bool):
        """
        Clears the current line
        """
        if reset:
            # We don't clear the previous line so that it stays visible for viewing
            print("\r", end="", flush=True)
            # print("\r" + " " * 200 + "\r", end="", flush=True)
        else:
            print()

    def iterator(self, *args, **kwargs):
        """
        Creates a monitored iterator
        """
        return _MonitorIterator(*args, **kwargs)

    def monitor(self, *args, **kwargs):
        """
        Creates a code section monitor
        """
        return _Monitor(*args, **kwargs)

    def progress(self, *args, **kwargs):
        """
        Creates a progress monitor
        """
        return _Progress(*args, **kwargs)

    def info(self, *args, **kwargs):
        """
        Pretty prints a set of values.
        """
        assert len(args) == 0

        for key, value in kwargs.items():
            print("{}: {}{}{}".format(key,
                                      colors.CodeStart + colors.BrightColor.cyan,
                                      value,
                                      colors.CodeStart + colors.Reset))
