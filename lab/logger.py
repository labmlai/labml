"""
# Logger class

*Should be initialized via `Experiment`*

This module contains logging and monotring helpers.

Logger prints to the screen and writes TensorBoard summaries.
"""
import signal
import time
from collections import deque, ItemsView
from typing import Dict, List, Tuple, Iterator, Optional

import numpy as np
import tensorflow as tf

from lab import colors
from lab.colors import ANSICode


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


class _DelayedKeyboardInterrupt:
    """
    ### Capture `KeyboardInterrupt` and fire it later
    """

    def __init__(self, logger: 'Logger'):
        self.signal_received = None
        self.logger = logger

    def __enter__(self):
        self.signal_received = None
        # Start capturing
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        # Pass second interrupt without delaying
        if self.signal_received is not None:
            self.old_handler(*self.signal_received)
            return

        # Store the interrupt signal for later
        self.signal_received = (sig, frame)
        self.logger.log('\nSIGINT received. Delaying KeyboardInterrupt.',
                        color=colors.Color.red)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Reset handler
        signal.signal(signal.SIGINT, self.old_handler)

        # Pass on any captured interrupt signals
        if self.signal_received is not None:
            self.old_handler(*self.signal_received)


class _Monitor:
    """
    ### Monitors a section of code

    *Should be initialized via `Logger`.*

    This is used monitor time taken for the section of code to run.
    It also helps structure the code.

    You can set `is_successful` to `False` if the execution failed.
    """

    def __init__(self, name: str, *,
                 logger: 'Logger',
                 silent: bool = False, timed: bool = True):
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
        self.logger = logger

    def __enter__(self):
        if self.silent:
            return self

        self.logger.log(f"{self.name}...", new_line=False)

        if self.timed:
            self._start_time = time.time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.silent:
            return

        parts = []
        if self.is_successful:
            parts.append(("[DONE]", colors.BrightColor.green))
        else:
            parts.append(("[FAIL]", colors.BrightColor.red))

        if self.timed:
            time_end = time.time()
            duration_ms = 1000 * (time_end - self._start_time)
            parts.append((f"\t{duration_ms :,.2f}ms",
                          colors.BrightColor.cyan))

        self.logger.log_color(parts)


class _MonitorIteratorSection:
    """
    ### Monitors a section of code within an iterator

    *Should only be initialzed via `_MonitorIterator`.*

    It keeps track of moving exponentiol average of
     time spent on the section through out all iterations.
    """

    def __init__(self, parent, name: str, *,
                 logger: 'Logger'):
        self.parent = parent
        self.name = name
        self._start_time = 0

        self.beta = 0.9
        self.beta_pow = 1

        self.estimated_time = 0

        self.logger = logger

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
        self.logger.log(f"{self.name}:", new_line=False)
        self.logger.log(" ...", color=colors.Color.orange, new_line=False)
        self.logger.log(" " * 8, new_line=False)
        self.logger.pop_current_line()
        self._start_time = time.time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.add_time(time.time() - self._start_time)
        self.logger.pop_current_line()
        time_ms = 1000 * self.get_time()
        self.logger.log(f"{time_ms:10,.2f}ms  ",
                        color=colors.BrightColor.cyan,
                        new_line=False)


class _UnmonitoredIteratorSection:
    """
    ### Unmonitored section in a monitored iterator.

    Used for structuring code.
    """

    def __init__(self, parent, name: str, *,
                 logger: 'Logger'):
        self.parent = parent
        self.name = name
        self.logger = logger

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MonitorIterator:
    """
    ### Monitor an iterator

    *Should be initialized only via `Logger`.*
    """

    def __init__(self, iterator: range, *,
                 logger: 'Logger'):
        """
        Creates an iterator with a range `iterator`.

        See example for usage.
        """
        self.iterator = iterator
        self.sections = {}
        self._start_time = 0
        self.steps = len(iterator)
        self.counter = 0
        self.logger = logger

    def section(self, name: str, is_monitored=True):
        """
        Creates a monitored section with given name.
        """

        if is_monitored:
            if name not in self.sections:
                self.sections[name] = _MonitorIteratorSection(self, name,
                                                              logger=self.logger)
        else:
            if name not in self.sections:
                self.sections[name] = _UnmonitoredIteratorSection(self, name,
                                                                  logger=self.logger)

        return self.sections[name]

    def unmonitored(self, name: str):
        """
        Creates an unmonitored section with given name.
        """

        return self.section(name, is_monitored=False)

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

        spent_h = int(spent // 60)
        spent_m = int(spent % 60)
        remain_h = int(remain // 60)
        remain_m = int(remain % 60)

        self.logger.log(f"  {spent_h:3d}:{spent_m:02d}m/{remain_h:3d}:{remain_m:02d}m  ",
                        color=colors.BrightColor.purple,
                        new_line=False)


class _Progress:
    """
    ### Manually monitor percentage progress.
    """

    def __init__(self, total: float, *,
                 logger: 'Logger'):
        self.total = total
        self.logger = logger
        self.logger.log(f" {0.0 :4.0f}%", new_line=False)

    def update(self, value):
        """
        Update progress
        """
        percentage = min(1, max(0, value / self.total)) * 100
        self.logger.pop_current_line()
        self.logger.log(f" {percentage :4.0f}%",
                        new_line=False)

    def clear(self):
        """
        Clear line
        """
        self.logger.pop_current_line()


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
        self.scalars = {}
        self.writer: tf.summary.FileWriter = None
        self.print_order = []
        self.progress_indicators = []
        self.tf_summaries = []

        self.is_color = is_color

        self.current_line = []
        self.over_write_line = ""

    def ansi_code(self, text: str, color: List[ANSICode] or ANSICode or None):
        """
        ### Add ansi color codes
        """
        if color is None:
            return text
        elif type(color) is list:
            return "".join(color) + f"{text}{colors.Reset}"
        else:
            return f"{color}{text}{colors.Reset}"

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
        lim = self.count_text(self.over_write_line,
                              self.count_text(text))

        if lim < len(self.over_write_line):
            text += self.over_write_line[lim:]
            self.over_write_line = text

        print("\r" + text, end=end_char, flush=True)
        if new_line:
            self.current_line = []
            self.over_write_line = ""

    def count_text(self, text, limit=None):
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
                      is_progress: Optional[bool] = None):
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
        self.tf_summaries = []

    def write(self, *, global_step: int, new_line: bool = True):
        """
        ### Output the stored log values to screen and TensorBoard summaries.
        """
        self._write_to_writer(global_step=global_step)
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
        return _Monitor(*args, **kwargs)

    def progress(self, *args, **kwargs):
        """
        ### Create a progress monitor
        """
        kwargs['logger'] = self
        return _Progress(*args, **kwargs)

    def delayed_keyboard_interrupt(self):
        """
        ### Create a section with a delayed keyboard interrupt
        """
        return _DelayedKeyboardInterrupt(self)

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
