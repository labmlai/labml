"""
# Logger class

*Should be initialized via `Experiment`*

This module contains logging and monotring helpers.

Logger prints to the screen and writes TensorBoard summaries.
"""
import typing
from typing import List, Tuple, Optional, Dict

from lab import colors
from lab.colors import ANSICode
from lab.logger_class import iterator
from lab.logger_class.delayed_keyboard_interrupt import DelayedKeyboardInterrupt
from lab.logger_class.loop import Loop
from lab.logger_class.sections import Section, OuterSection, LoopingSection, section_factory
from lab.logger_class.store import Store
from lab.logger_class.writers import Writer, ProgressDictWriter, ScreenWriter


class ProgressSaver:
    def save(self, progress: Dict[str, str]):
        raise NotImplementedError()


class CheckpointSaver:
    def save(self, global_step, args):
        raise NotImplementedError()


logger_singleton = None


class Logger:
    """
    ## 🖨 Logger class
    """

    def __init__(self):
        """
        ### Initializer
        """
        if logger_singleton is not None:
            raise RuntimeError("Only one instance of logger can exist")

        self.__store = Store()
        self.__writers: List[Writer] = []

        self.__loop: Optional[Loop] = None
        self.__sections: List[Section] = []

        self.__indicators_print = []
        self.__progress_dict = {}

        self.__screen_writer = ScreenWriter(True)
        self.__progress_dict_writer = ProgressDictWriter()

        self.__progress_saver: Optional[ProgressSaver] = None
        self.__checkpoint_saver: Optional[CheckpointSaver] = None

        self.__start_global_step: Optional[int] = None
        self.__global_step: Optional[int] = None
        self.__last_global_step: Optional[int] = None

    def set_progress_saver(self, saver: ProgressSaver):
        self.__progress_saver = saver

    def set_checkpoint_saver(self, saver: CheckpointSaver):
        self.__checkpoint_saver = saver

    @property
    def global_step(self) -> int:
        if self.__global_step is not None:
            return self.__global_step

        global_step = 0
        if self.__start_global_step is not None:
            global_step = self.__start_global_step

        if self.__loop is not None:
            return global_step + self.__loop.counter

        if self.__last_global_step is not None:
            return self.__last_global_step

        return global_step

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

        if new_line:
            end_char = '\n'
        else:
            end_char = ''

        text = "".join(message)

        print("\r" + text, end=end_char, flush=True)

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

        if is_print:
            self.__screen_writer.add_indicator(name)

        if is_progress is None:
            is_progress = is_print

        if is_progress:
            self.__progress_dict_writer.add_indicator(name)

        if is_pair:
            assert not is_print and not is_progress and not is_histogram and queue_limit is None

        self.__store.add_indicator(name,
                                   queue_limit=queue_limit,
                                   is_histogram=is_histogram,
                                   is_pair=is_pair)

    def store(self, *args, **kwargs):
        """
        ### Stores a value in the logger_class.

        This may be added to a queue, a list or stored as
        a TensorBoard histogram depending on the
        type of the indicator.
        """

        self.__store.store(*args, **kwargs)

    def set_global_step(self, global_step):
        self.__global_step = global_step

    def set_start_global_step(self, global_step):
        self.__start_global_step = global_step

    def add_global_step(self, global_step: int = 1):
        if self.__global_step is None:
            if self.__start_global_step is not None:
                self.__global_step = self.__start_global_step
            else:
                self.__global_step = 0

        self.__global_step += global_step

    @property
    def progress_dict(self):
        return self.__progress_dict

    @staticmethod
    def new_line():
        print()

    def write(self):
        """
        ### Output the stored log values to screen and TensorBoard summaries.
        """

        global_step = self.global_step

        for w in self.__writers:
            self.__store.write(w, global_step)
        self.__indicators_print = self.__store.write(self.__screen_writer, global_step)
        self.__progress_dict = self.__store.write(self.__progress_dict_writer, global_step)
        self.__store.clear()

        parts = [(f"{self.global_step :8,}:  ", colors.BrightColor.orange)]
        if self.__loop is None:
            parts += self.__indicators_print
        else:
            parts += self.__loop.log_sections()
            parts += self.__indicators_print
            parts += self.__loop.log_progress()

        self.log_color(parts, new_line=False)

    def save_progress(self):
        if self.__progress_saver is None:
            return

        self.__progress_saver.save(self.__progress_dict)

    def save_checkpoint(self, *args):
        if self.__checkpoint_saver is None:
            return

        self.__checkpoint_saver.save(self.global_step, args)

    def iterator(self, name, iterable: typing.Union[typing.Iterable, typing.Sized, int],
                 total_steps: Optional[int] = None, *,
                 is_silent: bool = False,
                 is_timed: bool = True):
        return iterator.Iterator(logger=self,
                                 name=name,
                                 iterable=iterable,
                                 is_silent=is_silent,
                                 is_timed=is_timed,
                                 total_steps=total_steps,
                                 is_enumarate=False)

    def enumerator(self, name, iterable: typing.Sized, *,
                   is_silent: bool = False,
                   is_timed: bool = True):
        return iterator.Iterator(logger=self,
                                 name=name,
                                 iterable=iterable,
                                 is_silent=is_silent,
                                 is_timed=is_timed,
                                 total_steps=None,
                                 is_enumarate=True)

    def section(self, name, *,
                is_silent: bool = False,
                is_timed: bool = True,
                is_partial: bool = False,
                total_steps: float = 1.0):

        if self.__loop is not None:
            if len(self.__sections) != 0:
                raise RuntimeError("No nested sections within loop")

            section = self.__loop.get_section(name=name,
                                              is_silent=is_silent,
                                              is_timed=is_timed,
                                              is_partial=is_partial,
                                              total_steps=total_steps)
            self.__sections.append(section)
        else:
            self.__sections.append(section_factory(logger=self,
                                                   name=name,
                                                   is_silent=is_silent,
                                                   is_timed=is_timed,
                                                   is_partial=is_partial,
                                                   total_steps=total_steps,
                                                   is_looping=False,
                                                   level=len(self.__sections)))

        return self.__sections[-1]

    def progress(self, steps: float):
        if len(self.__sections) == 0:
            raise RuntimeError("You must be within a section to report progress")

        if self.__sections[-1].progress(steps):
            self.__log_line()

    def set_successful(self, is_successful=True):
        if len(self.__sections) == 0:
            raise RuntimeError("You must be within a section to report success")

        self.__sections[-1].is_successful = is_successful
        self.__log_line()

    def loop(self, iterator: range, *,
             is_print_iteration_time=True):
        if len(self.__sections) != 0:
            raise RuntimeError("Cannot start a loop within a section")

        self.__loop = Loop(iterator=iterator, logger=self,
                           is_print_iteration_time=is_print_iteration_time)
        return self.__loop

    def finish_loop(self):
        if len(self.__sections) != 0:
            raise RuntimeError("Cannot be within a section when finishing the loop")
        self.__last_global_step = self.global_step
        self.__loop = None

    def section_enter(self, section):
        if len(self.__sections) == 0:
            raise RuntimeError("Entering a section without creating a section.\n"
                               "Always use logger_class.section to create a section")

        if section is not self.__sections[-1]:
            raise RuntimeError("Entering a section other than the one last_created\n"
                               "Always user with logger_class.section(...):")

        if len(self.__sections) > 1 and not self.__sections[-2].is_parented:
            self.__sections[-2].make_parent()
            self.new_line()

        self.__log_line()

    def __log_line(self):
        if len(self.__sections) == 0:
            return

        self.log_color(self.__sections[-1].log(), new_line=False)

    def section_exit(self, section):
        if len(self.__sections) == 0:
            raise RuntimeError("Impossible")

        if section is not self.__sections[-1]:
            raise RuntimeError("Impossible")

        self.__log_line()
        self.__sections.pop(-1)

    def delayed_keyboard_interrupt(self):
        """
        ### Create a section with a delayed keyboard interrupt
        """
        return DelayedKeyboardInterrupt(self)

    def _log_key_value(self, items: List[Tuple[any, any]], is_show_count=True):
        max_key_len = 0
        for k, v in items:
            max_key_len = max(max_key_len, len(str(k)))

        count = 0
        for k, v in items:
            count += 1
            spaces = " " * (max_key_len - len(str(k)))
            self.log_color([(f"{spaces}{k}: ", None),
                            (str(v), colors.Style.bold)])

        if is_show_count:
            self.log_color([
                ("Total ", None),
                (str(count), colors.Style.bold),
                (" item(s)", None)])

    def info(self, *args, **kwargs):
        """
        ### 🎨 Pretty prints a set of values.
        """

        if len(args) == 0:
            self._log_key_value([(k, v) for k, v in kwargs.items()], False)
        elif len(args) == 1:
            assert len(kwargs.keys()) == 0
            arg = args[0]
            if type(arg) == list:
                self._log_key_value([(i, v) for i, v in enumerate(arg)])
            elif type(arg) == dict:
                self._log_key_value([(k, v) for k, v in arg.items()])
        else:
            assert len(kwargs.keys()) == 0
            self._log_key_value([(i, v) for i, v in enumerate(args)], False)


logger_singleton = Logger()
