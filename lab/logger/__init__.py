"""
# Logger class

*Should be initialized via `Experiment`*

This module contains logging and monotring helpers.

Logger prints to the screen and writes TensorBoard summaries.
"""

from typing import List, Tuple, Optional

from lab import colors
from lab.colors import ANSICode
from lab.logger.delayed_keyboard_interrupt import DelayedKeyboardInterrupt
from lab.logger.loop import Loop
from lab.logger.sections import Section, OuterSection, LoopingSection, section_factory
from lab.logger.store import Store
from lab.logger.writers import Writer, ProgressDictWriter, ScreenWriter


class Logger:
    """
    ## 🖨 Logger class
    """

    def __init__(self, *, is_color=True):
        """
        ### Initializer
        :param is_color: whether to use colours in console output
        """
        self.__store = Store()
        self.__writers: List[Writer] = []

        self.is_color = is_color

        self.__loop: Optional[Loop] = None
        self.__sections: List[Section] = []

        self.__indicators_print = []
        self.__progress_dict = {}

        self.__screen_writer = ScreenWriter(is_color)
        self.__progress_dict_writer = ProgressDictWriter()

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
        ### Stores a value in the logger.

        This may be added to a queue, a list or stored as
        a TensorBoard histogram depending on the
        type of the indicator.
        """

        self.__store.store(*args, **kwargs)

    def set_global_step(self, global_step):
        if self.__loop is None:
            raise RuntimeError("Cannot set global step from outside the loop")
        self.__loop.global_step = global_step

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

        if self.__loop is None:
            raise RuntimeError("Cannot write stats without loop")

        global_step = self.__loop.global_step

        for w in self.__writers:
            self.__store.write(w, global_step)
        self.__indicators_print = self.__store.write(self.__screen_writer, global_step)
        self.__progress_dict = self.__store.write(self.__progress_dict_writer, global_step)
        self.__store.clear()
        self.__log_line()

    def section(self, name, *,
                is_silent: bool = False,
                is_timed: bool = True,
                is_partial: bool = False,
                total_steps: float = 1.0):

        if len(self.__sections) != 0:
            raise RuntimeError("There can only be one section running at a time.")

        if self.__loop is not None:
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
                                                   is_looping=False))

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

    def loop(self, iterator: range):
        if len(self.__sections) != 0:
            raise RuntimeError("Cannot start a loop within a section")

        self.__loop = Loop(iterator=iterator, logger=self)
        return self.__loop

    def finish_loop(self):
        if len(self.__sections) != 0:
            raise RuntimeError("Cannot be within a section when finishing the loop")
        self.__loop = None

    def section_enter(self, section):
        if len(self.__sections) == 0:
            raise RuntimeError("Entering a section without creating a section.\n"
                               "Always use logger.section to create a section")

        if section is not self.__sections[-1]:
            raise RuntimeError("Entering a section other than the one last_created\n"
                               "Always user with logger.section(...):")

        self.__log_line()

    def __log_line(self):
        if self.__loop is not None:
            self.__log_looping_line()
            return

        if len(self.__sections) == 0:
            return

        self.log_color(self.__sections[-1].log(), new_line=False)

    def __log_looping_line(self):
        parts = self.__loop.log_global_step()
        parts += self.__loop.log_sections()
        parts += self.__indicators_print
        parts += self.__loop.log_progress()

        self.log_color(parts, new_line=False)

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
