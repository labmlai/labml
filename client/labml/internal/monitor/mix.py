from typing import Tuple, Union, Callable, Any, List, cast, Sized, TYPE_CHECKING, Iterator

Receiver = Union[str, Callable[[Any], None]]
MyIterator = Union[Sized, int]

if TYPE_CHECKING:
    from ..monitor import Monitor


class SplitIterator:
    def __init__(self, iterator: Tuple[Receiver, MyIterator]):
        receiver, it = iterator
        if isinstance(receiver, str):
            self.name = receiver
            self.callback = None
        else:
            self.name = receiver.__name__
            self.callback = receiver

        if isinstance(it, int):
            self.iterator = range(it)
            self.size = it
        else:
            self.iterator = it
            self.size = len(it)

        self.idx = 0

    def __iter__(self):
        self.idx = 0
        self.it = iter(cast(Iterator, self.iterator))
        return self

    def __len__(self):
        return self.size

    def __next__(self):
        self.idx += 1
        v = next(self.it)
        if self.callback:
            self.callback(v)
            return v
        else:
            return v


class Mix:
    iterators: List[SplitIterator]
    it: List[SplitIterator]

    def __init__(self, *,
                 total_iterations,
                 iterators: List[Tuple[Receiver, MyIterator]],
                 is_monit: bool,
                 logger: 'Monitor'):
        self.logger = logger
        self.is_monit = is_monit
        self.total_iterations = total_iterations
        self.iterators = [SplitIterator(it) for it in iterators]
        self.section = None

    def __iter__(self):
        self.it = [cast(SplitIterator, iter(it)) for it in self.iterators]
        self.idx = 0
        return self

    def __next__(self):
        finished = False
        if self.section is not None:
            self.section.progress(self.current_iterator.idx)
            self.section.__exit__(None, None, None)
            self.section = None
        while not finished:
            finished = True
            for it in self.it:
                if it.idx < len(it):
                    finished = False
                    if it.idx < (self.idx + 1) * len(it) // self.total_iterations:
                        if self.is_monit:
                            self.section = self.logger.section(it.name,
                                                               is_partial=True,
                                                               is_silent=False,
                                                               is_timed=True,
                                                               total_steps=it.size,
                                                               is_children_silent=False,
                                                               is_new_line=True)
                            self.section.__enter__()
                        self.current_iterator = it
                        return it.name, next(it)

            self.idx += 1

        raise StopIteration()


class Enumerate(Sized):
    def __init__(self, it: Sized):
        self.iterator = it

    def __iter__(self):
        self.idx = 0
        self.it = iter(cast(Iterator, self.iterator))
        return self

    def __next__(self):
        idx = self.idx
        self.idx += 1
        return idx, next(self.it)

    def __len__(self) -> int:
        return len(self.iterator)
