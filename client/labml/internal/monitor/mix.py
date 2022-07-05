from typing import Tuple, Union, Callable, Any, List, cast, Sized, TYPE_CHECKING, Iterator

Receiver = Union[str, Callable[[Any], None]]

if TYPE_CHECKING:
    from ..monitor import Monitor


class SplitIterator:
    def __init__(self, iterator: Tuple[Receiver, Sized]):
        receiver, it = iterator
        if isinstance(receiver, str):
            self.name = receiver
            self.callback = None
            self.is_iterate = True
        elif callable(receiver):
            self.name = receiver.__name__
            self.callback = receiver
            self.is_iterate = False
        else:
            raise ValueError('receiver must be str or callable', receiver)

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
    current_iterator: SplitIterator

    def __init__(self, *,
                 total_iterations,
                 iterators: List[Tuple[Receiver, Sized]],
                 is_monit: bool,
                 monitor: 'Monitor',
                 ):
        self.monitor = monitor
        self.is_monit = is_monit
        self.total_iterations = total_iterations
        self.iterators = [SplitIterator(it) for it in iterators]
        self.section = None

    def __iter__(self):
        self.it = [cast(SplitIterator, iter(it)) for it in self.iterators]
        self.idx = 0
        return self

    def get_next(self):
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
                            self.section = self.monitor.section(it.name,
                                                                is_partial=True,
                                                                is_silent=False,
                                                                is_timed=True,
                                                                total_steps=it.size,
                                                                is_children_silent=False,
                                                                is_new_line=True,
                                                                is_not_in_loop=False,
                                                                is_track=False)
                            self.section.__enter__()
                        self.current_iterator = it
                        if it.is_iterate:
                            return it.name, next(it)
                        else:
                            next(it)
                            return None

            self.idx += 1

        raise StopIteration()

    def __next__(self):
        while True:
            n = self.get_next()
            if n is not None:
                return n


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
