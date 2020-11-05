from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Tracker


class Namespace:
    def __init__(self, *,
                 tracker: 'Trakcer',
                 name: str):
        self.tracker = tracker
        self.name = name

    def __enter__(self):
        self.tracker.namespace_enter(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker.namespace_exit(self)
