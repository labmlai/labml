from typing import Generic, TypeVar, Optional

from . import StateModule

T = TypeVar('T')


class SimpleState(Generic[T]):
    state: Optional[T]

    def __init__(self):
        self.state = None

    def get(self) -> T:
        return self.state

    def set(self, data: T):
        self.state = data

    def reset(self):
        self.state = None


class SimpleStateModule(StateModule, Generic[T]):
    data: SimpleState[T]

    def __init__(self):
        super().__init__()

    def set(self, data: T):
        self.data.set(data)

    def get(self) -> T:
        return self.data.get()

    def create_state(self):
        return SimpleState()

    def set_state(self, data: any):
        self.data = data

    def on_epoch_start(self):
        self.data.reset()

    def on_epoch_end(self):
        pass
