from abc import ABC


class StateModule:
    def __init__(self):
        pass

    # def __call__(self):
    #     raise NotImplementedError

    def create_state(self) -> any:
        raise NotImplementedError

    def set_state(self, data: any):
        raise NotImplementedError

    def on_epoch_start(self):
        raise NotImplementedError

    def on_epoch_end(self):
        raise NotImplementedError


class Metric(StateModule, ABC):
    def track(self):
        pass
