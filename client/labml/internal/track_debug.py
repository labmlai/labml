from typing import Optional


class TrackerDebug:
    def __init__(self):
        self.is_debug = False
        self.values = {}

    def store(self, name: str, value: any):
        if not self.is_debug:
            return

        if name not in self.values:
            self.values[name] = []

        self.values[name].append(value)

    def get(self, name: str):
        return self.values.get(name, None)


_internal: Optional[TrackerDebug] = None


def tracker_debug_singleton() -> TrackerDebug:
    global _internal
    if _internal is None:
        _internal = TrackerDebug()

    return _internal
