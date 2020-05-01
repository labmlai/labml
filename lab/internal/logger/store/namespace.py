from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Store


class Namespace:
    def __init__(self, *,
                 store: 'Store',
                 name: str):
        self._store = store
        self.name = name

    def __enter__(self):
        self._store.namespace_enter(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._store.namespace_exit(self)
