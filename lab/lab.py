from pathlib import PurePath


class Lab:
    """
    ### Lab

    Lab contains the lab specific properties.
    """
    def __init__(self, *, path: PurePath):
        self.path = path

    @property
    def experiments(self) -> PurePath:
        return self.path / "logs"
