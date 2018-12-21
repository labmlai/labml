from pathlib import PurePath, Path
from typing import List


class Lab:
    """
    ### Lab

    Lab contains the lab specific properties.
    """

    def __init__(self, *, path: PurePath):
        self.path = path

    @property
    def experiments(self) -> PurePath:
        """
        ### Experiments path
        """
        return self.path / "logs"

    @property
    def experiments_path(self):
        """
        ### Experiments path
        """
        return Path(self.experiments)

    def get_experiments(self) -> List[Path]:
        """
        Get list of experiments
        """
        return [child for child in self.experiments_path.iterdir()]
