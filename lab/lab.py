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

    def get_experiments(self) -> List[Path]:
        """
        Get list of experiments
        """
        experiments_path = Path(self.experiments)
        return [child for child in experiments_path.iterdir()]
