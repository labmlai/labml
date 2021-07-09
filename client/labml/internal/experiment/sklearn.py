import pathlib
from typing import Dict
from joblib import dump, load

from . import ModelSaver, experiment_singleton


# This is under construction

class SKLearnModelSaver(ModelSaver):
    def __init__(self, name: str, model: any):
        self.name = name
        self.model = model

    def save(self, checkpoint_path: pathlib.Path) -> any:
        file_name = f"{self.name}.pth"
        dump(self.model, str(checkpoint_path / file_name))
        return file_name

    def loadl(self, checkpoint_path: pathlib.Path, info: any):
        file_name: str = info

        return load(str(checkpoint_path / file_name))


def add_models(models: Dict[str, any]):
    exp = experiment_singleton()
    savers = {name: SKLearnModelSaver(name, model) for name, model in models.items()}

    exp.checkpoint_saver.add_savers(savers)
