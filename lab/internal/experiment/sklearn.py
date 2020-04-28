import pathlib
from typing import Dict
from joblib import dump, load

from . import Checkpoint, experiment_singleton


class SkLearnCheckpoint(Checkpoint):
    def save_model(self,
                   name: str,
                   model: any,
                   checkpoint_path: pathlib.Path) -> any:
        file_name = f"{name}.pth"
        dump(model, str(checkpoint_path / file_name))
        return file_name

    def load_model(self,
                   name: str,
                   model: any,
                   checkpoint_path: pathlib.Path,
                   info: any):
        file_name: str = info

        return load(str(checkpoint_path / file_name))


def add_models(models: Dict[str, any]):
    exp = experiment_singleton()
    if exp.checkpoint_saver is None:
        exp.checkpoint_saver = SkLearnCheckpoint(exp.run.checkpoint_path)

    exp.checkpoint_saver.add_models(models)
