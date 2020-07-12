import importlib
from pathlib import PurePath
from typing import Dict
from labml.internal import experiment as _experiment


def new_run(python_file: PurePath, configs: Dict[str, any]):
    from labml import lab
    lab_path = lab.get_path()
    module_path = python_file.relative_to(lab_path)
    module_path = str(module_path).replace('/', '.').replace('.py', '')
    _experiment.global_params_singleton().configs = configs
    experiment_module = importlib.import_module(module_path)

    experiment_module.main()
