import importlib
import multiprocessing as mp
import multiprocessing.connection
from pathlib import PurePath
from typing import Dict, Optional

from labml import lab
from labml.internal import experiment as _experiment


def new_run(python_file: PurePath, *,
            configs: Optional[Dict[str, any]] = None,
            comment: Optional[str] = None,
            lab_path: Optional[PurePath] = None):
    if lab_path is None:
        lab_path = lab.get_path()
    module_path = python_file.relative_to(lab_path)
    module_path = str(module_path).replace('/', '.').replace('.py', '')
    _experiment.global_params_singleton().configs = configs
    _experiment.global_params_singleton().comment = comment
    experiment_module = importlib.import_module(module_path)

    experiment_module.main()


class Process:
    _child: mp.connection.Connection
    _process: mp.Process

    def __init__(self, python_file: PurePath, *,
                 configs: Optional[Dict[str, any]],
                 comment: Optional[str],
                 lab_path: PurePath):
        self._child, parent = mp.Pipe()
        self._process = mp.Process(target=_run_process,
                                   args=(parent, python_file,
                                         configs, comment,
                                         lab_path))

    def __call__(self):
        self._process.start()
        ready = self._child.recv()
        assert ready == 'ready'
        print(self._process.pid)

        self._child.send("start")


def _run_process(remote: multiprocessing.connection.Connection,
                 python_file: PurePath,
                 configs: Optional[Dict[str, any]],
                 comment: Optional[str],
                 lab_path: PurePath):
    module_path = python_file.relative_to(lab_path)
    module_path = str(module_path).replace('/', '.').replace('.py', '')
    _experiment.global_params_singleton().configs = configs
    _experiment.global_params_singleton().comment = comment
    experiment_module = importlib.import_module(module_path)
    remote.send("ready")
    # sys.stdout = None

    while True:
        try:
            cmd = remote.recv()
            if cmd == "start":
                experiment_module.main()
            else:
                raise NotImplementedError
        except KeyboardInterrupt as e:
            print("Interrupted")
            print(e)
            break


def new_run_process(python_file: PurePath, *,
                    configs: Optional[Dict[str, any]] = None,
                    comment: Optional[str] = None):
    p = Process(python_file,
                configs=configs,
                comment=comment,
                lab_path=lab.get_path())
    p()
