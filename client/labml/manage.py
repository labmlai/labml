from pathlib import PurePath
from typing import Dict, Optional


def new_run(python_file: PurePath, *,
            configs: Optional[Dict[str, any]] = None,
            comment: Optional[str] = None):
    from labml.internal.manage.process import new_run as _new_run
    _new_run(python_file, configs=configs, comment=comment)


def new_run_process(python_file: PurePath, *,
                    configs: Optional[Dict[str, any]] = None,
                    comment: Optional[str] = None):
    new_run_process(python_file, configs=configs, comment=comment)
