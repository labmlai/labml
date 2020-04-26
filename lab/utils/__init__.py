import os
import pathlib
from typing import Set, List


def get_caller_file(ignore_callers: Set[str] = None):
    if ignore_callers is None:
        ignore_callers = {}

    import inspect

    frames: List[inspect.FrameInfo] = inspect.stack()
    lab_src = pathlib.PurePath(__file__).parent.parent

    for f in frames:
        module_path = pathlib.PurePath(f.filename)
        if str(module_path).startswith(str(lab_src)):
            continue
        if str(module_path) in ignore_callers:
            continue
        if str(module_path).startswith('<ipython'):
            break
        return str(module_path)

    return os.path.abspath('')