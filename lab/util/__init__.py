import pathlib
import random
import string
from typing import Dict, Set, List

import yaml


def yaml_load(s: str):
    return yaml.load(s, Loader=yaml.FullLoader)


def yaml_dump(obj: any):
    return yaml.dump(obj, default_flow_style=False)


def rm_tree(path_to_remove: pathlib.Path):
    if path_to_remove.is_dir():
        for f in path_to_remove.iterdir():
            if f.is_dir():
                rm_tree(f)
            else:
                f.unlink()
        path_to_remove.rmdir()
    else:
        path_to_remove.unlink()


def random_string(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


def is_ipynb():
    try:
        cfg = get_ipython().config
        if cfg['IPKernelApp'] is None:
            return False

        app: Dict = cfg['IPKernelApp']
        return len(app.keys()) > 0
    except NameError:
        return False


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
        return str(module_path)

    return ''


if __name__ == '__main__':
    print(is_ipynb())
