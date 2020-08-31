import pathlib
import random
import string
from typing import Dict, Callable

import yaml

get_ipython: Callable


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
        if len(app.keys()) > 0:
            return True
        else:
            return False
    except NameError:
        return False


def is_ipynb_pycharm():
    if not is_ipynb():
        return False

    if is_colab() or is_kaggle():
        return False

    import os
    if '_' not in os.environ:
        return True
    else:
        return False


def is_colab():
    import sys
    return 'google.colab' in sys.modules


def is_kaggle():
    import sys
    return 'kaggle_gcp' in sys.modules


if __name__ == '__main__':
    print(is_ipynb())
