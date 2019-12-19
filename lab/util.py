import pathlib
import random
import string

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
