import time
from typing import Callable, List
from uuid import uuid4
from functools import wraps


def check_version(user_v, new_v) -> bool:
    for uv, nw in zip(user_v.split('.'), new_v.split('.')):
        if int(nw) == int(uv):
            continue
        elif int(nw) > int(uv):
            return True
        else:
            return False


def gen_token() -> str:
    return uuid4().hex


def time_this(function) -> Callable:
    @wraps(function)
    def time_wrapper(*args, **kwargs):
        start = time.time()
        r = function(*args, **kwargs)
        end = time.time()

        total_time = end - start
        print(function.__name__, total_time)

        return r

    return time_wrapper


def get_true_run_uuid(run_uuid: str) -> str:
    split = run_uuid.split('_')

    if len(split) > 1 and int(split[-1]) == 0:
        run_uuid = split[0]

    return run_uuid


def get_default_series_preference(series_names: List[str]) -> List[int]:
    return [1 if 'loss' in s.lower() else -1 for s in series_names]


def fill_preferences(series_names: List[str], preferences: List[int]) -> List[int]:
    if len(series_names) > len(preferences):
        preferences += [-1] * (len(series_names) - len(preferences))

    return preferences
