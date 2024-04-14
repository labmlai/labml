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
    series_pref = [1 if 'loss' == s.lower().split('.')[0] else -1 for s in series_names]
    if all(p == -1 for p in series_pref) and len(series_pref) > 0:
        series_pref[0] = 1
    return series_pref


def fill_preferences(series_names: List[str], preferences: List[int]) -> List[int]:
    if len(series_names) > len(preferences):
        preferences += [-1] * (len(series_names) - len(preferences))
    else:
        preferences = preferences[:len(series_names)]

    return preferences


def update_series_preferences(preferences: List[int], series: List[int], client_series_names: List[str])\
        -> List[int]:
    preference_dict = {s: p for s, p in zip(series, preferences)}

    new_preferences = []
    for s in client_series_names:
        if s in preference_dict:
            new_preferences.append(preference_dict[s])
        else:
            new_preferences.append(-1)

    return new_preferences
