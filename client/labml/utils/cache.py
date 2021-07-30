import functools
import json
import pickle
from pathlib import Path
from typing import Callable, Any, Optional

from labml import lab


def _cache_load(path: Path, file_type: str):
    if file_type == 'json':
        with open(str(path), 'r') as f:
            return json.load(f)
    elif file_type == 'pickle':
        with open(str(path), 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f'Unknown file type: {file_type}')


def _cache_save(path: Path, value: Any, file_type: str):
    if file_type == 'json':
        with open(str(path), 'w') as f:
            json.dump(value, f)
    elif file_type == 'pickle':
        with open(str(path), 'wb') as f:
            pickle.dump(value, f)
    else:
        raise ValueError(f'Unknown file type: {file_type}')


def _get_cache_path(name: str, file_type: str):
    cache_path = lab.get_data_path() / 'cache'
    if not cache_path.exists():
        cache_path.mkdir(parents=True)
    return cache_path / f'{name}.{file_type}'


def _cache_wrap(name: str, loader: Callable[[], Any], *, file_type: str) -> Any:
    path = _get_cache_path(name, file_type)
    if path.exists():
        return _cache_load(path, file_type)
    else:
        value = loader()
        _cache_save(path, value, file_type)

        return value


def cache(name: str, loader: Optional[Callable[[], Any]] = None, *, file_type: str = 'json') -> Any:
    """
    This caches results of a function. Can be used as a decorator or you can pass a lambda function
    to it that takes no arguments.

    *It doesn't cache by arguments.*

    Arguments:
        name (str): name of the cache
        loader (Callable[[], Any], optional): the function that generates the data to be cached

    Keyword Arguments:
        file_type (str, optional): The file type to store the data. Defaults to ``json``.
    """
    if loader is not None:
        return _cache_wrap(name, loader, file_type=file_type)

    def decorator_func(f: Callable):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            path = _get_cache_path(name, file_type)
            if path.exists():
                return _cache_load(path, file_type)
            else:
                value = f(*args, **kwargs)
                _cache_save(path, value, file_type)

                return value

        return wrapper

    return decorator_func


def cache_get(name: str, file_type: str = 'json') -> Any:
    """
    Get cached data.

    Arguments:
        name (str): name of the cache
        file_type (str, optional): The file type to store the data. Defaults to ``json``.
    """
    path = _get_cache_path(name, file_type)
    if path.exists():
        return _cache_load(path, file_type)
    else:
        return None


def cache_set(name: str, value: Any, file_type: str = 'json') -> Any:
    """
    Save data in cache.

    Arguments:
        name (str): name of the cache
        value (any): data to be cached
        file_type (str, optional): The file type to store the data. Defaults to ``json``.
    """
    path = _get_cache_path(name, file_type)
    _cache_save(path, value, file_type)
