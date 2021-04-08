import json
import pickle
from pathlib import Path
from typing import Callable, Any

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


def cache(name: str, loader: Callable[[], Any], file_type: str = 'json') -> Any:
    path = _get_cache_path(name, file_type)
    if path.exists():
        return _cache_load(path, file_type)
    else:
        value = loader()
        _cache_save(path, value, file_type)

        return value


def cache_get(name: str, file_type: str = 'json') -> Any:
    path = _get_cache_path(name, file_type)
    if path.exists():
        return _cache_load(path, file_type)
    else:
        return None


def cache_set(name: str, value: Any, file_type: str = 'json') -> Any:
    path = _get_cache_path(name, file_type)
    _cache_save(path, value, file_type)
