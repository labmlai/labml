import json
from typing import Callable, Any

from labml import lab


def cache(name: str, loader: Callable[[], Any], file_type: str = 'json') -> Any:
    cache_path = lab.get_data_path() / 'cache'
    if not cache_path.exists():
        cache_path.mkdir(parents=True)
    path = cache_path / f'{name}.{file_type}'
    if path.exists():
        with open(str(path), 'r') as f:
            if file_type == 'json':
                return json.load(f)
            else:
                raise ValueError(f'Unknown file type: {file_type}')
    else:
        value = loader()
        with open(str(path), 'w') as f:
            if file_type == 'json':
                json.dump(value, f)
            else:
                raise ValueError(f'Unknown file type: {file_type}')

        return value
