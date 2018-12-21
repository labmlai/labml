import io
import pathlib

import numpy as np
import yaml
from matplotlib import pyplot

import functools
import inspect
import warnings


def yaml_load(s: str):
    return yaml.load(s)


def yaml_dump(obj: any):
    return yaml.dump(obj, default_flow_style=False)


def overlay_image_green(result: np.ndarray,
                        base: np.ndarray,
                        overlay: np.ndarray,
                        base_factor: float):
    """
    #### Overlays a map on an image
    """
    result[:, :, 0] = base * base_factor
    result[:, :, 1] = base * base_factor
    result[:, :, 2] = base * base_factor
    result[:, :, 1] += (1 - base_factor) * 255 * overlay / (np.max(overlay) - np.min(overlay))


def create_png(frame: np.ndarray):
    """
    #### Create a PNG from a numpy array.
    """
    png = io.BytesIO()
    pyplot.imsave(png, frame, format='png', cmap='gray')
    return png


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


def deprecated(message: str):
    """
    Mark a class, a function or a class method as deprecated.
    """

    def decorator(deprecated_obj):
        if inspect.isclass(deprecated_obj):
            warning_msg = f"Deprecated class [{deprecated_obj.__name__}]: {message}"
        else:
            warning_msg = f"Deprecated function [{deprecated_obj.__name__}]: {message}"

        warned = dict(value=False)

        @functools.wraps(deprecated_obj)
        def new_func(*args, **kwargs):
            if not warned['value']:
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    warning_msg,
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                warned['value'] = True

            return deprecated_obj(*args, **kwargs)

        return new_func

    return decorator
