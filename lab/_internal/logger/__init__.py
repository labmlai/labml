from typing import Optional

import numpy as np

from .artifacts import Artifact
from .colors import StyleCode
from .indicators import Indicator
from .internal import LoggerInternal as _LoggerInternal

_internal: Optional[_LoggerInternal] = None


def internal() -> _LoggerInternal:
    global _internal
    if _internal is None:
        _internal = _LoggerInternal()

    return _internal


def save_numpy(name: str, array: np.ndarray):
    """
    Save a single numpy array.
    This is used to save processed data
    """
    internal().save_numpy(name, array)
