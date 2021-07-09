from typing import List

from labml.internal.logger.destinations import Destination
from labml.internal.util import is_ipynb, is_ipynb_pycharm


def create_destination() -> List[Destination]:
    from labml.internal.logger.destinations.console import ConsoleDestination

    if is_ipynb():
        if is_ipynb_pycharm():
            from labml.internal.logger.destinations.ipynb_pycharm import IpynbPyCharmDestination
            return [IpynbPyCharmDestination(), ConsoleDestination(False)]
        else:
            from labml.internal.logger.destinations.ipynb import IpynbDestination
            return [IpynbDestination(), ConsoleDestination(False)]
    else:
        return [ConsoleDestination(True)]
