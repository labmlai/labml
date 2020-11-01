from labml.internal.logger.destinations import Destination
from labml.internal.util import is_ipynb, is_ipynb_pycharm


def create_destination() -> Destination:
    if is_ipynb():
        if is_ipynb_pycharm():
            from labml.internal.logger.destinations.ipynb_pycharm import IpynbPyCharmDestination
            return IpynbPyCharmDestination()
        else:
            from labml.internal.logger.destinations.ipynb import IpynbDestination
            return IpynbDestination()
    else:
        from labml.internal.logger.destinations.console import ConsoleDestination
        return ConsoleDestination()
