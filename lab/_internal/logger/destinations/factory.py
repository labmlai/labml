from lab._internal.logger.destinations import Destination
from lab._internal.util import is_ipynb


def create_destination() -> Destination:
    if is_ipynb():
        from lab._internal.logger.destinations.ipynb import IpynbDestination
        return IpynbDestination()
    else:
        from lab._internal.logger.destinations.console import ConsoleDestination
        return ConsoleDestination()
