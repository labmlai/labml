from lab.internal.logger.destinations import Destination
from lab.internal.util import is_ipynb


def create_destination() -> Destination:
    if is_ipynb():
        from lab.internal.logger.destinations.ipynb import IpynbDestination
        return IpynbDestination()
    else:
        from lab.internal.logger.destinations.console import ConsoleDestination
        return ConsoleDestination()
