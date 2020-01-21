from lab.logger.destinations import Destination
from lab.util import is_ipynb


def create_destination() -> Destination:
    if is_ipynb():
        from lab.logger.destinations.ipynb import IpynbDestination
        return IpynbDestination()
    else:
        from lab.logger.destinations.console import ConsoleDestination
        return ConsoleDestination()