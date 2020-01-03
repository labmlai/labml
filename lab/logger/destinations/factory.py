from lab.logger.destinations import Destination
from lab.logger.destinations.console import ConsoleDestination
from lab.util import is_ipynb


def create_destination() -> Destination:
    if is_ipynb():
        return IpynbDestination()
    else:
        return ConsoleDestination()