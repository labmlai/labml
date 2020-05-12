from labml.internal.logger.destinations import Destination
from labml.internal.util import is_ipynb


def create_destination() -> Destination:
    if is_ipynb():
        from labml.internal.logger.destinations.ipynb import IpynbDestination
        return IpynbDestination()
    else:
        from labml.internal.logger.destinations.console import ConsoleDestination
        return ConsoleDestination()
