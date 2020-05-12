import argparse

from labml import logger
from labml.logger import Text


def _open_dashboard():
    try:
        import labml_dashboard
    except (ImportError, ModuleNotFoundError):
        logger.log("Cannot import ", ('labml_dashboard', Text.highlight), '.')
        logger.log('Install with ',
                   ('pip install labml_dashboard', Text.value))
        return

    labml_dashboard.start_server()


def main():
    parser = argparse.ArgumentParser(description='LabML CLI')
    parser.add_argument('command', choices=['dashboard'])

    args = parser.parse_args()

    if args.command == 'dashboard':
        _open_dashboard()
    else:
        assert False
