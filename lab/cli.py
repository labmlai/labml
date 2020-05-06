import argparse

from lab import logger
from lab.logger import Text


def _open_dashboard():
    try:
        import lab_dashboard
    except (ImportError, ModuleNotFoundError):
        logger.log("Cannot import ", ('lab_dashboard', Text.highlight), '.')
        logger.log('Install with ',
                   ('pip install machine_learning_lab_dashboard', Text.value))
        return

    lab_dashboard.start_server()


def main():
    parser = argparse.ArgumentParser(description='Lab CLI')
    parser.add_argument('command', choices=['dashboard'])

    args = parser.parse_args()

    if args.command == 'dashboard':
        _open_dashboard()
    else:
        assert False
