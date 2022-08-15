import logging
from logging import StreamHandler
from logging.handlers import RotatingFileHandler

from labml_app.settings import IS_LOCAL_SETUP

_LOG_PATH = 'labml_app.log'
_MAX_BYTES = 100000


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    blue = "\x1b[36;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def _init_streaming_handler():
    streaming = StreamHandler()
    streaming.setFormatter(CustomFormatter())

    return streaming


def _init_file_handler():
    file_handler = RotatingFileHandler(filename=_LOG_PATH, maxBytes=_MAX_BYTES)
    file_handler.setFormatter(CustomFormatter())

    return file_handler


logger = logging.getLogger('LabML logger')
logger.setLevel(logging.INFO)

logger.addHandler(_init_streaming_handler())
if not IS_LOCAL_SETUP:
    logger.addHandler(_init_file_handler())
