import sys

from labml import logger
from labml.logger import Text


def main():
    logger.log([
        ("Hello", Text.success),
        ", ",
        ("World", Text.warning)
    ])

    logger.log(str(sys.argv), Text.title)


if __name__ == '__main__':
    main()
