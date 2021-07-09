import os
import sys
import time

from labml import logger
from labml.logger import Text


def main():
    logger.log(f'{os.getpid()}')
    for i in range(60 * 10):
        logger.log([
            ("Hello", Text.success),
            ", ",
            ("World", Text.warning),
            ": ",
            (str(i), Text.danger)
        ])
        time.sleep(1)

    logger.log(str(sys.argv), Text.title)


if __name__ == '__main__':
    main()
