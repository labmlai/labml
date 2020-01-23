import math
import time

from lab import logger
from lab.logger.indicators import Queue


def loop():
    logger.info(a=2, b=1)

    logger.add_indicator(Queue("reward", 10, True))
    for i in range(100):
        logger.add_global_step(1)
        logger.store(loss=100 / (i + 1), reward=math.pow(2, (i + 1)))
        logger.write()
        if (i + 1) % 2 == 0:
            logger.new_line()

        time.sleep(0.3)


if __name__ == '__main__':
    loop()
