import time

from lab import logger


def loop():
    logger.info(a=2, b=1)

    logger.add_indicator('loss_ma', queue_limit=2)
    for i in range(10):
        logger.add_global_step(1)
        logger.store(loss=100 / (i + 1), loss_ma=100 / (i + 1))
        logger.write()
        if (i + 1) % 2 == 0:
            logger.new_line()

        time.sleep(2)


if __name__ == '__main__':
    loop()
