from lab import logger


def loop():
    logger.info(a=2, b=1)

    logger.add_indicator('loss')
    for i in range(10):
        logger.store(loss=100 / (i + 1))
        logger.write()
        if (i + 1) % 2 == 0:
            logger.new_line()


if __name__ == '__main__':
    loop()
