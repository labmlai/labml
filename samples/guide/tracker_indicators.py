import math
import time

from lab import logger, tracker


def loop():
    logger.inspect(a=2, b=1)

    tracker.set_queue("reward", 10, True)
    for i in range(100):
        tracker.save(i, loss=100 / (i + 1), reward=math.pow(2, (i + 1)))
        if (i + 1) % 2 == 0:
            tracker.save(valid=i ** 10)
            logger.log()

        time.sleep(0.3)


if __name__ == '__main__':
    loop()
