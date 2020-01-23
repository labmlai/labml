import time

from lab import logger


def simple_section():
    with logger.section("Simple section"):
        # code to load data
        time.sleep(2)


def unsuccessful_section():
    with logger.section("Unsuccessful section"):
        time.sleep(1)
        logger.set_successful(False)


def progress():
    with logger.section("Progress", total_steps=100):
        for i in range(100):
            time.sleep(0.1)
            # Multiple training steps in the inner loop
            logger.progress(i)


if __name__ == '__main__':
    simple_section()
    unsuccessful_section()
    progress()
