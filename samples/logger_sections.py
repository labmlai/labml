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


def loop_section():
    for step in logger.loop(range(0, 10)):
        with logger.section("Step"):
            time.sleep(0.5)
        with logger.section("Step2"):
            time.sleep(0.1)
        logger.write()


def loop_partial_section():
    for step in logger.loop(range(0, 10)):
        with logger.section("Step", is_partial=True):
            time.sleep(0.5)
            logger.progress((step % 5 + 1) / 5)
        logger.write()


def partial_section():
    for i in range(10):
        with logger.section("Partial", is_partial=True):
            time.sleep(1)


if __name__ == '__main__':
    # simple_section()
    # unsuccessful_section()
    # progress()
    # partial_section()
    # loop_section()
    loop_partial_section()