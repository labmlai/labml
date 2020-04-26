import time

from lab import loop, monit, tracker, logger


def simple_section():
    with monit.section("Simple section"):
        # code to load data
        time.sleep(2)


def unsuccessful_section():
    with monit.section("Unsuccessful section"):
        time.sleep(1)
        monit.fail()


def progress():
    with monit.section("Progress", total_steps=100):
        for i in range(100):
            time.sleep(0.1)
            # Multiple training steps in the inner loop
            monit.progress(i)


def loop_section():
    for step in loop.loop(range(0, 10)):
        with monit.section("Step"):
            time.sleep(0.5)
        with monit.section("Step2"):
            time.sleep(0.1)
        tracker.save()
    logger.log()


def loop_partial_section():
    for step in loop.loop(range(0, 10)):
        with monit.section("Step", is_partial=True):
            time.sleep(0.5)
            monit.progress((step % 5 + 1) / 5)
        tracker.save()


if __name__ == '__main__':
    simple_section()
    unsuccessful_section()
    progress()
    loop_section()
    loop_partial_section()
