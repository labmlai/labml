import time

import numpy as np

from labml import tracker, logger


# dummy train function
def train():
    return np.random.randint(100)


def main():
    # Reset global step because we incremented in previous loop
    tracker.set_global_step(0)

    for i in range(1, 401):
        tracker.add_global_step()
        loss = train()
        tracker.add(loss=loss)
        if i % 10 == 0:
            tracker.save()
        if i % 100 == 0:
            logger.log()
        time.sleep(0.02)


if __name__ == '__main__':
    main()
