import uuid
import time
import concurrent.futures

from labml import experiment, monit, tracker

WORLD_SIZE = 8
UUID = uuid.uuid4().hex

print(UUID)


def worker(rank: int):
    experiment.create(uuid=UUID,
                      name='Distributed Training Simulator',
                      distributed_rank=rank,
                      distributed_world_size=WORLD_SIZE,
                      writers={'screen', 'labml'}
                      )
    with experiment.start():
        for i in monit.loop(50):
            for j in range(10):
                tracker.add('loss', i * 10 + j)
                time.sleep(0.1)
            tracker.save()


def main():
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=WORLD_SIZE)

    for i in range(WORLD_SIZE):
        pool.submit(worker, i)

    # wait for all tasks to complete
    pool.shutdown(wait=True)

    print("Completed")


if __name__ == '__main__':
    main()
