import uuid
import time
import threading

from labml import experiment, monit, tracker

WORLD_SIZE = 2
UUID = uuid.uuid4().hex

print(UUID)


def worker(rank: int):
    experiment.create(uuid=UUID,
                      name='Distributed Training Simulator',
                      distributed_rank=rank,
                      distributed_world_size=WORLD_SIZE,
                      writers={'screen', 'labml'}
                      )
    print(f'created experiment for rank:{rank}')
    with experiment.start():
        for i in monit.loop(250):
            for j in range(10):
                tracker.add('loss', i * 10 + j)
                time.sleep(0.25)
            print(f'sending data rank {rank}')
            tracker.save()


def main():
    threads = list()
    for i in range(WORLD_SIZE):
        x = threading.Thread(target=worker, args=(i,))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        thread.join()

    print("Completed")


if __name__ == '__main__':
    main()