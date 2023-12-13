import uuid
from labml import experiment, monit, tracker

WORLD_SIZE = 3
UUID = uuid.uuid4().hex

print(UUID)


def worker(rank: int):
    experiment.create(uuid=UUID,
                      name='Distributed Training Simulator',
                      distributed_rank=rank,
                      distributed_world_size=WORLD_SIZE,
                      )
    print(f'created experiment for rank:{rank}')
    with experiment.start():
        for i in monit.loop(500):
            for j in range(10):
                tracker.add('loss', i * 10 + j)
            print(f'sending data rank {rank}')
            tracker.save()


def main():
    for i in range(WORLD_SIZE):
        worker(i)


if __name__ == '__main__':
    main()