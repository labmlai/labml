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
                tracker.add('loss1', i * 10 + j)
                tracker.add('loss2', i * 10 + j)
                tracker.add('loss3', i * 10 + j)
                tracker.add('loss4', i * 10 + j)
                tracker.add('loss5', i * 10 + j)
                tracker.add('loss6', i * 10 + j)
                tracker.add('loss7', i * 10 + j)
                tracker.add('loss8', i * 10 + j)
                tracker.add('loss9', i * 10 + j)
                tracker.add('loss10', i * 10 + j)
                tracker.add('loss11', i * 10 + j)
                tracker.add('loss12', i * 10 + j)
                tracker.add('loss13', i * 10 + j)
                tracker.add('loss14', i * 10 + j)
                tracker.add('loss15', i * 10 + j)
                tracker.add('loss16', i * 10 + j)
                tracker.add('loss17', i * 10 + j)
                tracker.add('loss18', i * 10 + j)
            print(f'sending data rank {rank}')
            tracker.save()


def main():
    for i in range(WORLD_SIZE):
        worker(i)


if __name__ == '__main__':
    main()