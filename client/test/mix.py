from labml import monit, tracker, logger


def main():
    import time

    for n, v in monit.mix(((lambda x: print('sample', x)), 2), ('valid', range(4))):
        print(n, v)

    print('----')

    for n, v in monit.mix(2, ('train', range(0, 32, 8)), ('valid', 4)):
        print(n, v)

    for _ in monit.loop(2):
        for n, v in monit.mix(5, ('train', range(50)), ('valid', range(10))):
            time.sleep(0.05)
            # print(n, v)
            tracker.save({n: v})
        tracker.new_line()


if __name__ == '__main__':
    main()
