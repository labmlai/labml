from labml.logger import inspect


def main():
    inspect([f'{i:05}' for i in range(50)])
    inspect(x=[f'{i:05}' for i in range(50)])


if __name__ == '__main__':
    main()
