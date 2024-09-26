from labml import logger

if __name__ == '__main__':
    arr = [
        {'v': 1, 's': 'text1'},
        {'v': 1, 's': 'text1'},
    ]

    logger.inspect(arr, _n=-1, _expand=True)
