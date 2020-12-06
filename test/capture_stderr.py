import sys
from io import StringIO


class MyStringIO(StringIO):
    def close(self, *args, **kwargs):  # real signature unknown
        super().close()

    # def getvalue(self, *args, **kwargs):  # real signature unknown
    #     super().getvalue()

    def read(self, *args, **kwargs):  # real signature unknown
        super().read(*args, **kwargs)

    def readable(self, *args, **kwargs):  # real signature unknown
        super().readable()

    def readline(self, *args, **kwargs):  # real signature unknown
        super().readline()

    def seek(self, *args, **kwargs):  # real signature unknown
        super().seek(*args, **kwargs)

    def seekable(self, *args, **kwargs):  # real signature unknown
        super().seekable()

    def tell(self, *args, **kwargs):  # real signature unknown
        super().tell()

    def truncate(self, *args, **kwargs):  # real signature unknown
        super().truncate()

    def writable(self, *args, **kwargs):  # real signature unknown
        super().writable()

    def write(self, *args, **kwargs):  # real signature unknown
        super().write(*args, **kwargs)
        sys.stdout.write(*args, **kwargs)
        self.original.write(*args, **kwargs)

    def __init__(self, original):  # real signature unknown
        super().__init__()
        self.original = original


def test():
    _stderr = sys.stderr
    sys.stderr = _stringio = MyStringIO(_stderr)
    print("oops\noops", file=sys.stderr)
    raise RuntimeError('test')
    # print(_stringio.getvalue().splitlines())


if __name__ == '__main__':
    test()
