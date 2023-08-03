from deepx.utils.wrappers import watch_kwargs


@watch_kwargs
def func(a, b, c=5):
    print(a, b, c)


class Class:
    @watch_kwargs
    def __init__(self, a, b, c=5):
        print(a, b, c)


def test_watch_kwargs():
    func(a=1, b=2, c=3, d=4, x="test")
    Class(a=1, b=2, c=3, d=4, x="test")
