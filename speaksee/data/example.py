
class Example(object):
    """Defines a single training or test example.
    Stores each column of the example as an attribute.
    """

    @classmethod
    def fromdict(cls, data):
        ex = cls()
        for key, val in data.items():
            setattr(ex, key, val)

        return ex
