from collections import namedtuple


def to_namedtuple(dictionary):
    return namedtuple("PandasRow", dictionary.keys())(**dictionary)
