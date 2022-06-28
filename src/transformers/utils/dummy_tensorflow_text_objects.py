# This file is autogenerated by the command `make fix-copies`, do not edit.
# flake8: noqa
from ..utils import DummyObject, requires_backends


class TFBertTokenizer(metaclass=DummyObject):
    _backends = ["tensorflow_text"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tensorflow_text"])
