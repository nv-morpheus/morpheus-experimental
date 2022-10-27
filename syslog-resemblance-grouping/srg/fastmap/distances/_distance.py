import abc

class InputError(Exception):
    """
    Exception raised for errors in the input.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class Distance(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def get_name(self):
        pass

    @abc.abstractmethod
    def calculate(self, x, y) -> float:
        pass
    