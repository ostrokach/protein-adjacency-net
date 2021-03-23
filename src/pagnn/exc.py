"""Exceptions."""


class SequenceTooShortError(Exception):
    pass


class SequenceTooLongError(Exception):
    """This exception is raised when the sequence is too long to find a negative sequence."""
    pass


class MaxNumberOfTriesExceededError(Exception):
    pass
