"""The external editor API; helpers remain in the four responsibility modules."""


def generate(*args, **kwargs):
    from .interface import generate as run
    return run(*args, **kwargs)


__all__ = ["generate"]
