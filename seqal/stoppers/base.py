from abc import ABC


class BaseStopper(ABC):
    """BaseStopper class

    This is a base class to inherit for different method to stop the active learning cycle.
    Every stop method class should inherit this class.
    """

    def stop(self) -> bool:
        """Determine stop the cycle or not"""
        raise NotImplementedError
