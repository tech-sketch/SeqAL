from .base import BaseStopper


class BudgetStopper(BaseStopper):
    """Budget stopper class

    Args:
        BaseStopper (_type_): Base class of stopper
    """

    def __init__(self, goal: float, unit_price: float) -> None:
        self.goal = goal
        self.unit_price = unit_price

    def stop(self, unit_count: int) -> bool:
        """Stop active learning cycle if out of budget

        Args:
            unit_count (int): How many unit have been processed.

        Returns:
            bool: True or False.
        """
        if self.unit_price * unit_count >= self.goal:
            return True
        else:
            return False


class MetricStopper(BaseStopper):
    """Metric score stopper class

    Args:
        BaseStopper (_type_): Base class of stopper
    """

    def __init__(self, goal: float) -> None:
        self.goal = goal

    def stop(self, score: float) -> bool:
        """Stop active learning cycle if result meet the goal

        Args:
            score (float): Metric score.

        Returns:
            bool: True or False.
        """
        if score >= self.goal:
            return True
        else:
            return False
