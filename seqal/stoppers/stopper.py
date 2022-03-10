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


class F1Stopper(BaseStopper):
    """F1 score stopper class

    Args:
        BaseStopper (_type_): Base class of stopper
    """

    def __init__(self, goal: float) -> None:
        self.goal = goal

    def stop(self, result: dict, micro: bool = True, macro: bool = False) -> bool:
        """Stop active learning cycle if result meet the goal

        Args:
            result (dict): Evaluation result
            micro (bool, optional): Compare with f1-micro. Defaults to True.
            macro (bool, optional): Compare with f1-macro. Defaults to False.

        Returns:
            bool: True or False.
        """
        score_type = "macro avg"
        if micro:
            score_type = "micro avg"
        score = result.classification_report[score_type]["f1-score"]
        if score >= self.goal:
            return True
        else:
            return False
