from unittest.mock import MagicMock

import pytest

from seqal.stoppers import BudgetStopper, F1Stopper


class TestF1Stopper:
    """Test F1Stopper class"""

    @pytest.mark.parametrize(
        "micro,micro_score,macro_score,expected",
        [
            (True, 16, 0, True),
            (True, 14, 0, False),
            (False, 0, 16, True),
            (False, 0, 14, False),
        ],
    )
    def test_stop(
        self,
        micro: bool,
        micro_score: int,
        macro_score: int,
        expected: bool,
    ) -> None:
        """Test F1Stopper.stop function"""
        # Arrange
        stopper = F1Stopper(goal=15)
        classification_report = {
            "micro avg": {"f1-score": micro_score},
            "macro avg": {"f1-score": macro_score},
        }
        result = MagicMock(classification_report=classification_report)

        # Act
        decision = stopper.stop(result, micro=micro)

        # Assert
        assert decision == expected


class TestBudgetStopper:
    """Test BudgetStopper class"""

    @pytest.mark.parametrize("unit_count,expected", [(10, False), (20, True)])
    def test_stop(self, unit_count: int, expected: bool) -> None:
        """Test BudgetStopper.stop function"""
        # Arrange
        stopper = BudgetStopper(goal=15, unit_price=1)

        # Act
        decision = stopper.stop(unit_count)

        # Assert
        assert decision == expected
