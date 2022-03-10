from unittest.mock import MagicMock

import pytest

from seqal.stoppers import BudgetStopper, F1Stopper


class TestF1Stopper:
    """Test F1Stopper class"""

    @pytest.mark.parametrize(
        "micro,micro_score,macro,macro_score,expected",
        [
            (True, 16, False, 0, True),
            (True, 14, False, 0, False),
            (False, 0, True, 16, True),
            (False, 0, True, 14, False),
        ],
    )
    def test_stop(
        self,
        micro: bool,
        micro_score: int,
        macro: bool,
        macro_score: int,
        expected: bool,
    ) -> None:
        """Test stop function"""
        # Arrange
        stopper = F1Stopper(goal=15)
        classification_report = {
            "micro avg": {"f1-score": micro_score},
            "macro avg": {"f1-score": macro_score},
        }
        result = MagicMock(classification_report=classification_report)

        # Act
        decision = stopper.stop(result, micro=micro, macro=macro)

        # Assert
        assert decision == expected


class TestBudgetStopper:
    """Test BudgetStopper class"""

    @pytest.mark.parametrize("unit_count,expected", [(10, False), (20, True)])
    def test_stop(self, unit_count: int, expected: bool) -> None:
        """Test stop function"""
        # Arrange
        stopper = BudgetStopper(goal=15, unit_price=1)

        # Act
        decision = stopper.stop(unit_count)

        # Assert
        assert decision == expected
