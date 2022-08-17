import pytest

from seqal.stoppers import BudgetStopper, MetricStopper


class TestMetricStopper:
    """Test MetricStopper class"""

    @pytest.mark.parametrize(
        "micro_score,expected",
        [
            (16, True),
            (14, False),
        ],
    )
    def test_stop(
        self,
        micro_score: int,
        expected: bool,
    ) -> None:
        """Test F1Stopper.stop function"""
        # Arrange
        stopper = MetricStopper(goal=15)

        # Act
        decision = stopper.stop(micro_score)

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
