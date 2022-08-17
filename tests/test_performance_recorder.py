from unittest.mock import MagicMock

from seqal.performance_recorder import PerformanceRecorder


class TestPerformanceRecorder:
    """Test PerformanceRecorder class"""

    def test_get_result(self) -> None:
        """Test PerformanceRecorder.get_result"""
        # Arrange
        data = 20
        performance_recorder = PerformanceRecorder()
        result = MagicMock(
            log_line="0.77\t0.28\t0.41\t0.26",
            classification_report={
                "micro avg": {"f1-score": 0.41},
                "macro avg": {"f1-score": 0.33},
                "weighted avg": {"f1-score": 0.23},
                "samples avg": {"f1-score": 0.13},
                "PER": {
                    "precision": 0.82,
                    "recall": 0.51,
                    "f1-score": 0.63,
                    "support": 1617,
                },
            },
        )

        # Act
        performance_recorder.get_result(data, result)
        iteration_performance = performance_recorder.performance_list[0]

        # Assert
        assert iteration_performance.data == 20
        assert iteration_performance.precision == 0.77
        assert iteration_performance.recall == 0.28
        assert iteration_performance.accuracy == 0.26
        assert iteration_performance.micro_f1 == 0.41
        assert iteration_performance.macro_f1 == 0.33
        assert iteration_performance.weighted_f1 == 0.23
        assert iteration_performance.samples_f1 == 0.13
        assert iteration_performance.label_scores["PER"]["f1-score"] == 0.63
