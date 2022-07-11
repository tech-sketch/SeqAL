from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
from flair.training_utils import Result


@dataclass
class IterationPerformance:
    """Performance in each iteration"""

    data: float
    precision: float
    recall: float
    accuracy: float
    micro_f1: float
    macro_f1: float


class PerformanceRecorder:
    """Performance of all iterations"""

    def __init__(self) -> None:
        self.performance_list: List[IterationPerformance] = []

    def get_result(self, data: int, result: Result) -> None:
        """Get performance result

        Args:
            data (int): How many data used. This number could be a percentage or real count number.
            result (Result): The performance result of evaluation.
        """
        precision, recall, _, accuracy = [
            float(score) for score in result.log_line.split("\t")
        ]
        micro_f1 = result.classification_report["micro avg"]["f1-score"]
        macro_f1 = result.classification_report["macro avg"]["f1-score"]
        iteration_performance = IterationPerformance(
            data=data,
            precision=precision,
            recall=recall,
            accuracy=accuracy,
            micro_f1=micro_f1,
            macro_f1=macro_f1,
        )
        self.performance_list.append(iteration_performance)

    def save(self, file_path: str) -> None:
        """Save performance to file.

        Args:
            file_path (str): Path to save file.
        """
        with open(file_path, 'w', encoding="utf-8") as file:
            for performance in self.performance_list:
                line = (
                    f"{performance.data},{performance.precision},{performance.recall},{performance.f1},{performance.accuracy},{performance.micro_f1},{performance.macro_f1}"
                )
                file.write(line)
            file.write("\n")

    def plot(
        self,
        metric: str = "micro_f1",
        sampling_method: str = "sampling_method",
        save_path: str = "",
    ) -> None:
        """Draw performance graph

        Args:
            metric (str, optional): The metric to plot. Defaults to "f1".
                                    Available options: "precision", "recall", "f1", "accuracy", "micro_f1", "macro_f1"
            sampling_method (str, optional): The sampling method name. Defaults to "sampling_method".
            save_img (bool, optional): Save performance to image. Defaults to True.
        """
        data = [
            iteration_performance.data
            for iteration_performance in self.performance_list
        ]
        scores = [
            getattr(iteration_performance, metric)
            for iteration_performance in self.performance_list
        ]

        plt.plot(data, scores, label=sampling_method)

        # plt.xlim(0, 50)
        plt.ylim(0, 100)

        plt.title("Performance on Dataset")
        plt.xlabel("Used data")
        plt.ylabel(metric)
        plt.legend(loc="lower right")
        plt.grid()
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()
