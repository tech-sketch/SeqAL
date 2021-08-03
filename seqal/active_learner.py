from typing import Callable, List, Tuple

from flair.data import Sentence
from flair.trainers import ModelTrainer
from torch.nn import Module

from seqal.datasets import Corpus


class ActiveLearner:
    """Active learning workflow class.

    Args:
        estimator: The estimator to be used in the active learning loop.
        query_strategy: Function providing the query strategy for the active learning loop,
            for instance, seqal.uncertainty.uncertainty_sampling.
        corpus: Corpus contains train(labeled data), dev, test (data pool).
        **kwargs: keyword arguments.
    Attributes:
        estimator: The estimator to be used in the active learning loop.
        query_strategy: Function providing the query strategy for the active learning loop.
        corpus: The corpus to be used in active learning loop.
    """

    def __init__(
        self, estimator: Module, query_strategy: Callable, corpus: Corpus, **kwargs
    ) -> None:
        assert callable(query_strategy), "query_strategy must be callable"
        self.clean_estimator = estimator
        self.trained_estimator = None
        self.query_strategy = query_strategy
        self.corpus = corpus
        self.kwargs = kwargs

    def fit(self, save_path: str = "resources/init_train") -> None:
        """Train model on labeled data.

        Args:
            save_path (str, optional): Log and model save path. Defaults to "resources/init_train".
        """
        # estimator = copy.deepcopy(self.clean_estimator)  # TODO: Test error
        estimator = self.clean_estimator
        trainer = ModelTrainer(estimator, self.corpus)
        trainer.train(save_path, **self.kwargs)
        self.trained_estimator = estimator

    def query(
        self, sents: List[Sentence], query_number: int
    ) -> Tuple[List[Sentence], List[Sentence]]:
        """Query data from pool (sents).

        Args:
            sents (List[Sentence]): Data pool that consist of sentences.
            query_number (int): batch query number.

        Returns:
            Tuple[List[Sentence], List[Sentence]]:
                sents: The data pool after removing query samples.
                query_samples: Query samples.
        """

        return self.query_strategy(sents, self.estimator, query_number)

    def teach(
        self, query_samples: Sentence, save_path: str = "resources/retrain"
    ) -> None:
        """Retrain model on new labeled dataset.

        Args:
            query_samples (Sentence): new labeled data.
            save_path (str, optional): Log and model save path. Defaults to "resources/retrain".
        """
        self.corpus.add_query_samples(query_samples)
        self.fit(save_path)
