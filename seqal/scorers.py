from typing import List

import numpy as np

from seqal.base_scorer import BaseScorer
from seqal.datasets import Sentence
from seqal.tagger import SequenceTagger


class LeastConfidenceScorer(BaseScorer):
    """Least confidence scorer

    https://dl.acm.org/doi/10.5555/1619410.1619452

    Args:
        BaseScorer: BaseScorer class.
    """

    def __call__(
        self,
        sents: List[Sentence],
        tag_type: str,
        query_number: int,
        token_based: bool = False,
        **kwargs
    ) -> List[int]:
        """Least confidence sampling workflow

        Args:
            sents (List[Sentence]): Sentences in data pool.
            tag_type (str): Tag type to predict.
            query_number (int): batch query number.
            token_based (bool, optional): If true, using query number as token number to query data.
                                          If false, using query number as sentence number to query data.

        kwargs:
            tagger: The tagger after training
            label_names (List[str]): Label name of all dataset
            embeddings: The embeddings method

        Returns:
            List[int]: Queried sentence ids.
        """
        tagger = kwargs["tagger"]
        self.predict(sents, tagger)
        scores = self.score(sents, tagger)
        sorted_sent_ids = self.sort(scores, order="descend")
        queried_sent_ids = self.query(sents, sorted_sent_ids, query_number, token_based)
        return queried_sent_ids

    def score(self, sents: List[Sentence], tagger: SequenceTagger) -> np.ndarray:
        log_probs = tagger.log_probability(sents)
        scores = 1 - np.exp(log_probs)
        return scores


class MaxNormLogProbScorer(BaseScorer):
    """Maximum Normalized Log-Probability scorer

    https://arxiv.org/abs/1707.05928

    Args:
        BaseScorer: BaseScorer class.
    """

    def __call__(
        self,
        sents: List[Sentence],
        tag_type: str,
        query_number: int,
        token_based: bool = False,
        **kwargs
    ) -> List[int]:
        """Maximum Normalized Log-Probability sampling workflow

        Args:
            sents (List[Sentence]): Sentences in data pool.
            tag_type (str): Tag type to predict.
            query_number (int): batch query number.
            token_based (bool, optional): If true, using query number as token number to query data.
                                          If false, using query number as sentence number to query data.

        kwargs:
            tagger: The tagger after training
            label_names (List[str]): Label name of all dataset
            embeddings: The embeddings method

        Returns:
            List[int]: Queried sentence ids.
        """
        tagger = kwargs["tagger"]
        self.predict(sents, tagger)
        scores = self.score(sents, tagger)
        sorted_sent_ids = self.sort(scores, order="ascend")
        queried_sent_ids = self.query(sents, sorted_sent_ids, query_number, token_based)
        return queried_sent_ids

    def score(self, sents: List[Sentence], tagger: SequenceTagger) -> np.ndarray:
        log_probs = tagger.log_probability(sents)
        lengths = np.array([len(sent) for sent in sents])
        normed_log_probs = log_probs / lengths
        return normed_log_probs
