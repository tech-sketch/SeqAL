from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest
from flair.data import Sentence

from seqal.base_scorer import BaseScorer
from seqal.scorers import LeastConfidenceScorer, MaxNormLogProbScorer


@pytest.fixture()
def lc_scorer(scope="function"):
    lc_scorer = LeastConfidenceScorer()
    return lc_scorer


@pytest.fixture()
def mnlp_scorer(scope="function"):
    mnlp_scorer = MaxNormLogProbScorer()
    return mnlp_scorer


@pytest.fixture()
def mnlp_sents(scope="function"):
    s1 = MagicMock()
    s2 = MagicMock()
    s3 = MagicMock()
    s4 = MagicMock()
    s1.__len__ = MagicMock(return_value=1)
    s2.__len__ = MagicMock(return_value=1)
    s3.__len__ = MagicMock(return_value=1)
    s4.__len__ = MagicMock(return_value=1)
    sents = [s1, s2, s3, s4]
    return sents


class TestLeastConfidenceScorer:
    def test_score(self, lc_scorer: BaseScorer, sents: List[Sentence]) -> None:
        tagger = MagicMock()
        tagger.log_probability = MagicMock(
            return_value=np.array([-0.4, -0.3, -0.2, -0.1])
        )

        # Method result
        scores = lc_scorer.score(sents, tagger=tagger)

        # Expected result
        log_probs = np.array([-0.4, -0.3, -0.2, -0.1])
        expected = 1 - np.exp(log_probs)

        assert np.array_equal(scores, expected) is True

    def test_call(self, lc_scorer: BaseScorer, sents: List[Sentence]):
        tag_type = "ner"
        label_names = ["O", "I-PER", "I-LOC", "I-ORG", "I-MISC"]
        embeddings = MagicMock()
        tagger = MagicMock()
        tagger.predict = MagicMock(return_value=None)
        tagger.log_probability = MagicMock(
            return_value=np.array([-0.4, -0.3, -0.2, -0.1])
        )
        query_number = 4
        token_based = False

        # Test for the final result
        scores = lc_scorer(
            sents,
            tag_type,
            query_number,
            token_based,
            tagger=tagger,
            label_names=label_names,
            embeddings=embeddings,
        )
        expected = [0, 1, 2, 3]

        assert expected == scores


class TestMaxNormLogProbScorer:
    def test_score(self, mnlp_scorer: BaseScorer, mnlp_sents: List[Sentence]) -> None:
        tagger = MagicMock()
        tagger.log_probability = MagicMock(
            return_value=np.array([-0.4, -0.3, -0.2, -0.1])
        )

        # Method result
        scores = mnlp_scorer.score(mnlp_sents, tagger=tagger)

        # Expected result
        log_probs = np.array([-0.4, -0.3, -0.2, -0.1])
        lengths = np.array([len(sent) for sent in mnlp_sents])
        expected = log_probs / lengths

        assert np.array_equal(scores, expected) is True

    def test_call(self, mnlp_scorer: BaseScorer, mnlp_sents: List[Sentence]):
        # Because sents do not contains real tokens, mnlp_scorer.query() will be failed.
        # But the mnlp_scorer.query has been tested in base_scorer.query()
        # Here we only check the sorted_sent_ids.
        tagger = MagicMock()
        tagger.log_probability = MagicMock(
            return_value=np.array([-0.4, -0.3, -0.2, -0.1])
        )
        scores = mnlp_scorer.score(mnlp_sents, tagger=tagger)
        sorted_sent_ids = mnlp_scorer.sort(scores, order="ascend")

        # Expected result
        expected = [0, 1, 2, 3]

        assert expected == sorted_sent_ids


class TestDistributeSimilarityScorer:
    def test_score(self):
        # Raise error if sentence has no predictions

        # Raise error if sentence has no embeddings

        # If entity_list is empty

        # Expected output

        pass

    def test_entity_pair_score(self):
        # If entity pair list is empty

        # Expected output

        pass


class TestClusterSimilarityScorer:
    def test_score(self):
        # Raise error if sentence has no predictions

        # Raise error if sentence has no embeddings

        # If entity_list is empty

        # Expected output

        pass

    def test_clustering(self):
        # Input type check

        # If k is bigger than number of data

        # Expected output

        pass

    def test_cluster_entity(self):
        # If cluster is empty

        # Expected output

        pass


class TestCombinedScorer:
    def test_available_combination(self):
        # Raise error combination is not exist

        pass

    def test_score(self):
        # Raise error if sentence has no predictions

        # Raise error if sentence has no embeddings

        # Raise error if Scorer is not exist

        # Expected output

        pass

    def test_normalize_score(self):
        # Raise error if list length is not equal

        # Expected output

        pass
