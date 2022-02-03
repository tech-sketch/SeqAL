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


class TestLeastConfidenceScorer:
    def test_score_return_correct_result_if_log_probability_runs_after_prediction(
        self, lc_scorer: BaseScorer, predicted_sentences: List[Sentence]
    ) -> None:
        # Arrange
        tagger = MagicMock()
        tagger.log_probability = MagicMock(
            return_value=np.array([-0.4, -0.3, -0.2, -0.1])
        )
        log_probs = np.array([-0.4, -0.3, -0.2, -0.1])
        expected = 1 - np.exp(log_probs)

        # Act
        scores = lc_scorer.score(predicted_sentences, tagger=tagger)

        # Assert
        assert np.array_equal(scores, expected) is True

    def test_call_return_correct_result_if_sampling_workflow_works_fine(
        self, lc_scorer: BaseScorer, unlabeled_sentences: List[Sentence]
    ):
        # Arrange
        tag_type = "ner"
        label_names = ["O", "I-PER", "I-LOC", "I-ORG", "I-MISC"]
        query_number = 4
        token_based = False
        embeddings = MagicMock()
        tagger = MagicMock()
        lc_scorer.predict = MagicMock(return_value=None)
        lc_scorer.score = MagicMock(
            return_value=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        )

        # Act
        scores = lc_scorer(
            unlabeled_sentences,
            tag_type,
            query_number,
            token_based,
            tagger=tagger,
            label_names=label_names,
            embeddings=embeddings,
        )

        # Assert
        assert scores == [0, 1, 2, 3]


class TestMaxNormLogProbScorer:
    def test_score_return_correct_result_if_log_probability_runs_after_prediction(
        self, mnlp_scorer: BaseScorer, predicted_sentences: List[Sentence]
    ) -> None:
        # Arrange
        tagger = MagicMock()
        log_probs = np.array(
            [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05]
        )
        tagger.log_probability = MagicMock(return_value=log_probs)
        lengths = np.array([9, 2, 2, 30, 33, 33, 24, 40, 28, 38])
        expected = log_probs / lengths

        # Act
        scores = mnlp_scorer.score(predicted_sentences, tagger=tagger)

        # Assert
        assert np.array_equal(scores, expected) is True

    def test_call_return_correct_result_if_sampling_workflow_works_fine(
        self, mnlp_scorer: BaseScorer, unlabeled_sentences: List[Sentence]
    ):
        # Arrange
        tag_type = "ner"
        label_names = ["O", "I-PER", "I-LOC", "I-ORG", "I-MISC"]
        query_number = 4
        token_based = False
        embeddings = MagicMock()
        tagger = MagicMock()
        mnlp_scorer.predict = MagicMock(return_value=None)
        returned_norm_log_probs = np.array(
            [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05]
        )
        mnlp_scorer.score = MagicMock(return_value=returned_norm_log_probs)

        # Act
        scores = mnlp_scorer(
            unlabeled_sentences,
            tag_type,
            query_number,
            token_based,
            tagger=tagger,
            label_names=label_names,
            embeddings=embeddings,
        )

        # Assert
        assert scores == [0, 1, 2, 3]


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
