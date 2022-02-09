import random
from typing import List
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
import torch
from flair.data import Sentence

from seqal.base_scorer import BaseScorer
from seqal.data import Entities
from seqal.scorers import (
    DistributeSimilarityScorer,
    LeastConfidenceScorer,
    MaxNormLogProbScorer,
)


@pytest.fixture()
def lc_scorer(scope="function"):
    lc_scorer = LeastConfidenceScorer()
    return lc_scorer


@pytest.fixture()
def mnlp_scorer(scope="function"):
    mnlp_scorer = MaxNormLogProbScorer()
    return mnlp_scorer


@pytest.fixture()
def ds_scorer(scope="function"):
    ds_scorer = DistributeSimilarityScorer()
    return ds_scorer


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
        queried_sent_ids = lc_scorer(
            unlabeled_sentences,
            tag_type,
            query_number,
            token_based,
            tagger=tagger,
            label_names=label_names,
            embeddings=embeddings,
        )

        # Assert
        assert queried_sent_ids == [0, 1, 2, 3]


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
        queried_sent_ids = mnlp_scorer(
            unlabeled_sentences,
            tag_type,
            query_number,
            token_based,
            tagger=tagger,
            label_names=label_names,
            embeddings=embeddings,
        )

        # Assert
        assert queried_sent_ids == [0, 1, 2, 3]


class TestDistributeSimilarityScorer:
    def test_call_return_correct_result(
        self, ds_scorer: BaseScorer, unlabeled_sentences: List[Sentence]
    ) -> None:
        # Arrange
        tag_type = "ner"
        label_names = ["O", "I-PER", "I-LOC", "I-ORG", "I-MISC"]
        query_number = 4
        token_based = False
        embeddings = MagicMock()
        tagger = MagicMock()
        entities = Entities()
        entities.entities = [None]
        ds_scorer.get_entities = MagicMock(return_value=entities)
        ds_scorer.score = MagicMock(
            return_value=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        )

        # Act
        queried_sent_ids = ds_scorer(
            unlabeled_sentences,
            tag_type,
            query_number,
            token_based,
            tagger=tagger,
            label_names=label_names,
            embeddings=embeddings,
        )

        # Assert
        assert queried_sent_ids == [9, 8, 7, 6]

    def test_call_return_random_sent_ids_if_entities_is_empty(
        self, ds_scorer: BaseScorer, unlabeled_sentences: List[Sentence]
    ) -> None:
        # Arrange
        tag_type = "ner"
        label_names = ["O", "I-PER", "I-LOC", "I-ORG", "I-MISC"]
        query_number = 4
        token_based = False
        embeddings = MagicMock()
        tagger = MagicMock()
        entities = Entities()
        ds_scorer.get_entities = MagicMock(return_value=entities)

        random.seed(0)
        sent_ids = list(range(len(unlabeled_sentences)))
        expected_random_sent_ids = random.sample(sent_ids, len(sent_ids))

        # Act
        queried_sent_ids = ds_scorer(
            unlabeled_sentences,
            tag_type,
            query_number,
            token_based,
            tagger=tagger,
            label_names=label_names,
            embeddings=embeddings,
        )

        # Assert
        assert queried_sent_ids == expected_random_sent_ids[:query_number]

    def test_get_entities_raise_type_error_if_unlabeled_sentences_have_not_been_predicted(
        self, ds_scorer: BaseScorer, unlabeled_sentences: List[Sentence]
    ) -> None:
        # Arrange
        tag_type = "ner"
        embeddings = MagicMock()
        embeddings.embed = MagicMock(return_value=None)

        # Assert
        with pytest.raises(TypeError):
            # Act
            ds_scorer.get_entities(unlabeled_sentences, embeddings, tag_type)

    def test_calculate_diversity(self, ds_scorer: BaseScorer) -> None:

        # Arrange
        e0 = MagicMock(label="PER", vector=torch.tensor([-0.1, 0.1]))
        e1 = MagicMock(label="PER", vector=torch.tensor([0.1, 0.1]))
        e2 = MagicMock(label="PER", vector=torch.tensor([0.1, -0.1]))
        e3 = MagicMock(label="LOC", vector=torch.tensor([-0.1, -0.1]))

        entities_per_label = {"PER": [e0, e1, e2], "LOC": [e3]}
        sentence_entities = [e0, e3]

        # Act
        sentence_score = ds_scorer.calculate_diversity(
            sentence_entities, entities_per_label
        )

        # Assert
        assert sentence_score == 0

    def test_sentence_diversity(self, ds_scorer: BaseScorer) -> None:

        # Arrange
        e0 = MagicMock(label="PER", vector=torch.tensor([-0.1, 0.1]))
        e1 = MagicMock(label="PER", vector=torch.tensor([0.1, 0.1]))
        e2 = MagicMock(label="PER", vector=torch.tensor([0.1, -0.1]))
        e3 = MagicMock(label="LOC", vector=torch.tensor([-0.1, -0.1]))
        entities_per_sentence = {0: [e0, e3], 1: [e1, e2]}
        entities_per_label = {"PER": [e0, e1, e2], "LOC": [e3]}
        entities = Entities()
        type(entities).group_by_sentence = PropertyMock(
            return_value=entities_per_sentence
        )
        type(entities).group_by_label = PropertyMock(return_value=entities_per_label)

        # Act
        sentence_score = ds_scorer.sentence_diversities(entities)

        # Assert
        assert sentence_score[0] == 0
        assert (abs(-0.5 - sentence_score[1]) < 0.00001) is True

    def test_score(self, ds_scorer: BaseScorer) -> None:
        # Arrange
        sents = [0, 1]
        entities = Entities()
        ds_scorer.sentence_diversities = MagicMock(return_value={0: 0, 1: -0.5})

        # Act
        sentence_scores = ds_scorer.score(sents, entities)

        # Assert
        assert np.array_equal(sentence_scores, np.array([0, -0.5]))
