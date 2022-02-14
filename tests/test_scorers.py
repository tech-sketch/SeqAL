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
    ClusterSimilarityScorer,
    DistributeSimilarityScorer,
    LeastConfidenceScorer,
    MaxNormLogProbScorer,
)


@pytest.fixture()
def lc_scorer(scope="function"):
    """LeastConfidenceScorer instance"""
    lc_scorer_instance = LeastConfidenceScorer()
    return lc_scorer_instance


@pytest.fixture()
def mnlp_scorer(scope="function"):
    """MaxNormLogProbScorer instance"""
    mnlp_scorer_instance = MaxNormLogProbScorer()
    return mnlp_scorer_instance


@pytest.fixture()
def ds_scorer(scope="function"):
    """DistributeSimilarityScorer instance"""
    ds_scorer_instance = DistributeSimilarityScorer()
    return ds_scorer_instance


@pytest.fixture()
def cs_scorer(scope="function"):
    """MaxNormLogProbScorer instance"""
    cs_scorer_instance = ClusterSimilarityScorer()
    return cs_scorer_instance


class TestLeastConfidenceScorer:
    """Test LeastConfidenceScorer class"""

    def test_score_return_correct_result_if_log_probability_runs_after_prediction(
        self, lc_scorer: BaseScorer, predicted_sentences: List[Sentence]
    ) -> None:
        """Test score function return correct result if log_probability runs after prediction"""
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
        """Test call function return correct result"""
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
    """Test MaxNormLogProbScorer class"""

    def test_score_return_correct_result_if_log_probability_runs_after_prediction(
        self, mnlp_scorer: BaseScorer, predicted_sentences: List[Sentence]
    ) -> None:
        """Test score function return correct result if log_probability runs after prediction"""
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
        """Test call function return correct result"""
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
    """Test DistributeSimilarityScorer class"""

    def test_call_return_correct_result(
        self, ds_scorer: BaseScorer, unlabeled_sentences: List[Sentence]
    ) -> None:
        """Test call function return correct result"""
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
        """Test call function return random sentence ids if entities is empty"""
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
        """Test get_entities function raise type_error if unlabeled sentences have not been predicted"""
        # Arrange
        tag_type = "ner"
        embeddings = MagicMock()
        embeddings.embed = MagicMock(return_value=None)

        # Assert
        with pytest.raises(TypeError):
            # Act
            ds_scorer.get_entities(unlabeled_sentences, embeddings, tag_type)

    def test_calculate_diversity(self, ds_scorer: BaseScorer) -> None:
        """Test calculate diversity function"""
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
        """Test sentence diversity function"""
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
        """Test score function"""
        # Arrange
        sents = [0, 1]
        entities = Entities()
        ds_scorer.sentence_diversities = MagicMock(return_value={0: 0, 1: -0.5})

        # Act
        sentence_scores = ds_scorer.score(sents, entities)

        # Assert
        assert np.array_equal(sentence_scores, np.array([0, -0.5]))


class TestClusterSimilarityScorer:
    """Test ClusterSimilarityScorer class"""

    def test_call_return_correct_result(
        self, cs_scorer: BaseScorer, unlabeled_sentences: List[Sentence]
    ) -> None:
        """Test call function return correct result"""
        # Arrange
        tag_type = "ner"
        kmeans_params = {"n_cluster": 8, "n_init": 10, "random_state": 0}
        query_number = 4
        token_based = False
        embeddings = MagicMock()
        tagger = MagicMock()
        entities = Entities()
        entities.entities = [None]
        cs_scorer.get_entities = MagicMock(return_value=entities)
        cs_scorer.score = MagicMock(
            return_value=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        )

        # Act
        queried_sent_ids = cs_scorer(
            unlabeled_sentences,
            tag_type,
            query_number,
            token_based,
            tagger=tagger,
            kmeans_params=kmeans_params,
            embeddings=embeddings,
        )

        # Assert
        assert queried_sent_ids == [9, 8, 7, 6]

    def test_call_return_random_sent_ids_if_entities_is_empty(
        self, cs_scorer: BaseScorer, unlabeled_sentences: List[Sentence]
    ) -> None:
        """Test call function return random sentence ids if entities is empty"""
        # Arrange
        tag_type = "ner"
        kmeans_params = {"n_cluster": 8, "n_init": 10, "random_state": 0}
        query_number = 4
        token_based = False
        embeddings = MagicMock()
        tagger = MagicMock()
        entities = Entities()
        cs_scorer.get_entities = MagicMock(return_value=entities)

        random.seed(0)
        sent_ids = list(range(len(unlabeled_sentences)))
        expected_random_sent_ids = random.sample(sent_ids, len(sent_ids))

        # Act
        queried_sent_ids = cs_scorer(
            unlabeled_sentences,
            tag_type,
            query_number,
            token_based,
            tagger=tagger,
            kmeans_params=kmeans_params,
            embeddings=embeddings,
        )

        # Assert
        assert queried_sent_ids == expected_random_sent_ids[:query_number]

    def test_get_entities_raise_type_error_if_unlabeled_sentences_have_not_been_predicted(
        self, cs_scorer: BaseScorer, unlabeled_sentences: List[Sentence]
    ) -> None:
        """Test get_entities function raise type_error if unlabeled sentences have not been predicted"""
        # Arrange
        tag_type = "ner"
        embeddings = MagicMock()
        embeddings.embed = MagicMock(return_value=None)

        # Assert
        with pytest.raises(TypeError):
            # Act
            cs_scorer.get_entities(unlabeled_sentences, embeddings, tag_type)

    def test_kmeans_raise_key_error_if_n_cluster_param_is_not_found(
        self, cs_scorer: BaseScorer, unlabeled_sentences: List[Sentence]
    ) -> None:
        """Test kmeans function raise key_error if n_cluster parameter is not found"""
        # Arrange
        kmeans_params = {"n_init": 10, "random_state": 0}

        # Assert
        with pytest.raises(KeyError):
            # Act
            cs_scorer.kmeans(unlabeled_sentences, kmeans_params)

    def test_kmeans_return_correct_result(
        self, cs_scorer: BaseScorer, unlabeled_sentences: List[Sentence]
    ) -> None:
        """Test kmeans function return correct result"""
        # Arrange
        kmeans_params = {"n_clusters": 2, "n_init": 10, "random_state": 0}
        e0 = MagicMock(label="PER", vector=torch.tensor([1, 2]))
        e1 = MagicMock(label="PER", vector=torch.tensor([1, 4]))
        e2 = MagicMock(label="PER", vector=torch.tensor([1, 0]))
        e3 = MagicMock(label="PER", vector=torch.tensor([10, 2]))
        e4 = MagicMock(label="PER", vector=torch.tensor([10, 4]))
        e5 = MagicMock(label="PER", vector=torch.tensor([10, 0]))

        entities = Entities()
        entities.entities = [e0, e1, e2, e3, e4, e5]

        # Act
        cluster_centers_matrix, entity_cluster_nums = cs_scorer.kmeans(
            entities.entities, kmeans_params
        )

        # Assert
        assert np.array_equal(
            cluster_centers_matrix, np.array([[10.0, 2.0], [1.0, 2.0]])
        )
        assert np.array_equal(entity_cluster_nums, np.array([1, 1, 1, 0, 0, 0]))

    def test_assign_cluster(
        self, cs_scorer: BaseScorer, unlabeled_sentences: List[Sentence]
    ) -> None:
        """Test assign cluster function"""
        # Arrange
        e0 = MagicMock(label="PER", vector=torch.tensor([1, 2]))
        entities = Entities()
        entities.entities = [e0]
        entity_cluster_nums = np.array([0])

        # Act
        new_entities = cs_scorer.assign_cluster(entities, entity_cluster_nums)

        # Assert
        assert new_entities.entities[0].cluster == 0

    def test_calculate_diversity(self, cs_scorer: BaseScorer) -> None:
        """Test calculate diversity function"""
        # Arrange
        e0 = MagicMock(cluster=1, vector=torch.tensor([1, 2]))
        e1 = MagicMock(cluster=1, vector=torch.tensor([1, 4]))
        e2 = MagicMock(cluster=1, vector=torch.tensor([1, 0]))
        e3 = MagicMock(cluster=0, vector=torch.tensor([10, 2]))
        e4 = MagicMock(cluster=0, vector=torch.tensor([10, 4]))
        e5 = MagicMock(cluster=0, vector=torch.tensor([10, 0]))
        entities_per_cluster = {1: [e0, e1, e2], 0: [e3, e4, e5]}
        sentence_entities = [e0, e3]
        cluster_centers_matrix = np.array([[10.0, 2.0], [1.0, 2.0]])

        # Act
        sentence_score = cs_scorer.calculate_diversity(
            sentence_entities, entities_per_cluster, cluster_centers_matrix
        )

        # Assert
        assert (abs(0.7138 - sentence_score) < 0.001) is True

    def test_sentence_diversity(self, cs_scorer: BaseScorer) -> None:
        """Test sentence diversity function"""
        # Arrange
        e0 = MagicMock(cluster=1, vector=torch.tensor([1, 2]))
        e1 = MagicMock(cluster=1, vector=torch.tensor([1, 4]))
        e2 = MagicMock(cluster=1, vector=torch.tensor([1, 0]))
        e3 = MagicMock(cluster=0, vector=torch.tensor([10, 2]))
        e4 = MagicMock(cluster=0, vector=torch.tensor([10, 4]))
        e5 = MagicMock(cluster=0, vector=torch.tensor([10, 0]))
        entities_per_cluster = {1: [e0, e1, e2], 0: [e3, e4, e5]}
        entities = Entities()
        entities.entities = [e0, e1, e2, e3, e4, e5]
        cluster_centers_matrix = np.array([[10.0, 2.0], [1.0, 2.0]])
        entities_per_sentence = {0: [e0, e3]}
        type(entities).group_by_sentence = PropertyMock(
            return_value=entities_per_sentence
        )
        type(entities).group_by_cluster = PropertyMock(
            return_value=entities_per_cluster
        )

        # Act
        sentence_score = cs_scorer.sentence_diversities(
            entities, cluster_centers_matrix
        )

        # Assert
        assert (abs(0.7138 - sentence_score[0]) < 0.001) is True

    def test_score(self, cs_scorer: BaseScorer) -> None:
        """Test score function"""
        # Arrange
        sents = [0]  # Just one setnence
        kmeans_params = {"n_cluster": 8, "n_init": 10, "random_state": 0}

        cluster_centers_matrix = np.array([[10.0, 2.0], [1.0, 2.0]])
        entity_cluster_nums = np.array([1, 1, 1, 0, 0, 0])
        cs_scorer.kmeans = MagicMock(
            return_value=(cluster_centers_matrix, entity_cluster_nums)
        )

        e0 = MagicMock(cluster=1, vector=torch.tensor([1, 2]))
        e1 = MagicMock(cluster=1, vector=torch.tensor([1, 4]))
        e2 = MagicMock(cluster=1, vector=torch.tensor([1, 0]))
        e3 = MagicMock(cluster=0, vector=torch.tensor([10, 2]))
        e4 = MagicMock(cluster=0, vector=torch.tensor([10, 4]))
        e5 = MagicMock(cluster=0, vector=torch.tensor([10, 0]))
        entities = Entities()
        entities.entities = [e0, e1, e2, e3, e4, e5]
        cs_scorer.assign_cluster = MagicMock(return_value=entities)

        cs_scorer.sentence_diversities = MagicMock(return_value={0: 0.7138})

        # Act
        sentence_scores = cs_scorer.score(sents, entities, kmeans_params)

        # Assert
        assert np.array_equal(sentence_scores, np.array([0.7138]))
