import math
import random
from typing import Dict, List
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
import torch
from flair.data import Sentence
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from seqal.data import Entities, Entity
from seqal.samplers import (
    BaseSampler,
    ClusterSimilaritySampler,
    CombinedMultipleSampler,
    DistributeSimilaritySampler,
    LeastConfidenceSampler,
    MaxNormLogProbSampler,
    RandomSampler,
    StringNGramSampler,
)


@pytest.fixture()
def lc_sampler(scope="function"):
    """LeastConfidenceSampler instance"""
    lc_sampler = LeastConfidenceSampler()
    return lc_sampler


@pytest.fixture()
def mnlp_sampler(scope="function"):
    """MaxNormLogProbSampler instance"""
    mnlp_sampler = MaxNormLogProbSampler()
    return mnlp_sampler


@pytest.fixture()
def sn_sampler(scope="function"):
    """StringNGramSampler instance"""
    sn_sampler = StringNGramSampler()
    return sn_sampler


@pytest.fixture()
def ds_sampler(scope="function"):
    """DistributeSimilaritySampler instance"""
    ds_sampler = DistributeSimilaritySampler()
    return ds_sampler


@pytest.fixture()
def cs_sampler(scope="function"):
    """ClusterSimilaritySampler instance"""
    cs_sampler = ClusterSimilaritySampler()
    return cs_sampler


@pytest.fixture()
def cm_sampler(scope="function"):
    """CombinedMultipleSampler instance"""
    cm_sampler = CombinedMultipleSampler()
    return cm_sampler


@pytest.fixture()
def sampler_params(scope="function"):
    """Common parameters for sampler test"""
    params = {
        "tag_type": "ner",
        "query_number": 4,
        "label_names": ["O", "PER", "LOC"],
        "token_based": False,
        "tagger": MagicMock(),
        "embeddings": MagicMock(),
        "kmeans_params": {"n_clusters": 2, "n_init": 10, "random_state": 0},
    }
    return params


@pytest.fixture()
def entities4(scope="function"):
    """4 entities for ds_sampler test"""
    e0 = MagicMock(
        id=0, sent_id=0, label="PER", text="Peter", vector=torch.tensor([-0.1, 0.1])
    )
    e1 = MagicMock(
        id=0, sent_id=1, label="PER", text="Lester", vector=torch.tensor([0.1, 0.1])
    )
    e2 = MagicMock(
        id=1, sent_id=1, label="PER", text="Jessy", vector=torch.tensor([0.1, -0.1])
    )
    e3 = MagicMock(
        id=1, sent_id=0, label="LOC", text="NYC", vector=torch.tensor([-0.1, -0.1])
    )

    return [e0, e1, e2, e3]


@pytest.fixture()
def trigram_examples(scope="function"):
    """Trigram examples for test"""
    trigram_examples = {
        "Peter": ["$$P1", "$Pe1", "Pet1", "ete1", "ter1", "er$1", "r$$1"],
        "prepress": [
            "$$p1",
            "$pr1",
            "pre1",
            "rep1",
            "epr1",
            "pre2",
            "res1",
            "ess1",
            "ss$1",
            "s$$1",
        ],
        "Lester": ["$$L1", "$Le1", "Les1", "est1", "ste1", "ter1", "er$1", "r$$1"],
    }

    return trigram_examples


@pytest.fixture()
def entities_per_label(entities4: List[Entity]):
    """Entity list in each label"""
    entities_per_label = {
        "PER": [entities4[0], entities4[1], entities4[2]],
        "LOC": [entities4[3]],
    }
    return entities_per_label


@pytest.fixture()
def entities_per_sentence(entities4: List[Entity]):
    """Entity list in each sentence"""
    entities_per_sentence = {
        0: [entities4[0], entities4[3]],
        1: [entities4[1], entities4[2]],
    }

    return entities_per_sentence


@pytest.fixture()
def similarity_matrix_per_label(scope="function"):
    """Similarity matrix for each label"""
    similarity_matrix_per_label = {
        "PER": np.array(
            [
                [1.0000, 0.0000, -1.0000],
                [0.0000, 1.0000, 0.0000],
                [-1.0000, 0.0000, 1.0000],
            ]
        ),
        "LOC": np.array([[1.0000]]),
    }

    return similarity_matrix_per_label


@pytest.fixture()
def similarity_matrix_per_label_cosine_trigram(scope="function"):
    """Similarity matrix for each label"""
    similarity_matrix_per_label_cosine_trigram = {
        "PER": np.array(
            [
                [1.0, 0.400891862869, 0.0000],
                [0.400891862869, 1.0, 0.0000],
                [0.0000, 0.0000, 1.0],
            ]
        ),
        "LOC": np.array([[1]]),
    }

    return similarity_matrix_per_label_cosine_trigram


@pytest.fixture()
def entity_id_map(scope="function"):
    """Entity list in each sentence"""
    entity_id_map = {
        "PER": np.array([[0, 1], [1, 2]]),
        "LOC": np.array([[1, 0], [1, 1]]),
    }

    return entity_id_map


@pytest.fixture()
def entities6(scope="function"):
    """4 entities for ds_sampler test"""
    e0 = MagicMock(cluster=1, vector=torch.tensor([1, 2]))
    e1 = MagicMock(cluster=1, vector=torch.tensor([1, 4]))
    e2 = MagicMock(cluster=1, vector=torch.tensor([1, 0]))
    e3 = MagicMock(cluster=0, vector=torch.tensor([10, 2]))
    e4 = MagicMock(cluster=0, vector=torch.tensor([10, 4]))
    e5 = MagicMock(cluster=0, vector=torch.tensor([10, 0]))

    return [e0, e1, e2, e3, e4, e5]


def compare_exact(dict1, dict2):
    """Return whether two dicts of arrays are exactly equal"""
    if dict1.keys() != dict2.keys():
        return False
    return all(np.array_equal(dict1[key], dict2[key]) for key in dict1)


def compare_approximate(dict1, dict2):
    """Return whether two dicts of arrays are roughly equal"""
    if dict1.keys() != dict2.keys():
        return False
    return all(
        np.allclose(dict1[key], dict2[key], rtol=1e-05, atol=1e-08) for key in dict1
    )


class TestRandomSampler:
    """Test RandomSampler class"""

    def test_call_return_random_sent_ids(
        self,
        unlabeled_sentences: List[Sentence],
        sampler_params: dict,
    ) -> None:
        """Test call function return random sentence ids if entities is empty"""
        # Arrange
        random_sampler = RandomSampler()

        random.seed(0)
        sent_ids = list(range(len(unlabeled_sentences)))
        expected_random_sent_ids = random.sample(sent_ids, len(sent_ids))

        # Act

        queried_sent_ids = random_sampler(
            unlabeled_sentences,
            sampler_params["tag_type"],
            sampler_params["query_number"],
            sampler_params["token_based"],
            tagger=sampler_params["tagger"],
            label_names=sampler_params["label_names"],
            embeddings=sampler_params["embeddings"],
        )

        # Assert
        assert (
            queried_sent_ids
            == expected_random_sent_ids[: sampler_params["query_number"]]
        )


class TestLeastConfidenceSampler:
    """Test LeastConfidenceSampler class"""

    def test_score_return_correct_result_if_log_probability_runs_after_prediction(
        self, lc_sampler: BaseSampler, predicted_sentences: List[Sentence]
    ) -> None:
        """Test RandomSampler.score function return correct result if log_probability runs after prediction"""
        # Arrange
        tagger = MagicMock()
        tagger.log_probability = MagicMock(
            return_value=np.array([-0.4, -0.3, -0.2, -0.1])
        )
        log_probs = np.array([-0.4, -0.3, -0.2, -0.1])
        expected = 1 - np.exp(log_probs)

        # Act
        scores = lc_sampler.score(predicted_sentences, tagger=tagger)

        # Assert
        assert np.array_equal(scores, expected) is True

    def test_call_return_correct_result_if_sampling_workflow_works_fine(
        self,
        lc_sampler: BaseSampler,
        unlabeled_sentences: List[Sentence],
        sampler_params: dict,
    ):
        """Test call function return correct result"""
        # Arrange
        lc_sampler.predict = MagicMock(return_value=None)
        lc_sampler.score = MagicMock(
            return_value=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        )

        # Act
        queried_sent_ids = lc_sampler(
            unlabeled_sentences,
            sampler_params["tag_type"],
            sampler_params["query_number"],
            sampler_params["token_based"],
            tagger=sampler_params["tagger"],
            embeddings=sampler_params["embeddings"],
            label_names=sampler_params["label_names"],
        )

        # Assert
        assert queried_sent_ids == [0, 1, 2, 3]


class TestMaxNormLogProbSampler:
    """Test MaxNormLogProbSampler class"""

    def test_score_return_correct_result_if_log_probability_runs_after_prediction(
        self, mnlp_sampler: BaseSampler, predicted_sentences: List[Sentence]
    ) -> None:
        """Test MaxNormLogProbSampler.score function return correct result if log_probability runs after prediction"""
        # Arrange
        tagger = MagicMock()
        log_probs = np.array(
            [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05]
        )
        tagger.log_probability = MagicMock(return_value=log_probs)
        lengths = np.array([9, 2, 2, 30, 33, 33, 24, 40, 28, 38])
        expected = log_probs / lengths

        # Act
        scores = mnlp_sampler.score(predicted_sentences, tagger=tagger)

        # Assert
        assert np.array_equal(scores, expected) is True

    def test_call_return_correct_result_if_sampling_workflow_works_fine(
        self,
        mnlp_sampler: BaseSampler,
        unlabeled_sentences: List[Sentence],
        sampler_params: dict,
    ):
        """Test call function return correct result"""
        # Arrange
        mnlp_sampler.predict = MagicMock(return_value=None)
        returned_norm_log_probs = np.array(
            [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05]
        )
        mnlp_sampler.score = MagicMock(return_value=returned_norm_log_probs)

        # Act
        queried_sent_ids = mnlp_sampler(
            unlabeled_sentences,
            sampler_params["tag_type"],
            sampler_params["query_number"],
            sampler_params["token_based"],
            tagger=sampler_params["tagger"],
            label_names=sampler_params["label_names"],
            embeddings=sampler_params["embeddings"],
        )

        # Assert
        assert queried_sent_ids == [0, 1, 2, 3]


class TestStringNGramSampler:
    """Test StringNGramSampler class"""

    def test_call_return_correct_result(
        self,
        sn_sampler: BaseSampler,
        unlabeled_sentences: List[Sentence],
        sampler_params: dict,
    ) -> None:
        """Test call function return correct result"""
        # Arrange
        entities = Entities()
        entities.entities = [None]
        sn_sampler.get_entities = MagicMock(return_value=entities)
        sn_sampler.score = MagicMock(
            return_value=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        )

        # Act
        queried_sent_ids = sn_sampler(
            unlabeled_sentences,
            sampler_params["tag_type"],
            sampler_params["query_number"],
            sampler_params["token_based"],
            tagger=sampler_params["tagger"],
            label_names=sampler_params["label_names"],
            embeddings=sampler_params["embeddings"],
        )

        # Assert
        assert queried_sent_ids == [9, 8, 7, 6]

    def test_call_return_random_sent_ids_if_entities_is_empty(
        self,
        sn_sampler: BaseSampler,
        unlabeled_sentences: List[Sentence],
        sampler_params: dict,
    ) -> None:
        """Test call function return random sentence ids if entities is empty"""
        # Arrange
        entities = Entities()
        sn_sampler.get_entities = MagicMock(return_value=entities)

        random.seed(0)
        sent_ids = list(range(len(unlabeled_sentences)))
        expected_random_sent_ids = random.sample(sent_ids, len(sent_ids))

        # Act
        queried_sent_ids = sn_sampler(
            unlabeled_sentences,
            sampler_params["tag_type"],
            sampler_params["query_number"],
            sampler_params["token_based"],
            tagger=sampler_params["tagger"],
            label_names=sampler_params["label_names"],
            embeddings=sampler_params["embeddings"],
        )

        # Assert
        assert (
            queried_sent_ids
            == expected_random_sent_ids[: sampler_params["query_number"]]
        )

    def test_trigram(self, sn_sampler: BaseSampler, trigram_examples: dict) -> None:
        """Test StringNGramSampler.trigram function"""
        # Arrange
        entity = MagicMock(text="prepress")

        # Act
        trigrams = sn_sampler.trigram(entity)

        # Assert
        assert trigrams == trigram_examples["prepress"]

    def trigram_cosine_similarity(
        self, sn_sampler: BaseSampler, trigram_examples: dict
    ) -> None:
        """Test StringNGramSampler.trigram_cosine_similarity function"""
        # Arrange
        entity_trigram1 = trigram_examples["Peter"]
        entity_trigram2 = trigram_examples["Lester"]
        expected = len(set(entity_trigram1) & set(entity_trigram2)) / math.sqrt(
            len(entity_trigram1) * len(entity_trigram2)
        )

        # Act
        similarity = sn_sampler.trigram_cosine_similarity(
            entity_trigram1, entity_trigram2
        )

        # Assert
        assert expected == similarity

    def test_calculate_diversity(
        self,
        sn_sampler: BaseSampler,
        entities_per_sentence: dict,
        entity_id_map: dict,
        similarity_matrix_per_label: dict,
    ) -> None:
        """Test StringNGramSampler.calculate_diversity function"""
        # Arrange
        expected = {0: 0, 1: -0.5}

        # Act
        sentence_scores = sn_sampler.calculate_diversity(
            entities_per_sentence, entity_id_map, similarity_matrix_per_label
        )

        # Assert
        assert sentence_scores == expected

    def test_sentence_diversity(
        self, sn_sampler: BaseSampler, entities4: List[Entity]
    ) -> None:
        """Test StringNGramSampler.sentence_diversity function"""
        # Arrange
        entities = Entities()
        entities.entities = entities4
        expected = {0: 0.5, 1: 0}

        # Act
        sentence_scores = sn_sampler.sentence_diversities(entities)

        # Assert
        assert compare_approximate(sentence_scores, expected) is True

    def test_similarity_matrix_per_label(
        self,
        sn_sampler: BaseSampler,
        entities_per_label: dict,
        similarity_matrix_per_label_cosine_trigram: Dict[str, torch.Tensor],
    ) -> None:
        """Test StringNGramSampler.similarity_matrix_per_label function"""
        # Act
        sentence_scores = sn_sampler.similarity_matrix_per_label(entities_per_label)

        # Assert
        assert (
            compare_approximate(
                sentence_scores, similarity_matrix_per_label_cosine_trigram
            )
            is True
        )

    def test_get_entity_id_map(
        self,
        sn_sampler: BaseSampler,
        entities_per_sentence: dict,
        entities_per_label: dict,
        entity_id_map: dict,
    ) -> None:
        """Test StringNGramSampler.get_entity_id_map function"""
        # Arrange
        sentence_count = max(entities_per_sentence.keys()) + 1
        max_entity_count = max(
            [len(entities) for entities in entities_per_sentence.values()]
        )

        # Act
        entity_id_map_result = sn_sampler.get_entity_id_map(
            entities_per_label, sentence_count, max_entity_count
        )

        # Assert
        assert compare_exact(entity_id_map_result, entity_id_map) is True

    def test_score(self, sn_sampler: BaseSampler) -> None:
        """Test StringNGramSampler.score function"""
        # Arrange
        sents = [0, 1]
        entities = Entities()
        sn_sampler.sentence_diversities = MagicMock(return_value={0: 0, 1: -0.5})

        # Act
        sentence_scores = sn_sampler.score(sents, entities)

        # Assert
        assert np.array_equal(sentence_scores, np.array([0, -0.5]))


class TestDistributeSimilaritySampler:
    """Test DistributeSimilaritySampler class"""

    def test_call_return_correct_result(
        self,
        ds_sampler: BaseSampler,
        unlabeled_sentences: List[Sentence],
        sampler_params: dict,
    ) -> None:
        """Test call function return correct result"""
        # Arrange
        entities = Entities()
        entities.entities = [None]
        ds_sampler.get_entities = MagicMock(return_value=entities)
        ds_sampler.score = MagicMock(
            return_value=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        )

        # Act
        queried_sent_ids = ds_sampler(
            unlabeled_sentences,
            sampler_params["tag_type"],
            sampler_params["query_number"],
            sampler_params["token_based"],
            tagger=sampler_params["tagger"],
            label_names=sampler_params["label_names"],
            embeddings=sampler_params["embeddings"],
        )

        # Assert
        assert queried_sent_ids == [9, 8, 7, 6]

    def test_call_return_random_sent_ids_if_entities_is_empty(
        self,
        ds_sampler: BaseSampler,
        unlabeled_sentences: List[Sentence],
        sampler_params: dict,
    ) -> None:
        """Test call function return random sentence ids if entities is empty"""
        # Arrange
        entities = Entities()
        ds_sampler.get_entities = MagicMock(return_value=entities)

        random.seed(0)
        sent_ids = list(range(len(unlabeled_sentences)))
        expected_random_sent_ids = random.sample(sent_ids, len(sent_ids))

        # Act
        queried_sent_ids = ds_sampler(
            unlabeled_sentences,
            sampler_params["tag_type"],
            sampler_params["query_number"],
            sampler_params["token_based"],
            tagger=sampler_params["tagger"],
            label_names=sampler_params["label_names"],
            embeddings=sampler_params["embeddings"],
        )

        # Assert
        assert (
            queried_sent_ids
            == expected_random_sent_ids[: sampler_params["query_number"]]
        )

    def test_calculate_diversity(
        self,
        ds_sampler: BaseSampler,
        entities_per_sentence: dict,
        entity_id_map: dict,
        similarity_matrix_per_label: dict,
    ) -> None:
        """Test DistributeSimilaritySampler.calculate_diversity function"""
        # Arrange
        expected = {0: 0, 1: -0.5}

        # Act
        sentence_scores = ds_sampler.calculate_diversity(
            entities_per_sentence, entity_id_map, similarity_matrix_per_label
        )

        # Assert
        assert sentence_scores == expected

    def test_sentence_diversity(
        self, ds_sampler: BaseSampler, entities4: List[Entity]
    ) -> None:
        """Test DistributeSimilaritySampler.sentence_diversity function"""
        # Arrange
        entities = Entities()
        entities.entities = entities4
        expected = {0: 0, 1: -0.5}

        # Act
        sentence_scores = ds_sampler.sentence_diversities(entities)

        # Assert
        assert compare_approximate(sentence_scores, expected) is True

    def test_similarity_matrix_per_label(
        self,
        ds_sampler: BaseSampler,
        entities_per_label: dict,
        similarity_matrix_per_label: Dict[str, torch.Tensor],
    ) -> None:
        """Test DistributeSimilaritySampler.similarity_matrix_per_label function"""
        # Act
        sentence_scores = ds_sampler.similarity_matrix_per_label(entities_per_label)

        # Assert
        assert compare_approximate(sentence_scores, similarity_matrix_per_label) is True

    def test_get_entity_id_map(
        self,
        ds_sampler: BaseSampler,
        entities_per_sentence: dict,
        entities_per_label: dict,
        entity_id_map: dict,
    ) -> None:
        """Test DistributeSimilaritySampler.get_entity_id_map function"""
        # Arrange
        sentence_count = max(entities_per_sentence.keys()) + 1
        max_entity_count = max(
            [len(entities) for entities in entities_per_sentence.values()]
        )

        # Act
        entity_id_map_result = ds_sampler.get_entity_id_map(
            entities_per_label, sentence_count, max_entity_count
        )

        # Assert
        assert compare_exact(entity_id_map_result, entity_id_map) is True

    def test_score(self, ds_sampler: BaseSampler) -> None:
        """Test DistributeSimilaritySampler.score function"""
        # Arrange
        sents = [0, 1]
        entities = Entities()
        ds_sampler.sentence_diversities = MagicMock(return_value={0: 0, 1: -0.5})

        # Act
        sentence_scores = ds_sampler.score(sents, entities)

        # Assert
        assert np.array_equal(sentence_scores, np.array([0, -0.5]))


class TestClusterSimilaritySampler:
    """Test ClusterSimilaritySampler class"""

    def test_call_return_correct_result(
        self,
        cs_sampler: BaseSampler,
        unlabeled_sentences: List[Sentence],
        sampler_params: dict,
    ) -> None:
        """Test call function return correct result"""
        # Arrange
        entities = Entities()
        entities.entities = [None]
        cs_sampler.get_entities = MagicMock(return_value=entities)
        cs_sampler.score = MagicMock(
            return_value=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        )

        # Act
        queried_sent_ids = cs_sampler(
            unlabeled_sentences,
            sampler_params["tag_type"],
            sampler_params["query_number"],
            sampler_params["token_based"],
            tagger=sampler_params["tagger"],
            embeddings=sampler_params["embeddings"],
            kmeans_params=sampler_params["kmeans_params"],
        )

        # Assert
        assert queried_sent_ids == [9, 8, 7, 6]

    def test_call_return_random_sent_ids_if_entities_is_empty(
        self,
        cs_sampler: BaseSampler,
        unlabeled_sentences: List[Sentence],
        sampler_params: dict,
    ) -> None:
        """Test call function return random sentence ids if entities is empty"""
        # Arrange
        entities = Entities()
        cs_sampler.get_entities = MagicMock(return_value=entities)

        random.seed(0)
        sent_ids = list(range(len(unlabeled_sentences)))
        expected_random_sent_ids = random.sample(sent_ids, len(sent_ids))

        # Act
        queried_sent_ids = cs_sampler(
            unlabeled_sentences,
            sampler_params["tag_type"],
            sampler_params["query_number"],
            sampler_params["token_based"],
            tagger=sampler_params["tagger"],
            embeddings=sampler_params["embeddings"],
            kmeans_params=sampler_params["kmeans_params"],
        )

        # Assert
        assert (
            queried_sent_ids
            == expected_random_sent_ids[: sampler_params["query_number"]]
        )

    def test_kmeans_raise_key_error_if_n_cluster_param_is_not_found(
        self, cs_sampler: BaseSampler, unlabeled_sentences: List[Sentence]
    ) -> None:
        """Test ClusterSimilaritySampler.kmeans function raise key_error if n_cluster parameter is not found"""
        # Arrange
        kmeans_params = {"n_init": 10, "random_state": 0}

        # Assert
        with pytest.raises(KeyError):
            # Act
            cs_sampler.kmeans(unlabeled_sentences, kmeans_params)

    def test_kmeans_return_correct_result(
        self, cs_sampler: BaseSampler, sampler_params: dict, entities6: List[Entity]
    ) -> None:
        """Test ClusterSimilaritySampler.kmeans function return correct result"""
        # Arrange
        entities = Entities()
        entities.entities = entities6

        # Act
        cluster_centers_matrix, entity_cluster_nums = cs_sampler.kmeans(
            entities.entities, sampler_params["kmeans_params"]
        )

        # Assert
        assert np.array_equal(
            cluster_centers_matrix, np.array([[10.0, 2.0], [1.0, 2.0]])
        )
        assert np.array_equal(entity_cluster_nums, np.array([1, 1, 1, 0, 0, 0]))

    def test_assign_cluster(self, cs_sampler: BaseSampler) -> None:
        """Test ClusterSimilaritySampler.assign_cluster function"""
        # Arrange
        e0 = MagicMock(label="PER", vector=torch.tensor([1, 2]))
        entities = Entities()
        entities.entities = [e0]
        entity_cluster_nums = np.array([0])

        # Act
        new_entities = cs_sampler.assign_cluster(entities, entity_cluster_nums)

        # Assert
        assert new_entities.entities[0].cluster == 0

    def test_calculate_diversity(
        self, cs_sampler: BaseSampler, entities6: List[Entity]
    ) -> None:
        """Test ClusterSimilaritySampler.calculate_diversity function"""
        # Arrange
        entities_per_cluster = {
            1: [entities6[0], entities6[1], entities6[2]],
            0: [entities6[3], entities6[4], entities6[5]],
        }
        sentence_entities = [entities6[0], entities6[3]]
        cluster_centers_matrix = np.array([[10.0, 2.0], [1.0, 2.0]])

        # Act
        sentence_score = cs_sampler.calculate_diversity(
            sentence_entities, entities_per_cluster, cluster_centers_matrix
        )

        # Assert
        np.testing.assert_allclose([sentence_score], [0.7138], rtol=1e-3)

    def test_sentence_diversity(
        self, cs_sampler: BaseSampler, entities6: List[Entity]
    ) -> None:
        """Test ClusterSimilaritySampler.sentence_diversities function"""
        # Arrange
        entities_per_cluster = {
            1: [entities6[0], entities6[1], entities6[2]],
            0: [entities6[3], entities6[4], entities6[5]],
        }
        entities = Entities()
        entities.entities = entities6
        cluster_centers_matrix = np.array([[10.0, 2.0], [1.0, 2.0]])
        entities_per_sentence = {0: [entities6[0], entities6[3]]}
        type(entities).group_by_sentence = PropertyMock(
            return_value=entities_per_sentence
        )
        type(entities).group_by_cluster = PropertyMock(
            return_value=entities_per_cluster
        )

        # Act
        sentence_score = cs_sampler.sentence_diversities(
            entities, cluster_centers_matrix
        )

        # Assert
        np.testing.assert_allclose([sentence_score[0]], [0.7138], rtol=1e-3)

    def test_score(self, cs_sampler: BaseSampler, entities6: List[Entity]) -> None:
        """Test ClusterSimilaritySampler.score function"""
        # Arrange
        sents = [0]  # Just one setnence
        kwargs = {"kmeans_params": {"n_clusters": 8, "n_init": 10, "random_state": 0}}

        cluster_centers_matrix = np.array([[10.0, 2.0], [1.0, 2.0]])
        entity_cluster_nums = np.array([1, 1, 1, 0, 0, 0])
        cs_sampler.kmeans = MagicMock(
            return_value=(cluster_centers_matrix, entity_cluster_nums)
        )

        entities = Entities()
        entities.entities = entities6
        cs_sampler.assign_cluster = MagicMock(return_value=entities)
        cs_sampler.sentence_diversities = MagicMock(return_value={0: 0.7138})

        # Act
        sentence_scores = cs_sampler.score(sents, entities, kwargs)

        # Assert
        assert np.array_equal(sentence_scores, np.array([0.7138]))

    def test_get_kmeans_params_return_normal_value(
        self, cs_sampler: BaseSampler
    ) -> None:
        """Test ClusterSimilaritySampler.get_kmeans_params return normal value."""
        # Arrange
        kwargs = {"kmeans_params": {"n_clusters": 8, "n_init": 10, "random_state": 0}}

        # Act
        kmeans_params = cs_sampler.get_kmeans_params(kwargs)

        # Assert
        assert kmeans_params == kwargs["kmeans_params"]

    def test_get_kmeans_params_return_raise_name_error(
        self, cs_sampler: BaseSampler
    ) -> None:
        """Test ClusterSimilaritySampler.get_kmeans_params raise error if parameters are inconpatible"""
        # Arrange
        kwargs = {}

        # Assert
        with pytest.raises(NameError):
            # Act
            cs_sampler.get_kmeans_params(kwargs)


class TestCombinedMultipleSampler:
    """Test CombinedMultipleSampler class"""

    def test_get_sampler_type_return_default_value(
        self, cm_sampler: BaseSampler
    ) -> None:
        """Test CombinedMultipleSampler.get_sampler_type return default value"""
        # Arrange
        kwargs = {}

        # Act
        sampler_type = cm_sampler.get_sampler_type(kwargs)

        # Assert
        assert sampler_type == "lc_ds"

    def test_get_sampler_type_return_normal_value(
        self, cm_sampler: BaseSampler
    ) -> None:
        """Test CombinedMultipleSampler.get_sampler_type return normal value"""
        # Arrange
        kwargs = {"sampler_type": "mnlp_ds"}

        # Act
        sampler_type = cm_sampler.get_sampler_type(kwargs)

        # Assert
        assert sampler_type == "mnlp_ds"

    def test_get_sampler_type_return_raise_name_error(
        self, cm_sampler: BaseSampler
    ) -> None:
        """Test CombinedMultipleSampler.get_sampler_type raise error if parameters are inconpatible"""
        # Arrange
        kwargs = {"sampler_type": "lcc_ds"}

        # Assert
        with pytest.raises(NameError):
            # Act
            cm_sampler.get_sampler_type(kwargs)

    def test_get_combined_type_return_default_value(
        self, cm_sampler: BaseSampler
    ) -> None:
        """Test CombinedMultipleSampler.get_combined_type return default value"""

        # Arrange
        kwargs = {}

        # Act
        combined_type = cm_sampler.get_combined_type(kwargs)

        # Assert
        assert combined_type == "parallel"

    def test_get_combined_type_return_normal_value(
        self, cm_sampler: BaseSampler
    ) -> None:
        """Test CombinedMultipleSampler.get_combined_type return normal value"""

        # Arrange
        kwargs = {"combined_type": "series"}

        # Act
        combined_type = cm_sampler.get_combined_type(kwargs)

        # Assert
        assert combined_type == "series"

    def test_get_scaler_return_default_value(self, cm_sampler: BaseSampler) -> None:
        """Test CombinedMultipleSampler.get_scaler return default value"""

        # Arrange
        kwargs = {}

        # Act
        scaler = cm_sampler.get_scaler(kwargs)

        # Assert
        assert isinstance(scaler, MinMaxScaler) is True

    def test_get_scaler_return_normal_value(self, cm_sampler: BaseSampler) -> None:
        """Test CombinedMultipleSampler.get_scaler return normal value"""

        # Arrange
        kwargs = {"scaler": MinMaxScaler()}

        # Act
        scaler = cm_sampler.get_scaler(kwargs)

        # Assert
        assert isinstance(scaler, MinMaxScaler) is True

    def test_check_combined_type_return_raise_name_error(
        self, cm_sampler: BaseSampler
    ) -> None:
        """Test CombinedMultipleSampler.get_combined_type raise error if parameters are inconpatible"""

        # Arrange
        kwargs = {"sampler_type": "lc_ds", "combined_type": "mix"}

        # Assert
        with pytest.raises(NameError):
            # Act
            cm_sampler.get_combined_type(kwargs)

    def test_get_samplers_with_lc_ds(self, cm_sampler: BaseSampler) -> None:
        """Test CombinedMultipleSampler.get_samplers for lc_ds samples"""

        # Arrange
        sampler_type = "lc_ds"

        # Act
        uncertainty_sampler, diversity_sampler = cm_sampler.get_samplers(sampler_type)

        # Assert
        assert isinstance(uncertainty_sampler, LeastConfidenceSampler)
        assert isinstance(diversity_sampler, DistributeSimilaritySampler)

    def test_get_samplers_with_lc_cs(self, cm_sampler: BaseSampler) -> None:
        """Test CombinedMultipleSampler.get_samplers for lc_cs samples"""

        # Arrange
        sampler_type = "lc_cs"

        # Act
        uncertainty_sampler, diversity_sampler = cm_sampler.get_samplers(sampler_type)

        # Assert
        assert isinstance(uncertainty_sampler, LeastConfidenceSampler)
        assert isinstance(diversity_sampler, ClusterSimilaritySampler)

    def test_get_samplers_with_mnlp_ds(self, cm_sampler: BaseSampler) -> None:
        """Test CombinedMultipleSampler.get_samplers for mnlp_ds samples"""

        # Arrange
        sampler_type = "mnlp_ds"

        # Act
        uncertainty_sampler, diversity_sampler = cm_sampler.get_samplers(sampler_type)

        # Assert
        assert isinstance(uncertainty_sampler, MaxNormLogProbSampler)
        assert isinstance(diversity_sampler, DistributeSimilaritySampler)

    def test_get_samplers_with_mnlp_cs(self, cm_sampler: BaseSampler) -> None:
        """Test CombinedMultipleSampler.get_samplers for mnlp_cs samples"""

        # Arrange
        sampler_type = "mnlp_cs"

        # Act
        uncertainty_sampler, diversity_sampler = cm_sampler.get_samplers(sampler_type)

        # Assert
        assert isinstance(uncertainty_sampler, MaxNormLogProbSampler)
        assert isinstance(diversity_sampler, ClusterSimilaritySampler)

    @pytest.mark.parametrize("scaler", [MinMaxScaler(), StandardScaler()])
    def test_normalize_samplers_by_scaler(
        self, cm_sampler: BaseSampler, scaler: BaseEstimator
    ) -> None:
        """Test CombinedMultipleSampler.normalize_scores by scaler"""

        # Arrange
        uncertainty_scores = np.array([-0.09, -0.07, -0.05, -0.03, -0.01])
        diversity_scores = np.array([0.2, 0.4, 0.6, 0.8, 1])
        concatenate_scores = np.stack([uncertainty_scores, diversity_scores])
        normalized_scores = scaler.fit_transform(np.transpose(concatenate_scores))
        expected_scores = normalized_scores.sum(axis=1)

        # Act
        scores = cm_sampler.normalize_scores(
            uncertainty_scores, diversity_scores, scaler
        )

        # Assert
        assert np.allclose(scores, expected_scores) is True

    def test_call_return_correct_result_with_series_lc_ds(
        self,
        cm_sampler: BaseSampler,
        lc_sampler: BaseSampler,
        ds_sampler: BaseSampler,
        unlabeled_sentences: List[Sentence],
        sampler_params: dict,
    ) -> None:
        """Test CombinedMultipleSampler call function return correct result with lc_ds"""

        # Arrange
        lc_sampler.predict = MagicMock(return_value=None)
        lc_sampler.score = MagicMock(
            return_value=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        )

        entities = Entities()
        entities.entities = [None]
        ds_sampler.get_entities = MagicMock(return_value=entities)
        ds_sampler.score = MagicMock(
            return_value=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        )

        sampler_type = "lc_ds"
        combined_type = "series"
        cm_sampler.get_samplers = MagicMock(return_value=(lc_sampler, ds_sampler))

        # Act
        queried_sent_ids = cm_sampler(
            unlabeled_sentences,
            sampler_params["tag_type"],
            sampler_params["query_number"],
            sampler_params["token_based"],
            tagger=sampler_params["tagger"],
            label_names=sampler_params["label_names"],
            embeddings=sampler_params["embeddings"],
            sampler_type=sampler_type,
            combined_type=combined_type,
        )

        # Assert
        assert queried_sent_ids == [7, 6, 5, 4]

    def test_call_return_random_sent_ids_if_entities_is_empty(
        self,
        cm_sampler: BaseSampler,
        lc_sampler: BaseSampler,
        ds_sampler: BaseSampler,
        unlabeled_sentences: List[Sentence],
        sampler_params: dict,
    ) -> None:
        """Test CombinedMultipleSampler call function return random sentence ids if entities is empty"""
        # Arrange
        sampler_type = "lc_ds"
        combined_type = "parallel"
        entities = Entities()
        cm_sampler.predict = MagicMock(return_value=None)
        cm_sampler.get_entities = MagicMock(return_value=entities)

        random.seed(0)
        sent_ids = list(range(len(unlabeled_sentences)))
        expected_random_sent_ids = random.sample(sent_ids, len(sent_ids))

        # Act
        queried_sent_ids = cm_sampler(
            unlabeled_sentences,
            sampler_params["tag_type"],
            sampler_params["query_number"],
            sampler_params["token_based"],
            tagger=sampler_params["tagger"],
            embeddings=sampler_params["embeddings"],
            kmeans_params=sampler_params["kmeans_params"],
            sampler_type=sampler_type,
            combined_type=combined_type,
        )

        # Assert
        assert (
            queried_sent_ids
            == expected_random_sent_ids[: sampler_params["query_number"]]
        )

    def test_call_return_correct_result_with_parallel_lc_ds(
        self,
        cm_sampler: BaseSampler,
        lc_sampler: BaseSampler,
        ds_sampler: BaseSampler,
        unlabeled_sentences: List[Sentence],
        sampler_params: dict,
    ) -> None:
        """Test CombinedMultipleSampler call function return correct result with parallel lc_ds"""

        # Arrange
        sampler_type = "lc_ds"
        combined_type = "parallel"
        cm_sampler.predict = MagicMock(return_value=[None])
        entities = Entities()
        entities.entities = [None]
        cm_sampler.get_entities = MagicMock(return_value=entities)

        lc_sampler.score = MagicMock(
            return_value=np.array([0.09, 0.07, 0.05, 0.03, 0.01])
        )
        ds_sampler.score = MagicMock(return_value=np.array([0.2, 0.4, 0.6, 0.8, 1]))
        cm_sampler.get_samplers = MagicMock(return_value=(lc_sampler, ds_sampler))
        # normalized_scores = array([2. , 1.5, 1. , 0.5, 0. ])

        # Act
        queried_sent_ids = cm_sampler(
            unlabeled_sentences,
            sampler_params["tag_type"],
            sampler_params["query_number"],
            sampler_params["token_based"],
            tagger=sampler_params["tagger"],
            label_names=sampler_params["label_names"],
            embeddings=sampler_params["embeddings"],
            sampler_type=sampler_type,
            combined_type=combined_type,
        )

        # Assert
        assert queried_sent_ids == [0, 1, 2, 3]

    def test_call_return_correct_result_with_parallel_lc_cs(
        self,
        cm_sampler: BaseSampler,
        lc_sampler: BaseSampler,
        cs_sampler: BaseSampler,
        unlabeled_sentences: List[Sentence],
        sampler_params: dict,
    ) -> None:
        """Test CombinedMultipleSampler call function return correct result with parallel lc_cs"""

        # Arrange
        sampler_type = "lc_ds"
        combined_type = "parallel"
        cm_sampler.predict = MagicMock(return_value=[None])
        entities = Entities()
        entities.entities = [None]
        cm_sampler.get_entities = MagicMock(return_value=entities)

        lc_sampler.score = MagicMock(
            return_value=np.array([0.09, 0.07, 0.05, 0.03, 0.01])
        )
        cs_sampler.score = MagicMock(return_value=np.array([0.2, 0.4, 0.6, 0.8, 1]))
        cm_sampler.get_samplers = MagicMock(return_value=(lc_sampler, cs_sampler))
        # normalized_scores = array([2. , 1.5, 1. , 0.5, 0. ])

        # Act
        queried_sent_ids = cm_sampler(
            unlabeled_sentences,
            sampler_params["tag_type"],
            sampler_params["query_number"],
            sampler_params["token_based"],
            tagger=sampler_params["tagger"],
            label_names=sampler_params["label_names"],
            embeddings=sampler_params["embeddings"],
            kmeans_params=sampler_params["kmeans_params"],
            sampler_type=sampler_type,
            combined_type=combined_type,
        )

        # Assert
        assert queried_sent_ids == [0, 1, 2, 3]

    def test_call_return_correct_result_with_parallel_mnlp_ds(
        self,
        cm_sampler: BaseSampler,
        mnlp_sampler: BaseSampler,
        ds_sampler: BaseSampler,
        unlabeled_sentences: List[Sentence],
        sampler_params: dict,
    ) -> None:
        """Test CombinedMultipleSampler call function return correct result with parallel mnlp_ds"""

        # Arrange
        sampler_type = "mnlp_ds"
        combined_type = "parallel"
        cm_sampler.predict = MagicMock(return_value=[None])
        entities = Entities()
        entities.entities = [None]
        cm_sampler.get_entities = MagicMock(return_value=entities)

        mnlp_sampler.score = MagicMock(
            return_value=np.array([-0.09, -0.07, -0.05, -0.03, -0.01])
        )
        ds_sampler.score = MagicMock(return_value=np.array([0.2, 0.4, 0.6, 0.8, 1]))
        cm_sampler.get_samplers = MagicMock(return_value=(mnlp_sampler, ds_sampler))
        # normalized_scores = array([2. , 1.5, 1. , 0.5, 0. ])

        # Act
        queried_sent_ids = cm_sampler(
            unlabeled_sentences,
            sampler_params["tag_type"],
            sampler_params["query_number"],
            sampler_params["token_based"],
            tagger=sampler_params["tagger"],
            label_names=sampler_params["label_names"],
            embeddings=sampler_params["embeddings"],
            sampler_type=sampler_type,
            combined_type=combined_type,
        )

        # Assert
        assert queried_sent_ids == [0, 1, 2, 3]

    def test_call_return_correct_result_with_parallel_mnlp_cs(
        self,
        cm_sampler: BaseSampler,
        mnlp_sampler: BaseSampler,
        cs_sampler: BaseSampler,
        unlabeled_sentences: List[Sentence],
        sampler_params: dict,
    ) -> None:
        """Test CombinedMultipleSampler call function return correct result with parallel mnlp_cs"""

        # Arrange
        sampler_type = "mnlp_cs"
        combined_type = "parallel"
        cm_sampler.predict = MagicMock(return_value=[None])
        entities = Entities()
        entities.entities = [None]
        cm_sampler.get_entities = MagicMock(return_value=entities)

        mnlp_sampler.score = MagicMock(
            return_value=np.array([-0.09, -0.07, -0.05, -0.03, -0.01])
        )
        cs_sampler.score = MagicMock(return_value=np.array([0.2, 0.4, 0.6, 0.8, 1]))
        cm_sampler.get_samplers = MagicMock(return_value=(mnlp_sampler, cs_sampler))
        # normalized_scores = array([2. , 1.5, 1. , 0.5, 0. ])

        # Act
        queried_sent_ids = cm_sampler(
            unlabeled_sentences,
            sampler_params["tag_type"],
            sampler_params["query_number"],
            sampler_params["token_based"],
            tagger=sampler_params["tagger"],
            label_names=sampler_params["label_names"],
            embeddings=sampler_params["embeddings"],
            sampler_type=sampler_type,
            combined_type=combined_type,
        )

        # Assert
        assert queried_sent_ids == [0, 1, 2, 3]
