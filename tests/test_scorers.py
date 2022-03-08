import random
from typing import Dict, List
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
import torch
from flair.data import Sentence
from sklearn.preprocessing import MinMaxScaler

from seqal.base_scorer import BaseScorer
from seqal.data import Entities, Entity
from seqal.scorers import (
    ClusterSimilarityScorer,
    CombinedMultipleScorer,
    DistributeSimilarityScorer,
    LeastConfidenceScorer,
    MaxNormLogProbScorer,
    RandomScorer,
    StringNGramScorer,
)


@pytest.fixture()
def lc_scorer(scope="function"):
    """LeastConfidenceScorer instance"""
    lc_scorer = LeastConfidenceScorer()
    return lc_scorer


@pytest.fixture()
def mnlp_scorer(scope="function"):
    """MaxNormLogProbScorer instance"""
    mnlp_scorer = MaxNormLogProbScorer()
    return mnlp_scorer


@pytest.fixture()
def sn_scorer(scope="function"):
    """StringNGramScorer instance"""
    sn_scorer = StringNGramScorer()
    return sn_scorer


@pytest.fixture()
def ds_scorer(scope="function"):
    """DistributeSimilarityScorer instance"""
    ds_scorer = DistributeSimilarityScorer()
    return ds_scorer


@pytest.fixture()
def cs_scorer(scope="function"):
    """ClusterSimilarityScorer instance"""
    cs_scorer = ClusterSimilarityScorer()
    return cs_scorer


@pytest.fixture()
def cm_scorer(scope="function"):
    """CombinedMultipleScorer instance"""
    cm_scorer = CombinedMultipleScorer()
    return cm_scorer


@pytest.fixture()
def scorer_params(scope="function"):
    """Common parameters for scorer test"""
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
    """4 entities for ds_scorer test"""
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
def similarity_matrix_per_label_cosine_n_gram(scope="function"):
    """Similarity matrix for each label"""
    similarity_matrix_per_label_cosine_n_gram = {
        "PER": np.array(
            [
                [1.4000, 0.547722557505, 0.0000],
                [0.547722557505, 1.333333333333, 0.0000],
                [0.0000, 0.0000, 1.4000],
            ]
        ),
        "LOC": np.array([[1.666666666667]]),
    }

    return similarity_matrix_per_label_cosine_n_gram


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
    """4 entities for ds_scorer test"""
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


class TestRandomScorer:
    """Test RandomScorer class"""

    def test_call_return_random_sent_ids(
        self,
        unlabeled_sentences: List[Sentence],
        scorer_params: dict,
    ) -> None:
        """Test call function return random sentence ids if entities is empty"""
        # Arrange
        random_scorer = RandomScorer()

        random.seed(0)
        sent_ids = list(range(len(unlabeled_sentences)))
        expected_random_sent_ids = random.sample(sent_ids, len(sent_ids))

        # Act

        queried_sent_ids = random_scorer(
            unlabeled_sentences,
            scorer_params["tag_type"],
            scorer_params["query_number"],
            scorer_params["token_based"],
            tagger=scorer_params["tagger"],
            label_names=scorer_params["label_names"],
            embeddings=scorer_params["embeddings"],
        )

        # Assert
        assert (
            queried_sent_ids
            == expected_random_sent_ids[: scorer_params["query_number"]]
        )


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
        self,
        lc_scorer: BaseScorer,
        unlabeled_sentences: List[Sentence],
        scorer_params: dict,
    ):
        """Test call function return correct result"""
        # Arrange
        lc_scorer.predict = MagicMock(return_value=None)
        lc_scorer.score = MagicMock(
            return_value=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        )

        # Act
        queried_sent_ids = lc_scorer(
            unlabeled_sentences,
            scorer_params["tag_type"],
            scorer_params["query_number"],
            scorer_params["token_based"],
            tagger=scorer_params["tagger"],
            embeddings=scorer_params["embeddings"],
            label_names=scorer_params["label_names"],
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
        self,
        mnlp_scorer: BaseScorer,
        unlabeled_sentences: List[Sentence],
        scorer_params: dict,
    ):
        """Test call function return correct result"""
        # Arrange
        mnlp_scorer.predict = MagicMock(return_value=None)
        returned_norm_log_probs = np.array(
            [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05]
        )
        mnlp_scorer.score = MagicMock(return_value=returned_norm_log_probs)

        # Act
        queried_sent_ids = mnlp_scorer(
            unlabeled_sentences,
            scorer_params["tag_type"],
            scorer_params["query_number"],
            scorer_params["token_based"],
            tagger=scorer_params["tagger"],
            label_names=scorer_params["label_names"],
            embeddings=scorer_params["embeddings"],
        )

        # Assert
        assert queried_sent_ids == [0, 1, 2, 3]


class TestStringNGramScorer:
    """Test StringNGramScorer class"""

    def test_call_return_correct_result(
        self,
        sn_scorer: BaseScorer,
        unlabeled_sentences: List[Sentence],
        scorer_params: dict,
    ) -> None:
        """Test call function return correct result"""
        # Arrange
        entities = Entities()
        entities.entities = [None]
        sn_scorer.get_entities = MagicMock(return_value=entities)
        sn_scorer.score = MagicMock(
            return_value=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        )

        # Act
        queried_sent_ids = sn_scorer(
            unlabeled_sentences,
            scorer_params["tag_type"],
            scorer_params["query_number"],
            scorer_params["token_based"],
            tagger=scorer_params["tagger"],
            label_names=scorer_params["label_names"],
            embeddings=scorer_params["embeddings"],
        )

        # Assert
        assert queried_sent_ids == [9, 8, 7, 6]

    def test_call_return_random_sent_ids_if_entities_is_empty(
        self,
        sn_scorer: BaseScorer,
        unlabeled_sentences: List[Sentence],
        scorer_params: dict,
    ) -> None:
        """Test call function return random sentence ids if entities is empty"""
        # Arrange
        entities = Entities()
        sn_scorer.get_entities = MagicMock(return_value=entities)

        random.seed(0)
        sent_ids = list(range(len(unlabeled_sentences)))
        expected_random_sent_ids = random.sample(sent_ids, len(sent_ids))

        # Act
        queried_sent_ids = sn_scorer(
            unlabeled_sentences,
            scorer_params["tag_type"],
            scorer_params["query_number"],
            scorer_params["token_based"],
            tagger=scorer_params["tagger"],
            label_names=scorer_params["label_names"],
            embeddings=scorer_params["embeddings"],
        )

        # Assert
        assert (
            queried_sent_ids
            == expected_random_sent_ids[: scorer_params["query_number"]]
        )

    def test_n_gram(self, sn_scorer: BaseScorer, entities4: List[Entity]) -> None:
        """Test n_gram function"""
        # Act
        n_grams = sn_scorer.n_gram(entities4[0])

        # Assert
        assert n_grams == ["$$P", "$Pe", "Pet", "ete", "ter", "er$", "r$$"]

    def test_calculate_diversity(
        self,
        sn_scorer: BaseScorer,
        entities_per_sentence: dict,
        entity_id_map: dict,
        similarity_matrix_per_label: dict,
    ) -> None:
        """Test calculate diversity function"""
        # Arrange
        expected = {0: 0, 1: -0.5}

        # Act
        sentence_scores = sn_scorer.calculate_diversity(
            entities_per_sentence, entity_id_map, similarity_matrix_per_label
        )

        # Assert
        assert sentence_scores == expected

    def test_sentence_diversity(
        self, sn_scorer: BaseScorer, entities4: List[Entity]
    ) -> None:
        """Test sentence_diversity function"""
        # Arrange
        entities = Entities()
        entities.entities = entities4
        expected = {0: 0.833333333333, 1: 0}

        # Act
        sentence_scores = sn_scorer.sentence_diversities(entities)

        # Assert
        assert compare_approximate(sentence_scores, expected) is True

    def test_similarity_matrix_per_label(
        self,
        sn_scorer: BaseScorer,
        entities_per_label: dict,
        similarity_matrix_per_label_cosine_n_gram: Dict[str, torch.Tensor],
    ) -> None:
        """Test similarity_matrix_per_label function"""
        # Act
        sentence_scores = sn_scorer.similarity_matrix_per_label(entities_per_label)

        # Assert
        assert (
            compare_approximate(
                sentence_scores, similarity_matrix_per_label_cosine_n_gram
            )
            is True
        )

    def test_get_entity_id_map(
        self,
        sn_scorer: BaseScorer,
        entities_per_sentence: dict,
        entities_per_label: dict,
        entity_id_map: dict,
    ) -> None:
        """Test get_entity_id_map function"""
        # Arrange
        sentence_count = max(entities_per_sentence.keys()) + 1
        max_entity_count = max(
            [len(entities) for entities in entities_per_sentence.values()]
        )

        # Act
        entity_id_map_result = sn_scorer.get_entity_id_map(
            entities_per_label, sentence_count, max_entity_count
        )

        # Assert
        assert compare_exact(entity_id_map_result, entity_id_map) is True

    def test_score(self, sn_scorer: BaseScorer) -> None:
        """Test score function"""
        # Arrange
        sents = [0, 1]
        entities = Entities()
        sn_scorer.sentence_diversities = MagicMock(return_value={0: 0, 1: -0.5})

        # Act
        sentence_scores = sn_scorer.score(sents, entities)

        # Assert
        assert np.array_equal(sentence_scores, np.array([0, -0.5]))


class TestDistributeSimilarityScorer:
    """Test DistributeSimilarityScorer class"""

    def test_call_return_correct_result(
        self,
        ds_scorer: BaseScorer,
        unlabeled_sentences: List[Sentence],
        scorer_params: dict,
    ) -> None:
        """Test call function return correct result"""
        # Arrange
        entities = Entities()
        entities.entities = [None]
        ds_scorer.get_entities = MagicMock(return_value=entities)
        ds_scorer.score = MagicMock(
            return_value=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        )

        # Act
        queried_sent_ids = ds_scorer(
            unlabeled_sentences,
            scorer_params["tag_type"],
            scorer_params["query_number"],
            scorer_params["token_based"],
            tagger=scorer_params["tagger"],
            label_names=scorer_params["label_names"],
            embeddings=scorer_params["embeddings"],
        )

        # Assert
        assert queried_sent_ids == [9, 8, 7, 6]

    def test_call_return_random_sent_ids_if_entities_is_empty(
        self,
        ds_scorer: BaseScorer,
        unlabeled_sentences: List[Sentence],
        scorer_params: dict,
    ) -> None:
        """Test call function return random sentence ids if entities is empty"""
        # Arrange
        entities = Entities()
        ds_scorer.get_entities = MagicMock(return_value=entities)

        random.seed(0)
        sent_ids = list(range(len(unlabeled_sentences)))
        expected_random_sent_ids = random.sample(sent_ids, len(sent_ids))

        # Act
        queried_sent_ids = ds_scorer(
            unlabeled_sentences,
            scorer_params["tag_type"],
            scorer_params["query_number"],
            scorer_params["token_based"],
            tagger=scorer_params["tagger"],
            label_names=scorer_params["label_names"],
            embeddings=scorer_params["embeddings"],
        )

        # Assert
        assert (
            queried_sent_ids
            == expected_random_sent_ids[: scorer_params["query_number"]]
        )

    def test_calculate_diversity(
        self,
        ds_scorer: BaseScorer,
        entities_per_sentence: dict,
        entity_id_map: dict,
        similarity_matrix_per_label: dict,
    ) -> None:
        """Test calculate diversity function"""
        # Arrange
        expected = {0: 0, 1: -0.5}

        # Act
        sentence_scores = ds_scorer.calculate_diversity(
            entities_per_sentence, entity_id_map, similarity_matrix_per_label
        )

        # Assert
        assert sentence_scores == expected

    def test_sentence_diversity(
        self, ds_scorer: BaseScorer, entities4: List[Entity]
    ) -> None:
        """Test sentence_diversity function"""
        # Arrange
        entities = Entities()
        entities.entities = entities4
        expected = {0: 0, 1: -0.5}

        # Act
        sentence_scores = ds_scorer.sentence_diversities(entities)

        # Assert
        assert compare_approximate(sentence_scores, expected) is True

    def test_similarity_matrix_per_label(
        self,
        ds_scorer: BaseScorer,
        entities_per_label: dict,
        similarity_matrix_per_label: Dict[str, torch.Tensor],
    ) -> None:
        """Test similarity_matrix_per_label function"""
        # Act
        sentence_scores = ds_scorer.similarity_matrix_per_label(entities_per_label)

        # Assert
        assert compare_approximate(sentence_scores, similarity_matrix_per_label) is True

    def test_get_entity_id_map(
        self,
        ds_scorer: BaseScorer,
        entities_per_sentence: dict,
        entities_per_label: dict,
        entity_id_map: dict,
    ) -> None:
        """Test get_entity_id_map function"""
        # Arrange
        sentence_count = max(entities_per_sentence.keys()) + 1
        max_entity_count = max(
            [len(entities) for entities in entities_per_sentence.values()]
        )

        # Act
        entity_id_map_result = ds_scorer.get_entity_id_map(
            entities_per_label, sentence_count, max_entity_count
        )

        # Assert
        assert compare_exact(entity_id_map_result, entity_id_map) is True

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
        self,
        cs_scorer: BaseScorer,
        unlabeled_sentences: List[Sentence],
        scorer_params: dict,
    ) -> None:
        """Test call function return correct result"""
        # Arrange
        entities = Entities()
        entities.entities = [None]
        cs_scorer.get_entities = MagicMock(return_value=entities)
        cs_scorer.score = MagicMock(
            return_value=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        )

        # Act
        queried_sent_ids = cs_scorer(
            unlabeled_sentences,
            scorer_params["tag_type"],
            scorer_params["query_number"],
            scorer_params["token_based"],
            tagger=scorer_params["tagger"],
            embeddings=scorer_params["embeddings"],
            kmeans_params=scorer_params["kmeans_params"],
        )

        # Assert
        assert queried_sent_ids == [9, 8, 7, 6]

    def test_call_return_random_sent_ids_if_entities_is_empty(
        self,
        cs_scorer: BaseScorer,
        unlabeled_sentences: List[Sentence],
        scorer_params: dict,
    ) -> None:
        """Test call function return random sentence ids if entities is empty"""
        # Arrange
        entities = Entities()
        cs_scorer.get_entities = MagicMock(return_value=entities)

        random.seed(0)
        sent_ids = list(range(len(unlabeled_sentences)))
        expected_random_sent_ids = random.sample(sent_ids, len(sent_ids))

        # Act
        queried_sent_ids = cs_scorer(
            unlabeled_sentences,
            scorer_params["tag_type"],
            scorer_params["query_number"],
            scorer_params["token_based"],
            tagger=scorer_params["tagger"],
            embeddings=scorer_params["embeddings"],
            kmeans_params=scorer_params["kmeans_params"],
        )

        # Assert
        assert (
            queried_sent_ids
            == expected_random_sent_ids[: scorer_params["query_number"]]
        )

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
        self, cs_scorer: BaseScorer, scorer_params: dict, entities6: List[Entity]
    ) -> None:
        """Test kmeans function return correct result"""
        # Arrange
        entities = Entities()
        entities.entities = entities6

        # Act
        cluster_centers_matrix, entity_cluster_nums = cs_scorer.kmeans(
            entities.entities, scorer_params["kmeans_params"]
        )

        # Assert
        assert np.array_equal(
            cluster_centers_matrix, np.array([[10.0, 2.0], [1.0, 2.0]])
        )
        assert np.array_equal(entity_cluster_nums, np.array([1, 1, 1, 0, 0, 0]))

    def test_assign_cluster(self, cs_scorer: BaseScorer) -> None:
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

    def test_calculate_diversity(
        self, cs_scorer: BaseScorer, entities6: List[Entity]
    ) -> None:
        """Test calculate diversity function"""
        # Arrange
        entities_per_cluster = {
            1: [entities6[0], entities6[1], entities6[2]],
            0: [entities6[3], entities6[4], entities6[5]],
        }
        sentence_entities = [entities6[0], entities6[3]]
        cluster_centers_matrix = np.array([[10.0, 2.0], [1.0, 2.0]])

        # Act
        sentence_score = cs_scorer.calculate_diversity(
            sentence_entities, entities_per_cluster, cluster_centers_matrix
        )

        # Assert
        np.testing.assert_allclose([sentence_score], [0.7138], rtol=1e-3)

    def test_sentence_diversity(
        self, cs_scorer: BaseScorer, entities6: List[Entity]
    ) -> None:
        """Test sentence diversity function"""
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
        sentence_score = cs_scorer.sentence_diversities(
            entities, cluster_centers_matrix
        )

        # Assert
        np.testing.assert_allclose([sentence_score[0]], [0.7138], rtol=1e-3)

    def test_score(self, cs_scorer: BaseScorer, entities6: List[Entity]) -> None:
        """Test score function"""
        # Arrange
        sents = [0]  # Just one setnence
        kmeans_params = {"n_cluster": 8, "n_init": 10, "random_state": 0}

        cluster_centers_matrix = np.array([[10.0, 2.0], [1.0, 2.0]])
        entity_cluster_nums = np.array([1, 1, 1, 0, 0, 0])
        cs_scorer.kmeans = MagicMock(
            return_value=(cluster_centers_matrix, entity_cluster_nums)
        )

        entities = Entities()
        entities.entities = entities6
        cs_scorer.assign_cluster = MagicMock(return_value=entities)
        cs_scorer.sentence_diversities = MagicMock(return_value={0: 0.7138})

        # Act
        sentence_scores = cs_scorer.score(sents, entities, kmeans_params)

        # Assert
        assert np.array_equal(sentence_scores, np.array([0.7138]))


class TestCombinedMultipleScorer:
    """Test CombinedMultipleScorer class"""

    def test_get_scorer_type_return_default_value(self, cm_scorer: BaseScorer) -> None:
        # Arrange
        kwargs = {}

        # Act
        scorer_type = cm_scorer.get_scorer_type(kwargs)

        # Assert
        assert scorer_type == "lc_ds"

    def test_get_scorer_type_return_normal_value(self, cm_scorer: BaseScorer) -> None:
        # Arrange
        kwargs = {"scorer_type": "mnlp_ds"}

        # Act
        scorer_type = cm_scorer.get_scorer_type(kwargs)

        # Assert
        assert scorer_type == "mnlp_ds"

    def test_get_scorer_type_return_raise_name_error(
        self, cm_scorer: BaseScorer
    ) -> None:
        # Arrange
        kwargs = {"scorer_type": "lcc_ds"}

        # Assert
        with pytest.raises(NameError):
            # Act
            cm_scorer.get_scorer_type(kwargs)

    def test_get_combined_type_return_default_value(
        self, cm_scorer: BaseScorer
    ) -> None:
        # Arrange
        kwargs = {}

        # Act
        combined_type = cm_scorer.get_combined_type(kwargs)

        # Assert
        assert combined_type == "parallel"

    def test_get_combined_type_return_normal_value(self, cm_scorer: BaseScorer) -> None:
        # Arrange
        kwargs = {"combined_type": "series"}

        # Act
        combined_type = cm_scorer.get_combined_type(kwargs)

        # Assert
        assert combined_type == "series"

    def test_check_combined_type_return_raise_name_error(
        self, cm_scorer: BaseScorer
    ) -> None:
        # Arrange
        kwargs = {"scorer_type": "lc_ds", "combined_type": "mix"}

        # Assert
        with pytest.raises(NameError):
            # Act
            cm_scorer.get_combined_type(kwargs)

    def test_get_scorers_with_lc_ds(self, cm_scorer: BaseScorer) -> None:
        # Arrange
        scorer_type = "lc_ds"

        # Act
        uncertainty_scorer, diversity_scorer = cm_scorer.get_scorers(scorer_type)

        # Assert
        assert isinstance(uncertainty_scorer, LeastConfidenceScorer)
        assert isinstance(diversity_scorer, DistributeSimilarityScorer)

    def test_get_scorers_with_lc_cs(self, cm_scorer: BaseScorer) -> None:
        # Arrange
        scorer_type = "lc_cs"

        # Act
        uncertainty_scorer, diversity_scorer = cm_scorer.get_scorers(scorer_type)

        # Assert
        assert isinstance(uncertainty_scorer, LeastConfidenceScorer)
        assert isinstance(diversity_scorer, ClusterSimilarityScorer)

    def test_get_scorers_with_mnlp_ds(self, cm_scorer: BaseScorer) -> None:
        # Arrange
        scorer_type = "mnlp_ds"

        # Act
        uncertainty_scorer, diversity_scorer = cm_scorer.get_scorers(scorer_type)

        # Assert
        assert isinstance(uncertainty_scorer, MaxNormLogProbScorer)
        assert isinstance(diversity_scorer, DistributeSimilarityScorer)

    def test_get_scorers_with_mnlp_cs(self, cm_scorer: BaseScorer) -> None:
        # Arrange
        scorer_type = "mnlp_cs"

        # Act
        uncertainty_scorer, diversity_scorer = cm_scorer.get_scorers(scorer_type)

        # Assert
        assert isinstance(uncertainty_scorer, MaxNormLogProbScorer)
        assert isinstance(diversity_scorer, ClusterSimilarityScorer)

    def test_normalize_scorers_by_min_max_scaler(self, cm_scorer: BaseScorer) -> None:
        # Arrange
        scaler = MinMaxScaler()
        uncertainty_scores = np.array([-0.09, -0.07, -0.05, -0.03, -0.01])
        diversity_scores = np.array([0.2, 0.4, 0.6, 0.8, 1])
        concatenate_scores = np.stack([uncertainty_scores, diversity_scores])
        normalized_scores = scaler.fit_transform(np.transpose(concatenate_scores))
        expected_scores = normalized_scores.sum(axis=1)

        # Act
        scores = cm_scorer.normalize_scores(uncertainty_scores, diversity_scores)

        # Assert
        assert np.allclose(scores, expected_scores) is True

    def test_call_return_correct_result_with_series_lc_ds(
        self,
        cm_scorer: BaseScorer,
        lc_scorer: BaseScorer,
        ds_scorer: BaseScorer,
        unlabeled_sentences: List[Sentence],
        scorer_params: dict,
    ) -> None:
        # Arrange
        lc_scorer.predict = MagicMock(return_value=None)
        lc_scorer.score = MagicMock(
            return_value=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        )

        entities = Entities()
        entities.entities = [None]
        ds_scorer.get_entities = MagicMock(return_value=entities)
        ds_scorer.score = MagicMock(
            return_value=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        )

        scorer_type = "lc_ds"
        combined_type = "series"
        cm_scorer.get_scorers = MagicMock(return_value=(lc_scorer, ds_scorer))

        # Act
        queried_sent_ids = cm_scorer(
            unlabeled_sentences,
            scorer_params["tag_type"],
            scorer_params["query_number"],
            scorer_params["token_based"],
            tagger=scorer_params["tagger"],
            label_names=scorer_params["label_names"],
            embeddings=scorer_params["embeddings"],
            scorer_type=scorer_type,
            combined_type=combined_type,
        )

        # Assert
        assert queried_sent_ids == [7, 6, 5, 4]

    def test_call_return_random_sent_ids_if_entities_is_empty(
        self,
        cm_scorer: BaseScorer,
        lc_scorer: BaseScorer,
        ds_scorer: BaseScorer,
        unlabeled_sentences: List[Sentence],
        scorer_params: dict,
    ) -> None:
        """Test call function return random sentence ids if entities is empty"""
        # Arrange
        scorer_type = "lc_ds"
        combined_type = "parallel"
        entities = Entities()
        cm_scorer.predict = MagicMock(return_value=None)
        cm_scorer.get_entities = MagicMock(return_value=entities)

        random.seed(0)
        sent_ids = list(range(len(unlabeled_sentences)))
        expected_random_sent_ids = random.sample(sent_ids, len(sent_ids))

        # Act
        queried_sent_ids = cm_scorer(
            unlabeled_sentences,
            scorer_params["tag_type"],
            scorer_params["query_number"],
            scorer_params["token_based"],
            tagger=scorer_params["tagger"],
            embeddings=scorer_params["embeddings"],
            kmeans_params=scorer_params["kmeans_params"],
            scorer_type=scorer_type,
            combined_type=combined_type,
        )

        # Assert
        assert (
            queried_sent_ids
            == expected_random_sent_ids[: scorer_params["query_number"]]
        )

    def test_call_return_correct_result_with_parallel_lc_ds(
        self,
        cm_scorer: BaseScorer,
        lc_scorer: BaseScorer,
        ds_scorer: BaseScorer,
        unlabeled_sentences: List[Sentence],
        scorer_params: dict,
    ) -> None:
        # Arrange
        scorer_type = "lc_ds"
        combined_type = "parallel"
        cm_scorer.predict = MagicMock(return_value=[None])
        entities = Entities()
        entities.entities = [None]
        cm_scorer.get_entities = MagicMock(return_value=entities)

        lc_scorer.score = MagicMock(
            return_value=np.array([0.09, 0.07, 0.05, 0.03, 0.01])
        )
        ds_scorer.score = MagicMock(return_value=np.array([0.2, 0.4, 0.6, 0.8, 1]))
        cm_scorer.get_scorers = MagicMock(return_value=(lc_scorer, ds_scorer))
        # normalized_scores = array([2. , 1.5, 1. , 0.5, 0. ])

        # Act
        queried_sent_ids = cm_scorer(
            unlabeled_sentences,
            scorer_params["tag_type"],
            scorer_params["query_number"],
            scorer_params["token_based"],
            tagger=scorer_params["tagger"],
            label_names=scorer_params["label_names"],
            embeddings=scorer_params["embeddings"],
            scorer_type=scorer_type,
            combined_type=combined_type,
        )

        # Assert
        assert queried_sent_ids == [0, 1, 2, 3]

    def test_call_return_correct_result_with_parallel_lc_cs(
        self,
        cm_scorer: BaseScorer,
        lc_scorer: BaseScorer,
        cs_scorer: BaseScorer,
        unlabeled_sentences: List[Sentence],
        scorer_params: dict,
    ) -> None:
        # Arrange
        scorer_type = "lc_ds"
        combined_type = "parallel"
        cm_scorer.predict = MagicMock(return_value=[None])
        entities = Entities()
        entities.entities = [None]
        cm_scorer.get_entities = MagicMock(return_value=entities)

        lc_scorer.score = MagicMock(
            return_value=np.array([0.09, 0.07, 0.05, 0.03, 0.01])
        )
        cs_scorer.score = MagicMock(return_value=np.array([0.2, 0.4, 0.6, 0.8, 1]))
        cm_scorer.get_scorers = MagicMock(return_value=(lc_scorer, cs_scorer))
        # normalized_scores = array([2. , 1.5, 1. , 0.5, 0. ])

        # Act
        queried_sent_ids = cm_scorer(
            unlabeled_sentences,
            scorer_params["tag_type"],
            scorer_params["query_number"],
            scorer_params["token_based"],
            tagger=scorer_params["tagger"],
            label_names=scorer_params["label_names"],
            embeddings=scorer_params["embeddings"],
            kmeans_params=scorer_params["kmeans_params"],
            scorer_type=scorer_type,
            combined_type=combined_type,
        )

        # Assert
        assert queried_sent_ids == [0, 1, 2, 3]

    def test_call_return_correct_result_with_parallel_mnlp_ds(
        self,
        cm_scorer: BaseScorer,
        mnlp_scorer: BaseScorer,
        ds_scorer: BaseScorer,
        unlabeled_sentences: List[Sentence],
        scorer_params: dict,
    ) -> None:
        # Arrange
        scorer_type = "lc_ds"
        combined_type = "parallel"
        cm_scorer.predict = MagicMock(return_value=[None])
        entities = Entities()
        entities.entities = [None]
        cm_scorer.get_entities = MagicMock(return_value=entities)

        mnlp_scorer.score = MagicMock(
            return_value=np.array([-0.09, -0.07, -0.05, -0.03, -0.01])
        )
        ds_scorer.score = MagicMock(return_value=np.array([0.2, 0.4, 0.6, 0.8, 1]))
        cm_scorer.get_scorers = MagicMock(return_value=(mnlp_scorer, ds_scorer))
        # normalized_scores = array([2. , 1.5, 1. , 0.5, 0. ])

        # Act
        queried_sent_ids = cm_scorer(
            unlabeled_sentences,
            scorer_params["tag_type"],
            scorer_params["query_number"],
            scorer_params["token_based"],
            tagger=scorer_params["tagger"],
            label_names=scorer_params["label_names"],
            embeddings=scorer_params["embeddings"],
            scorer_type=scorer_type,
            combined_type=combined_type,
        )

        # Assert
        assert queried_sent_ids == [0, 1, 2, 3]

    def test_call_return_correct_result_with_parallel_mnlp_cs(
        self,
        cm_scorer: BaseScorer,
        mnlp_scorer: BaseScorer,
        cs_scorer: BaseScorer,
        unlabeled_sentences: List[Sentence],
        scorer_params: dict,
    ) -> None:
        # Arrange
        scorer_type = "lc_ds"
        combined_type = "parallel"
        cm_scorer.predict = MagicMock(return_value=[None])
        entities = Entities()
        entities.entities = [None]
        cm_scorer.get_entities = MagicMock(return_value=entities)

        mnlp_scorer.score = MagicMock(
            return_value=np.array([-0.09, -0.07, -0.05, -0.03, -0.01])
        )
        cs_scorer.score = MagicMock(return_value=np.array([0.2, 0.4, 0.6, 0.8, 1]))
        cm_scorer.get_scorers = MagicMock(return_value=(mnlp_scorer, cs_scorer))
        # normalized_scores = array([2. , 1.5, 1. , 0.5, 0. ])

        # Act
        queried_sent_ids = cm_scorer(
            unlabeled_sentences,
            scorer_params["tag_type"],
            scorer_params["query_number"],
            scorer_params["token_based"],
            tagger=scorer_params["tagger"],
            label_names=scorer_params["label_names"],
            embeddings=scorer_params["embeddings"],
            scorer_type=scorer_type,
            combined_type=combined_type,
        )

        # Assert
        assert queried_sent_ids == [0, 1, 2, 3]
