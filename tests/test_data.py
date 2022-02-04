from unittest.mock import MagicMock, PropertyMock

import pytest
import torch
from flair.data import Sentence

from seqal.data import Entities, Entity
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


@pytest.fixture()
def entity_list(scope="function"):
    sentence = Sentence("George Washington went to Washington.")
    sentence[0].add_tag("ner", "B-PER")
    sentence[1].add_tag("ner", "E-PER")
    sentence[4].add_tag("ner", "S-LOC")
    spans = sentence.get_spans("ner")

    e0 = Entity(0, 0, spans[0])
    e1 = Entity(1, 0, spans[1])

    return [e0, e1]


class TestEntity:
    def test_vector_return_correct_result_if_embedding_exist(self) -> None:
        # Arrange
        span = MagicMock(
            tokens=[
                MagicMock(embedding=torch.tensor([0.0, -1.0])),
                MagicMock(embedding=torch.tensor([1.0, 0.0])),
            ]
        )
        entity = Entity(0, 0, span)
        expected = torch.tensor([0.5, -0.5])

        # Act
        embeddings = entity.vector

        # Assert
        assert torch.equal(embeddings, expected)

    def test_vector_raise_error_if_embedding_not_exist(self) -> None:
        # Arrage
        span = MagicMock(
            tokens=[
                MagicMock(embedding=torch.tensor([])),
                MagicMock(embedding=torch.tensor([])),
            ]
        )
        entity = Entity(0, 0, span)

        # Assert
        with pytest.raises(TypeError):
            # Act
            _ = entity.vector

    def test_label(self) -> None:
        # Arrage
        span = MagicMock(tag="PER")
        entity = Entity(0, 0, span)

        # Act
        label = entity.label

        # Assert
        assert label == "PER"


class TestEntities:
    def test_add(self) -> None:
        # Arrange
        span = MagicMock()
        entity = Entity(0, 0, span)

        # Act
        entities = Entities()
        entities.add(entity)

        # Assert
        assert entities.entities == [entity]

    def test_group_by_sentence(self) -> None:
        # Arrange
        span0, span1 = MagicMock(), MagicMock()
        e0 = Entity(0, 0, span0)
        e1 = Entity(1, 0, span1)
        expected_entities_per_sentence = {0: [e0, e1]}
        entities = Entities()
        entities.add(e0)
        entities.add(e1)

        # Act
        entities_per_sentence = entities.group_by_sentence

        # Assert
        assert expected_entities_per_sentence == entities_per_sentence

    def test_group_by_label(self) -> None:
        # Arrange
        e0, e1 = MagicMock(), MagicMock()
        type(e0).label = PropertyMock(return_value="PER")
        type(e1).label = PropertyMock(return_value="LOC")
        expected_entities_per_label = {"PER": [e0], "LOC": [e1]}
        entities = Entities()
        entities.add(e0)
        entities.add(e1)

        # Act
        entities_per_label = entities.group_by_label

        # Assert
        assert expected_entities_per_label == entities_per_label
