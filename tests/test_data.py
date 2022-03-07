from unittest.mock import MagicMock

import pytest
import torch

from seqal.data import Entities, Entity


class TestEntity:
    """Test Entity class"""

    def test_vector_return_correct_result_if_embedding_exist(self) -> None:
        """Test vector property return correct result if embedding exist"""
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
        """Test vector property raise error if embedding not exist"""
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
        """Test label property"""
        # Arrage
        span = MagicMock(tag="PER")
        entity = Entity(0, 0, span)

        # Act
        label = entity.label

        # Assert
        assert label == "PER"

    def test_text(self) -> None:
        """Test text property"""
        # Arrage
        span = MagicMock(text="Peter")
        entity = Entity(0, 0, span)

        # Act
        text = entity.text

        # Assert
        assert text == "Peter"


class TestEntities:
    """Test Entities class"""

    def test_add(self) -> None:
        """Test add function"""
        # Arrange
        span = MagicMock()
        entity = Entity(0, 0, span)

        # Act
        entities = Entities()
        entities.add(entity)

        # Assert
        assert entities.entities == [entity]

    def test_group_by_sentence(self) -> None:
        """Test group_by_sentence cached_property"""
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
        """Test group_by_label cached_property"""
        # Arrange
        e0 = MagicMock(label="PER")
        e1 = MagicMock(label="LOC")
        expected_entities_per_label = {"PER": [e0], "LOC": [e1]}
        entities = Entities()
        entities.add(e0)
        entities.add(e1)

        # Act
        entities_per_label = entities.group_by_label

        # Assert
        assert expected_entities_per_label == entities_per_label

    def test_group_by_cluster(self) -> None:
        """Test group_by_cluster cached_property"""
        # Arrange
        e0 = MagicMock(cluster=1)
        e1 = MagicMock(cluster=0)
        expected_entities_per_cluster = {1: [e0], 0: [e1]}
        entities = Entities()
        entities.add(e0)
        entities.add(e1)

        # Act
        entities_per_cluster = entities.group_by_cluster

        # Assert
        assert expected_entities_per_cluster == entities_per_cluster
