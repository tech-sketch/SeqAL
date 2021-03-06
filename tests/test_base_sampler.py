from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from flair.data import Sentence
from torch.nn.functional import cosine_similarity

from seqal.datasets import Corpus
from seqal.samplers import BaseSampler


@pytest.fixture()
def base_sampler(scope="function"):
    """A BaseSampler instance"""
    base_sampler = BaseSampler()
    return base_sampler


@pytest.fixture()
def matrix_multiple_var(scope="function"):
    """Embedding matrix for test"""
    mat1 = torch.tensor([[3, 4], [3, 4]], dtype=torch.float64)
    mat2 = torch.tensor([[7, 24], [7, 24], [7, 24]], dtype=torch.float64)
    expected = torch.tensor(
        [[0.9360, 0.9360, 0.9360], [0.9360, 0.9360, 0.9360]], dtype=torch.float64
    )

    return {"mat1": mat1, "mat2": mat2, "expected": expected}


@pytest.fixture
def word_embeddings(fixture_path: Path) -> dict:
    """Word embeddings"""

    def load_embeddings(File):
        """Load emebddings from file"""
        embeddings = {}
        with open(File, "r") as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                embeddings[word] = embedding
        return embeddings

    file_path = fixture_path / "embeddings/word_embeddings.txt"
    return load_embeddings(file_path)


class TestBaseSampler:
    """Test BaseSampler class"""

    def test_query_data_on_token_base_if_query_number_smaller_than_total_token_number(
        self, base_sampler: BaseSampler, corpus: Corpus
    ) -> None:
        """Test query data on token base on condition of query_number is smaller than total token number"""
        # Arrange
        ordered_indices = list(range(10))
        token_required = 12

        # Act
        query_idx = base_sampler.query(
            corpus.train.sentences,
            ordered_indices,
            query_number=token_required,
            token_based=True,
        )

        # Assert
        assert query_idx == [0, 1, 2]

    def test_query_data_on_token_base_if_query_number_bigger_than_total_token_number(
        self, base_sampler: BaseSampler, corpus: Corpus
    ) -> None:
        """Test query data on token base on condition of query_number is bigger than total token number"""
        # Arrange
        ordered_indices = list(range(10))
        token_required = 10000

        # Act
        query_idx = base_sampler.query(
            corpus.train.sentences,
            ordered_indices,
            query_number=token_required,
            token_based=True,
        )

        # Assert
        assert query_idx == ordered_indices

    def test_query_data_on_sentence_base_if_query_number_smaller_than_total_token_number(
        self, base_sampler: BaseSampler, corpus: Corpus
    ) -> None:
        """Test query data on sentence base on condition of query_number is smaller than total token number"""
        # Arrange
        ordered_indices = list(range(10))
        sentence_required = 2

        # Act
        query_idx = base_sampler.query(
            corpus.train.sentences,
            ordered_indices,
            query_number=sentence_required,
            token_based=False,
        )

        # Assert
        assert query_idx == [0, 1]

    def test_query_data_on_sentence_base_if_query_number_bigger_than_total_token_number(
        self, base_sampler: BaseSampler, corpus: Corpus
    ) -> None:
        """Test query data on sentence base on condition of query_number is bigger than total token number"""
        # Arrange
        ordered_indices = list(range(10))
        sentence_required = 11

        # Act
        query_idx = base_sampler.query(
            corpus.train.sentences,
            ordered_indices,
            query_number=sentence_required,
            token_based=False,
        )

        # Assert
        assert query_idx == ordered_indices

    def test_query_raise_error_if_queried_number_smaller_than_zero(
        self, base_sampler: BaseSampler, corpus: Corpus
    ) -> None:
        """Test query raise error messsge if query_number is smaller than 0"""
        # Arrange
        ordered_indices = list(range(10))
        token_required = 0

        # Assert
        with pytest.raises(ValueError):
            # Act
            base_sampler.query(
                corpus.train.sentences,
                ordered_indices,
                query_number=token_required,
                token_based=True,
            )

    def test_sort_with_ascend_order(self, base_sampler: BaseSampler) -> None:
        """Test sort data on ascend order"""
        # Arrange
        sent_scores = np.array([1.1, 5.5, 2.2])

        # Act
        indices = base_sampler.sort(sent_scores, order="ascend")

        # Assert
        assert indices == [0, 2, 1]

    def test_sort_with_descend_order(self, base_sampler: BaseSampler) -> None:
        """Test sort data on descend order"""
        # Arrange
        sent_scores = np.array([1.1, 5.5, 2.2])

        # Act
        indices = base_sampler.sort(sent_scores, order="descend")

        # Assert
        assert indices == [1, 2, 0]

    def test_sort_raise_type_error_if_scores_format_is_not_ndarray(
        self, base_sampler: BaseSampler
    ) -> None:
        """Test sort function raise type error if input format is incorrect"""
        # Arrange
        sent_scores = [1, 2, 3]

        # Assert
        with pytest.raises(TypeError):
            # Act
            base_sampler.sort(sent_scores)

    def test_sort_raise_value_error_if_input_order_is_unavailable_string(
        self, base_sampler: BaseSampler
    ) -> None:
        """Test sort function raise value error if order parameter is incorrect"""
        # Arrange
        sent_scores = np.array([1.1, 5.5, 2.2])

        # Assert
        with pytest.raises(ValueError):
            # Act
            base_sampler.sort(sent_scores, order="order")

    def test_similarity_matrix_return_correct_cosine_similarity_of_two_matrix(
        self, base_sampler: BaseSampler, matrix_multiple_var: dict
    ) -> None:
        """Test similarity_matrix function return correct result"""
        # Act
        sim_mt = base_sampler.similarity_matrix(
            matrix_multiple_var["mat1"], matrix_multiple_var["mat2"]
        )

        # Assert
        assert torch.equal(sim_mt, matrix_multiple_var["expected"]) is True

    def test_similarity_matrix_comparing_with_cosine_similarity(
        self, base_sampler: BaseSampler, word_embeddings: dict
    ) -> None:
        """Test similarity_matrix function return correct result"""
        # Arrange
        v0 = torch.tensor(word_embeddings["Jonh"], dtype=torch.float64)
        v1 = torch.tensor(word_embeddings["with"], dtype=torch.float64)
        v2 = torch.tensor(word_embeddings["Peter"], dtype=torch.float64)
        vectors = torch.stack([v0, v1, v2])
        excepted0 = cosine_similarity(torch.stack([v0]), vectors)
        excepted1 = cosine_similarity(torch.stack([v1]), vectors)
        excepted2 = cosine_similarity(torch.stack([v2]), vectors)

        # Act
        sim_mt = base_sampler.similarity_matrix(vectors, vectors)

        # Assert
        assert torch.allclose(sim_mt[0], excepted0) is True
        assert torch.allclose(sim_mt[1], excepted1) is True
        assert torch.allclose(sim_mt[2], excepted2) is True

    def test_similarity_matrix_raise_error_if_input_type_is_not_tensor(
        self, base_sampler: BaseSampler, matrix_multiple_var: dict
    ) -> None:
        """Test similarity_matrix function raise error input data type is not Tensor"""
        # Arrange
        mat1 = np.array([[3, 4], [3, 4]])

        # Assert
        with pytest.raises(TypeError):
            # Act
            base_sampler.similarity_matrix(mat1, matrix_multiple_var["mat2"])

    def test_similarity_matrix_raise_error_if_two_matrix_shape_is_not_compatible(
        self, base_sampler: BaseSampler, matrix_multiple_var: dict
    ) -> None:
        """Test similarity_matrix function raise error input matrix is not compatible"""
        # Arrange
        mat2 = matrix_multiple_var["mat2"].transpose(0, 1)

        # Assert
        with pytest.raises(RuntimeError):
            # Act
            base_sampler.similarity_matrix(matrix_multiple_var["mat1"], mat2)

    def test_get_entities_raise_type_error_if_unlabeled_sentences_have_not_been_predicted(
        self,
        base_sampler: BaseSampler,
        unlabeled_sentences: List[Sentence],
    ) -> None:
        """Test get_entities function raise type_error if unlabeled sentences have not been predicted"""
        # Arrange
        tag_type = "ner"
        embeddings = MagicMock()
        embeddings.embed = MagicMock(return_value=None)

        # Assert
        with pytest.raises(TypeError):
            # Act
            base_sampler.get_entities(unlabeled_sentences, embeddings, tag_type)

    def test_get_entity_return_correct_result(self, base_sampler: BaseSampler) -> None:
        """Test get_entities function return correct result if sentence contains entities"""
        # Arrange
        tag_type = "ner"
        sentence = Sentence("Peter is working")
        sentence[0].add_tag(tag_type, "PER")
        sentences = [sentence]
        embeddings = MagicMock()
        embeddings.embed = MagicMock(return_value=None)

        # Act
        entities = base_sampler.get_entities(sentences, embeddings, tag_type)

        # Assert
        assert entities.entities[0].id == 0
        assert entities.entities[0].sent_id == 0
        assert entities.entities[0].span.text == "Peter"
        assert entities.entities[0].label == "PER"
