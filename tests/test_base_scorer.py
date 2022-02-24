import numpy as np
import pytest
import torch

from seqal.base_scorer import BaseScorer
from seqal.datasets import Corpus


@pytest.fixture()
def base_scorer(scope="function"):
    """A BaseScorer instance"""
    base_scorer = BaseScorer()
    return base_scorer


@pytest.fixture()
def matrix_multiple_var(scope="function"):
    """Embedding matrix for test"""
    mat1 = torch.tensor([[3, 4], [3, 4]], dtype=torch.float32)
    mat2 = torch.tensor([[7, 24], [7, 24], [7, 24]], dtype=torch.float32)
    expected = torch.tensor(
        [[0.9360, 0.9360, 0.9360], [0.9360, 0.9360, 0.9360]], dtype=torch.float32
    )

    return {"mat1": mat1, "mat2": mat2, "expected": expected}


class TestBaseScorer:
    """Test BaseScorer class"""

    def test_query_data_on_token_base_if_query_number_smaller_than_total_token_number(
        self, base_scorer: BaseScorer, corpus: Corpus
    ) -> None:
        """Test query data on token base on condition of query_number is smaller than total token number"""
        # Arrange
        ordered_indices = list(range(10))
        token_required = 12

        # Act
        query_idx = base_scorer.query(
            corpus.train.sentences,
            ordered_indices,
            query_number=token_required,
            token_based=True,
        )

        # Assert
        assert query_idx == [0, 1, 2]

    def test_query_data_on_token_base_if_query_number_bigger_than_total_token_number(
        self, base_scorer: BaseScorer, corpus: Corpus
    ) -> None:
        """Test query data on token base on condition of query_number is bigger than total token number"""
        # Arrange
        ordered_indices = list(range(10))
        token_required = 10000

        # Act
        query_idx = base_scorer.query(
            corpus.train.sentences,
            ordered_indices,
            query_number=token_required,
            token_based=True,
        )

        # Assert
        assert query_idx == ordered_indices

    def test_query_data_on_sentence_base_if_query_number_smaller_than_total_token_number(
        self, base_scorer: BaseScorer, corpus: Corpus
    ) -> None:
        """Test query data on sentence base on condition of query_number is smaller than total token number"""
        # Arrange
        ordered_indices = list(range(10))
        sentence_required = 2

        # Act
        query_idx = base_scorer.query(
            corpus.train.sentences,
            ordered_indices,
            query_number=sentence_required,
            token_based=False,
        )

        # Assert
        assert query_idx == [0, 1]

    def test_query_data_on_sentence_base_if_query_number_bigger_than_total_token_number(
        self, base_scorer: BaseScorer, corpus: Corpus
    ) -> None:
        """Test query data on sentence base on condition of query_number is bigger than total token number"""
        # Arrange
        ordered_indices = list(range(10))
        sentence_required = 11

        # Act
        query_idx = base_scorer.query(
            corpus.train.sentences,
            ordered_indices,
            query_number=sentence_required,
            token_based=False,
        )

        # Assert
        assert query_idx == ordered_indices

    def test_query_raise_error_if_queried_number_smaller_than_zero(
        self, base_scorer: BaseScorer, corpus: Corpus
    ) -> None:
        """Test query raise error messsge if query_number is smaller than 0"""
        # Arrange
        ordered_indices = list(range(10))
        token_required = 0

        # Assert
        with pytest.raises(ValueError):
            # Act
            base_scorer.query(
                corpus.train.sentences,
                ordered_indices,
                query_number=token_required,
                token_based=True,
            )

    def test_sort_with_ascend_order(self, base_scorer: BaseScorer) -> None:
        """Test sort data on ascend order"""
        # Arrange
        sent_scores = np.array([1.1, 5.5, 2.2])

        # Act
        indices = base_scorer.sort(sent_scores, order="ascend")

        # Assert
        assert indices == [0, 2, 1]

    def test_sort_with_descend_order(self, base_scorer: BaseScorer) -> None:
        """Test sort data on descend order"""
        # Arrange
        sent_scores = np.array([1.1, 5.5, 2.2])

        # Act
        indices = base_scorer.sort(sent_scores, order="descend")

        # Assert
        assert indices == [1, 2, 0]

    def test_sort_raise_type_error_if_scores_format_is_not_ndarray(
        self, base_scorer: BaseScorer
    ) -> None:
        """Test sort function raise type error if input format is incorrect"""
        # Arrange
        sent_scores = [1, 2, 3]

        # Assert
        with pytest.raises(TypeError):
            # Act
            base_scorer.sort(sent_scores)

    def test_sort_raise_value_error_if_input_order_is_unavailable_string(
        self, base_scorer: BaseScorer
    ) -> None:
        """Test sort function raise value error if order parameter is incorrect"""
        # Arrange
        sent_scores = np.array([1.1, 5.5, 2.2])

        # Assert
        with pytest.raises(ValueError):
            # Act
            base_scorer.sort(sent_scores, order="order")

    def test_similarity_matrix_return_correct_cosine_similarity_of_two_matrix(
        self, base_scorer: BaseScorer, matrix_multiple_var: dict
    ) -> None:
        """Test similarity_matrix function return correct result"""
        # Act
        sim_mt = base_scorer.similarity_matrix(
            matrix_multiple_var["mat1"], matrix_multiple_var["mat2"]
        )

        # Assert
        assert torch.equal(sim_mt, matrix_multiple_var["expected"]) is True

    def test_similarity_matrix_if_tensor_dtype_is_not_float32(
        self, base_scorer: BaseScorer, matrix_multiple_var: dict
    ) -> None:
        """Test similarity_matrix function return correct result if input data type is not tensor.float32"""
        # Arrange
        mat1 = matrix_multiple_var["mat1"].to(dtype=torch.int32)
        mat2 = matrix_multiple_var["mat2"].to(dtype=torch.float64)

        # Act
        sim_mt = base_scorer.similarity_matrix(mat1, mat2)

        # Assert
        assert torch.equal(sim_mt, matrix_multiple_var["expected"]) is True

    def test_similarity_matrix_raise_error_if_input_type_is_not_tensor(
        self, base_scorer: BaseScorer, matrix_multiple_var: dict
    ) -> None:
        """Test similarity_matrix function raise error input data type is not Tensor"""
        # Arrange
        mat1 = np.array([[3, 4], [3, 4]])

        # Assert
        with pytest.raises(TypeError):
            # Act
            base_scorer.similarity_matrix(mat1, matrix_multiple_var["mat2"])

    def test_similarity_matrix_raise_error_if_two_matrix_shape_is_not_compatible(
        self, base_scorer: BaseScorer, matrix_multiple_var: dict
    ) -> None:
        """Test similarity_matrix function raise error input matrix is not compatible"""
        # Arrange
        mat2 = matrix_multiple_var["mat2"].transpose(0, 1)

        # Assert
        with pytest.raises(RuntimeError):
            # Act
            base_scorer.similarity_matrix(matrix_multiple_var["mat1"], mat2)

    def test_normalize_score(self):
        # Check input type array

        # Expected output

        pass
