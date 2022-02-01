import random
from email.mime import base
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from flair.data import Sentence
from flair.embeddings import StackedEmbeddings

from seqal.active_learner import ActiveLearner
from seqal.base_scorer import BaseScorer
from seqal.datasets import Corpus


@pytest.fixture()
def base_scorer(scope="function"):
    base_scorer = BaseScorer()
    return base_scorer


class TestBaseScorer:
    def test_query(self, base_scorer: BaseScorer, corpus: Corpus) -> None:
        ordered_indices = list(range(10))

        # Invalid input case
        with pytest.raises(ValueError):
            token_required = -2
            query_idx = base_scorer.query(
                corpus.train.sentences,
                ordered_indices,
                query_number=token_required,
                token_based=True,
            )

        # 1. Token base
        # 1.1 token_required is smaller than total token count of sentences
        token_required = 12
        query_idx = base_scorer.query(
            corpus.train.sentences,
            ordered_indices,
            query_number=token_required,
            token_based=True,
        )
        assert query_idx == [0, 1, 2]

        # 1.2 token_required is bigger than total token count of sentences
        token_required = 10000
        query_idx = base_scorer.query(
            corpus.train.sentences,
            ordered_indices,
            query_number=token_required,
            token_based=True,
        )
        assert query_idx == ordered_indices

        # 2. Sentence base
        # 2.1 sentence_required is smaller than total token count of sentences
        sentence_required = 2
        query_idx = base_scorer.query(
            corpus.train.sentences,
            ordered_indices,
            query_number=sentence_required,
            token_based=False,
        )
        assert query_idx == [0, 1]

        # 2.2 sentence_required is bigger than total token count of sentences
        sentence_required = 11
        query_idx = base_scorer.query(
            corpus.train.sentences,
            ordered_indices,
            query_number=sentence_required,
            token_based=False,
        )

        assert query_idx == ordered_indices

    def test_sort(self, base_scorer: BaseScorer) -> None:

        # Invalid input case
        with pytest.raises(TypeError):
            sent_scores = [1, 2, 3]
            base_scorer.sort(sent_scores)

        with pytest.raises(TypeError):
            sent_scores = ["This", "is", "test"]
            base_scorer.sort(sent_scores)

        with pytest.raises(ValueError):
            sent_scores = np.array([1.1, 5.5, 2.2])
            base_scorer.sort(sent_scores, order="order")

        # Ascend order
        sent_scores = np.array([1.1, 5.5, 2.2])
        indices = base_scorer.sort(sent_scores, order="ascend")
        assert indices == [0, 2, 1]

        # Descend order
        sent_scores = np.array([1.1, 5.5, 2.2])
        indices = base_scorer.sort(sent_scores, order="descend")
        assert indices == [1, 2, 0]

    def test_similarity_matrix(self, base_scorer: BaseScorer) -> None:
        # Correct case
        a = torch.tensor([[3, 4], [3, 4]], dtype=torch.float32)
        b = torch.tensor([[7, 24], [7, 24], [7, 24]], dtype=torch.float32)  # (3, 2)
        sim_mt = base_scorer.similarity_matrix(a, b)
        expected = torch.tensor(
            [[0.9360, 0.9360, 0.9360], [0.9360, 0.9360, 0.9360]], dtype=torch.float32
        )
        assert torch.equal(sim_mt, expected) is True

        # Raise error is input is not tensor
        with pytest.raises(TypeError):
            a = np.array([[3, 4], [3, 4]])  # (2, 2)
            b = torch.tensor([[7, 24], [7, 24], [7, 24]], dtype=torch.float32)  # (3, 2)
            sim_mt = base_scorer.similarity_matrix(a, b)

        # Normalize input dtype is to tensor.float32
        a = torch.tensor([[3, 4], [3, 4]], dtype=torch.int32)
        b = torch.tensor([[7, 24], [7, 24], [7, 24]], dtype=torch.float64)  # (3, 2)
        sim_mt = base_scorer.similarity_matrix(a, b)
        assert torch.equal(sim_mt, expected) is True

        # Raise error if input shape is not compatible
        with pytest.raises(RuntimeError):
            a = torch.tensor([[3, 4], [3, 4]], dtype=torch.float32)
            b = torch.tensor([[7, 24], [7, 24], [7, 24]], dtype=torch.float32)  # (3, 2)
            b = b.transpose(0, 1)
            sim_mt = base_scorer.similarity_matrix(a, b)

    def test_normalize_score(self):
        # Check input type array

        # Expected output

        pass
