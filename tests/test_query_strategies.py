import random
from typing import List
from unittest.mock import MagicMock

import numpy as np
from flair.data import Sentence
from flair.embeddings import StackedEmbeddings

from seqal.datasets import Corpus
from seqal.query_strategies import (
    cluster_sampling,
    lc_sampling,
    mnlp_sampling,
    random_sampling,
    similarity_sampling,
)


def test_random_sampling(corpus: Corpus) -> None:
    # Expected result
    random.seed(0)
    expected_idx = list(range(len(corpus.train.sentences)))
    random.shuffle(expected_idx)

    # Method result
    ordered_idx = random_sampling(corpus.train.sentences)

    assert expected_idx == ordered_idx


def test_lc_sampling(sents: List[Sentence]) -> None:
    tag_type = "ner"

    tagger = MagicMock()
    tagger.log_probability = MagicMock(return_value=np.array([1, 2, 3, 4]))

    # Method result
    ordered_idx = lc_sampling(sents, tag_type, tagger=tagger)

    # Expected result
    expected = [0, 1, 2, 3]

    assert expected == ordered_idx


def test_mnlp_sampling() -> None:
    tag_type = "ner"

    s1 = MagicMock()
    s2 = MagicMock()
    s3 = MagicMock()
    s4 = MagicMock()
    s1.__len__ = MagicMock(return_value=1)
    s2.__len__ = MagicMock(return_value=1)
    s3.__len__ = MagicMock(return_value=1)
    s4.__len__ = MagicMock(return_value=1)
    sents = [s1, s2, s3, s4]

    tagger = MagicMock()
    tagger.log_probability = MagicMock(return_value=np.array([1, 2, 3, 4]))

    # Method result
    ordered_idx = mnlp_sampling(sents, tag_type, tagger=tagger)

    # Expected result
    expected = [0, 1, 2, 3]

    assert expected == ordered_idx


def test_similarity_sampling(
    sents: List[Sentence], embeddings: StackedEmbeddings
) -> None:
    tag_type = "ner"
    label_names = ["O", "I-PER", "I-LOC", "I-ORG", "I-MISC"]

    # Expected result
    expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Method result
    ordered_idx = similarity_sampling(
        sents,
        tag_type,
        label_names=label_names,
        embeddings=embeddings,
    )
    assert expected == list(ordered_idx)


def test_cluster_sampling(sents: List[Sentence], embeddings: StackedEmbeddings) -> None:
    tag_type = "ner"
    label_names = ["O", "I-PER", "I-LOC", "I-ORG", "I-MISC"]

    # Expected result
    expected = [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]

    # Method result
    ordered_idx = cluster_sampling(
        sents, tag_type, label_names=label_names, embeddings=embeddings
    )
    assert expected == list(ordered_idx)
