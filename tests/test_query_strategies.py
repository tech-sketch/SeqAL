import random
from typing import List

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
from tests.conftest import FakeSentence


def test_random_sampling(corpus: Corpus) -> None:
    # Expected result
    random.seed(0)
    expected_idx = list(range(len(corpus.train.sentences)))
    random.shuffle(expected_idx)

    # Method result
    ordered_idx = random_sampling(corpus.train.sentences)

    assert expected_idx == ordered_idx


def test_lc_sampling(fake_sents: List[FakeSentence]) -> None:
    tag_type = "ner"
    # Expected result
    descend_indices = [0, 9, 5, 3, 4, 8, 7, 6, 2, 1]

    # Method result
    ordered_idx = lc_sampling(fake_sents, tag_type)
    assert descend_indices == list(ordered_idx)


def test_mnlp_sampling(fake_sents: List[FakeSentence]) -> None:
    tag_type = "ner"

    # Expected result
    ascend_indices = [6, 2, 1, 0, 9, 5, 3, 4, 8, 7]

    # Method result
    ordered_idx = mnlp_sampling(fake_sents, tag_type)
    assert ascend_indices == list(ordered_idx)


def test_similarity_sampling(sents: Sentence, embeddings: StackedEmbeddings) -> None:
    tag_type = "ner"
    label_names = ["O", "I-PER", "I-LOC", "I-ORG", "I-MISC"]

    # Expected result
    ascend_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Method result
    ordered_idx = similarity_sampling(
        sents,
        tag_type,
        label_names=label_names,
        embeddings=embeddings,
    )
    assert ascend_indices == list(ordered_idx)


def test_cluster_sampling(sents: Sentence, embeddings: StackedEmbeddings) -> None:
    tag_type = "ner"
    label_names = ["O", "I-PER", "I-LOC", "I-ORG", "I-MISC"]

    # Expected result
    ascend_indices = [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]

    # Method result
    ordered_idx = cluster_sampling(
        sents, tag_type, label_names=label_names, embeddings=embeddings
    )
    assert ascend_indices == list(ordered_idx)
