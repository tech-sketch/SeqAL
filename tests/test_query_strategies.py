import random
from typing import List

from flair.data import Sentence
from flair.embeddings import StackedEmbeddings

from seqal.active_learner import predict_data_pool
from seqal.datasets import Corpus
from seqal.query_strategies import (
    cluster_sampling,
    lc_sampling,
    mnlp_sampling,
    random_sampling,
    similarity_sampling,
)
from seqal.tagger import SequenceTagger


def test_random_sampling(corpus: Corpus) -> None:
    # Expected result
    random.seed(0)
    expected_idx = list(range(len(corpus.train.sentences)))
    random.shuffle(expected_idx)

    # Method result
    ordered_idx = random_sampling(corpus.train.sentences)

    assert expected_idx == ordered_idx


def test_lc_sampling(sents: List[Sentence], trained_tagger: SequenceTagger) -> None:
    tag_type = "ner"

    predict_data_pool(sents, trained_tagger)
    # Expected result
    ascend_indices = [2, 1, 0, 6, 3, 8, 4, 5, 9, 7]

    # Method result
    ordered_idx = lc_sampling(sents, tag_type, tagger=trained_tagger)
    assert ascend_indices == list(ordered_idx)


def test_mnlp_sampling(sents: List[Sentence], trained_tagger: SequenceTagger) -> None:
    tag_type = "ner"

    predict_data_pool(sents, trained_tagger)
    # Expected result
    ascend_indices = [6, 7, 9, 3, 8, 4, 5, 0, 2, 1]

    # Method result
    ordered_idx = mnlp_sampling(sents, tag_type, tagger=trained_tagger)
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
