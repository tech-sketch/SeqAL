from pathlib import Path

from flair.embeddings import StackedEmbeddings
from flair.models import SequenceTagger

from seqal.datasets import Corpus
from seqal.query_strategies import (
    ls_sampling,
    mnlp_sampling,
    random_sampling,
    similarity_sampling,
)


def test_random_sampling(corpus: Corpus) -> None:
    # Query single sentence
    query_idx = random_sampling(corpus.train.sentences, query_number=0)
    assert len(query_idx) == 1

    # Query multiple sentences
    query_idx = random_sampling(corpus.train.sentences, query_number=2)
    assert len(query_idx) == 2

    # Batch multiple sentences based on token count
    token_required = 12
    query_idx = random_sampling(
        corpus.train.sentences, query_number=token_required, token_based=True
    )
    assert len(query_idx) >= 1 and len(query_idx) <= 3


def test_ls_sampling(
    fixture_path: Path, corpus: Corpus, trained_tagger: SequenceTagger
) -> None:
    tag_type = "ner"

    # Query single sentence
    query_idx = ls_sampling(corpus.train.sentences, tag_type, query_number=0)
    assert len(query_idx) == 1

    # Query multiple sentences
    query_idx = ls_sampling(corpus.train.sentences, tag_type, query_number=2)
    assert len(query_idx) == 2

    # Batch multiple sentences based on token count
    token_required = 12
    query_idx = ls_sampling(
        corpus.train.sentences, tag_type, query_number=token_required, token_based=True
    )
    assert len(query_idx) >= 1 and len(query_idx) <= 3


def test_mnlp_sampling(
    fixture_path: Path, corpus: Corpus, trained_tagger: SequenceTagger
) -> None:
    tag_type = "ner"

    # Query single sentence
    query_idx = mnlp_sampling(corpus.train.sentences, tag_type, query_number=0)
    assert len(query_idx) == 1

    # Query multiple sentences
    query_idx = mnlp_sampling(corpus.train.sentences, tag_type, query_number=2)
    assert len(query_idx) == 2

    # Batch multiple sentences based on token count
    token_required = 12
    query_idx = mnlp_sampling(
        corpus.train.sentences, tag_type, query_number=token_required, token_based=True
    )
    assert len(query_idx) >= 1 and len(query_idx) <= 3


def test_similarity_sampling(
    fixture_path: Path, corpus: Corpus, embeddings: StackedEmbeddings
) -> None:
    tag_type = "ner"
    label_names = ["O", "I-PER", "I-LOC", "I-ORG", "I-MISC"]

    # Query single sentence
    query_idx = similarity_sampling(
        corpus.train.sentences,
        tag_type,
        query_number=0,
        label_names=label_names,
        embeddings=embeddings,
    )
    assert len(query_idx) == 1

    # Query multiple sentences
    query_idx = similarity_sampling(
        corpus.train.sentences,
        tag_type,
        query_number=2,
        label_names=label_names,
        embeddings=embeddings,
    )
    assert len(query_idx) == 2

    # Batch multiple sentences based on token count
    token_required = 12
    query_idx = similarity_sampling(
        corpus.train.sentences,
        tag_type,
        query_number=token_required,
        token_based=True,
        label_names=label_names,
        embeddings=embeddings,
    )
    assert len(query_idx) >= 1 and len(query_idx) <= 3
