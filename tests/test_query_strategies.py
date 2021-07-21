from pathlib import Path

from flair.models import SequenceTagger

from seqal.datasets import Corpus
from seqal.query_strategies import ls_sampling, mnlp_sampling, random_sampling


def test_random_sampling(corpus: Corpus) -> None:
    count = len(corpus.train.sentences)
    sents, query_samples = random_sampling(corpus.train.sentences)
    assert len(sents) == count - 1
    assert len(query_samples) == 1


def test_ls_sampling(
    fixture_path: Path, corpus: Corpus, trained_tagger: SequenceTagger
) -> None:
    count = len(corpus.train.sentences)
    sents, query_samples = ls_sampling(corpus.train.sentences, trained_tagger)
    assert len(sents) == count - 1
    assert len(query_samples) == 1


def test_mnlp_sampling(
    fixture_path: Path, corpus: Corpus, trained_tagger: SequenceTagger
) -> None:
    count = len(corpus.train.sentences)
    sents, query_samples = mnlp_sampling(corpus.train.sentences, trained_tagger)
    assert len(sents) == count - 1
    assert len(query_samples) == 1
