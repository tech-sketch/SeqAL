from pathlib import Path

from flair.data import Sentence
from flair.models import SequenceTagger

from seqal.datasets import Corpus
from seqal.query_strategies import ls_sampling, mnlp_sampling, random_sampling


def test_random_sampling(corpus: Corpus) -> None:
    query_id, query_inst = random_sampling(corpus)
    assert isinstance(query_id, int) is True
    assert isinstance(query_inst, Sentence) is True


def test_ls_sampling(
    fixture_path: Path, corpus: Corpus, trained_tagger: SequenceTagger
) -> None:
    query_id, query_inst = ls_sampling(corpus, trained_tagger)
    candidate_idx = [x for x in range(len(corpus.test.sentences))]
    assert query_id in candidate_idx
    assert isinstance(query_inst, Sentence) is True


def test_mnlp_sampling(
    fixture_path: Path, corpus: Corpus, trained_tagger: SequenceTagger
) -> None:
    query_id, query_inst = mnlp_sampling(corpus, trained_tagger)
    candidate_idx = [x for x in range(len(corpus.test.sentences))]
    assert query_id in candidate_idx
    assert isinstance(query_inst, Sentence) is True
