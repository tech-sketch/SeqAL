from seqal.datasets import Corpus
from seqal.utils import assign_id_corpus


class TestUtils:
    def test_assign_id_corpus(self, corpus: Corpus) -> None:
        corpus_with_ids = assign_id_corpus(corpus)
        train = corpus_with_ids.train
        dev = corpus_with_ids.dev
        test = corpus_with_ids.test

        assert train[0].id == 0
        assert dev[0].id == 10
        assert test[0].id == 15
