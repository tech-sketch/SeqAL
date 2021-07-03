from seqal.datasets import Corpus


class TestCorpus:
    def test_get_all_sentences(self, corpus: Corpus) -> None:
        assert len(corpus.get_all_sentences()) == 20


class TestColumnCorpus:
    def test__len__(self, corpus: Corpus) -> None:
        del corpus.train.sentences[0]
        assert len(corpus.train.sentences) == 9
