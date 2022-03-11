from seqal.datasets import Corpus


class TestCorpus:
    """Test Corpus class"""

    def test_get_all_sentences(self, corpus: Corpus) -> None:
        """Test get_all_sentences function"""
        assert len(corpus.get_all_sentences()) == 20

    def test_add_queried_samples(self, corpus: Corpus) -> None:
        """Test add_queried_samples function"""
        # Arrange
        queried_samples = corpus.dev.sentences

        # Act
        corpus.add_queried_samples(queried_samples)

        # Assert
        assert len(corpus.train.sentences) == 15


class TestColumnCorpus:
    def test__len__(self, corpus: Corpus) -> None:
        del corpus.train.sentences[0]
        assert len(corpus.train.sentences) == 9
