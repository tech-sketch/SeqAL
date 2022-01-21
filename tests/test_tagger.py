from unittest.mock import MagicMock

from torch import tensor

from seqal.datasets import Corpus
from seqal.tagger import SequenceTagger


class TestSequenceTagger:
    def test_log_probability(
        self, corpus: Corpus, trained_tagger: SequenceTagger
    ) -> None:
        sents = corpus.train.sentences
        trained_tagger.forward = MagicMock(return_value=None)
        trained_tagger._calculate_loss = MagicMock(
            return_value=(tensor([1, 2, 3]), None)
        )

        # Method result
        log_probs = trained_tagger.log_probability(sents)

        # Expected result
        expected = [-1, -2, -3]

        assert expected == log_probs.tolist()
