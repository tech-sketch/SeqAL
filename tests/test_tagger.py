from typing import List

import numpy as np
import pytest
from flair.data import Sentence

from seqal.datasets import Corpus
from seqal.tagger import SequenceTagger


class TestSequenceTagger:
    def test_log_probability_return_result_without_error_if_sentences_have_been_predicted(
        self, predicted_sentences: List[Sentence], trained_tagger: SequenceTagger
    ) -> None:
        # Act
        trained_tagger.log_probability(predicted_sentences)

    def test_log_probability_raise_index_error_if_unlabeled_sentences_have_not_been_predicted(
        self, unlabeled_sentences: List[Sentence], trained_tagger: SequenceTagger
    ) -> None:

        # Assert
        with pytest.raises(IndexError):
            # Act
            trained_tagger.log_probability(unlabeled_sentences)

    def test_log_probability_raise_error_if_labeled_sentences_have_not_been_predicted(
        self, corpus: Corpus, trained_tagger: SequenceTagger
    ) -> None:
        # Arrange
        sents = corpus.train.sentences  # Labeled sentence
        log_probs_of_labeled_sentences_before_prediction = (
            trained_tagger.log_probability(sents)
        )

        # Act
        trained_tagger.predict(sents)
        log_probs_of_labeled_sentences_after_prediction = (
            trained_tagger.log_probability(sents)
        )

        # Assert
        assert (
            np.array_equal(
                log_probs_of_labeled_sentences_before_prediction,
                log_probs_of_labeled_sentences_after_prediction,
            )
            is False
        )
