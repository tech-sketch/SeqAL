from pathlib import Path
from typing import List

from flair.data import Sentence

from seqal.active_learner import ActiveLearner, remove_query_samples


def test_remove_query_samples(unlabeled_sentences: List[Sentence]) -> None:
    """Test remove_query_samples function"""
    # Arrange
    sents = unlabeled_sentences[:5]
    query_idx = [1, 2, 4]
    expected_new_sents = [sents[0], sents[3]]
    expected_query_sents = [sents[1], sents[2], sents[4]]

    # Act
    new_sents, query_sents = remove_query_samples(sents, query_idx)

    # Assert
    assert new_sents == expected_new_sents
    assert query_sents == expected_query_sents


class TestActiveLearner:
    def test_fit(self, fixture_path: Path, learner: ActiveLearner) -> None:
        save_path = fixture_path / "output"
        learner.fit(save_path)
        del learner
