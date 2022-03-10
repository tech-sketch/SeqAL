from pathlib import Path
from typing import List

from flair.data import Sentence

from seqal.active_learner import ActiveLearner, remove_queried_samples


def test_remove_queried_samples(unlabeled_sentences: List[Sentence]) -> None:
    """Test remove_queried_samples function"""
    # Arrange
    sents = unlabeled_sentences[:5]
    queried_idx = [1, 2, 4]
    expected_new_sents = [sents[0], sents[3]]
    expected_queried_sents = [sents[1], sents[2], sents[4]]

    # Act
    new_sents, queried_sents = remove_queried_samples(sents, queried_idx)

    # Assert
    assert new_sents == expected_new_sents
    assert queried_sents == expected_queried_sents


class TestActiveLearner:
    """Test ActiveLearner class"""

    def test_fit_without_error(
        self, fixture_path: Path, learner: ActiveLearner
    ) -> None:
        """Test fit function works no problem"""
        # Arrange
        save_path = fixture_path / "output"

        # Act
        learner.fit(save_path)
        del learner
