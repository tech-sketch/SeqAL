from pathlib import Path
from typing import List

from flair.data import Sentence

from seqal.active_learner import ActiveLearner, remove_queried_samples
from seqal.datasets import Corpus


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

    def test_resume_without_error(
        self, fixture_path: Path, corpus: Corpus, trained_learner: ActiveLearner
    ) -> None:
        """Test fit function works no problem"""
        # Arrange
        save_path = fixture_path / "output"
        queried_samples = corpus.dev.sentences

        # Act
        trained_learner.resume(queried_samples, save_path)

    def test_teach_with_resume_false_return_new_model(
        self, corpus: Corpus, trained_learner: ActiveLearner
    ) -> None:
        """Test teach function when train a new model on all labeled data"""
        # Arrange
        model_id = id(trained_learner.trained_tagger)
        queried_samples = corpus.dev.sentences

        # Act
        trained_learner.teach(queried_samples, resume=False)

        # Assert
        assert model_id != id(trained_learner.trained_tagger)

    def test_teach_with_resume_true_return_same_model(
        self, corpus: Corpus, trained_learner: ActiveLearner
    ) -> None:
        """Test teach function when train a new model on all labeled data"""
        # Arrange
        model_id = id(trained_learner.trained_tagger)
        queried_samples = corpus.dev.sentences

        # Act
        trained_learner.teach(queried_samples, resume=True)

        # Assert
        assert model_id == id(trained_learner.trained_tagger)
