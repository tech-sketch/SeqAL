from pathlib import Path

from seqal.active_learner import ActiveLearner


class TestActiveLearner:
    def test_fit(self, fixture_path: Path, learner: ActiveLearner) -> None:
        save_path = fixture_path / "output"
        learner.fit(save_path)
        del learner
