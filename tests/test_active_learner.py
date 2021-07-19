from pathlib import Path

from flair.data import Sentence

from seqal.active_learner import ActiveLearner


class TestActiveLearner:
    def test_fit(self, fixture_path: Path, learner: ActiveLearner) -> None:
        save_path = fixture_path / "output"
        learner.fit(save_path)
        del learner

    def test_query(self, fixture_path: Path, learner: ActiveLearner) -> None:
        save_path = fixture_path / "output"
        learner.fit(save_path)

        query_id, query_inst = learner.query()
        assert isinstance(query_id, int) is True
        assert isinstance(query_inst, Sentence) is True
