from pathlib import Path

from seqal.active_learner import ActiveLearner, query_data_by_indices
from seqal.datasets import Corpus


def test_query_data_by_indices(corpus: Corpus) -> None:
    ordered_indices = list(range(10))

    # Query single sentence
    query_idx = query_data_by_indices(
        corpus.train.sentences, ordered_indices, query_number=0, token_based=False
    )
    assert query_idx == [0]

    # Query multiple sentences
    query_idx = query_data_by_indices(
        corpus.train.sentences, ordered_indices, query_number=2, token_based=False
    )

    assert query_idx == [0, 1]

    # Batch multiple sentences based on token count
    token_required = 12
    query_idx = query_data_by_indices(
        corpus.train.sentences,
        ordered_indices,
        query_number=token_required,
        token_based=True,
    )
    assert query_idx == [0, 1, 2]


class TestActiveLearner:
    def test_fit(self, fixture_path: Path, learner: ActiveLearner) -> None:
        save_path = fixture_path / "output"
        learner.fit(save_path)
        del learner
