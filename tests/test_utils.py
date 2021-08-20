from seqal.datasets import Corpus
from seqal.utils import add_tags, assign_id_corpus


class TestUtils:
    def test_assign_id_corpus(self, corpus: Corpus) -> None:
        corpus_with_ids = assign_id_corpus(corpus)
        train = corpus_with_ids.train
        dev = corpus_with_ids.dev
        test = corpus_with_ids.test

        assert train[0].id == 0
        assert dev[0].id == 10
        assert test[0].id == 15

    def test_add_tags(self, corpus: Corpus) -> None:
        query_labels = [
            {
                "text": "I love Berlin .",
                "labels": [{"start_pos": 7, "text": "Berlin", "label": "S-LOC"}],
            },
            {"text": "This book is great.", "labels": []},
        ]

        annotated_sents = add_tags(query_labels)
        assert len(annotated_sents[0].get_spans("ner")) == 1
        assert len(annotated_sents[1].get_spans("ner")) == 0
