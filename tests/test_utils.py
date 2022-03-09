from pathlib import Path
from typing import List
from unittest.mock import MagicMock

from flair.data import Sentence

from seqal import utils
from seqal.datasets import Corpus


def test_assign_id_corpus(corpus: Corpus) -> None:
    """Test assign_id_corpus function"""
    corpus_with_ids = utils.assign_id_corpus(corpus)
    train = corpus_with_ids.train
    dev = corpus_with_ids.dev
    test = corpus_with_ids.test

    assert train[0].id == 0
    assert dev[0].id == 10
    assert test[0].id == 15


def test_add_tags() -> None:
    """Test add_tags function"""
    query_labels = [
        {
            "text": "I love Berlin .",
            "labels": [{"start_pos": 7, "text": "Berlin", "label": "S-LOC"}],
        },
        {"text": "This book is great.", "labels": []},
    ]

    annotated_sents = utils.add_tags(query_labels)
    assert len(annotated_sents[0].get_spans("ner")) == 1
    assert len(annotated_sents[1].get_spans("ner")) == 0


def test_entity_ratio_return_0(unlabeled_sentences: List[Sentence]):
    """Test entity_ratio if no entities"""
    # Act
    result = utils.entity_ratio(unlabeled_sentences)

    # Assert
    assert result == 0


def test_entity_ratio():
    """Test entity_ratio return normal result"""
    # Arrange
    sentence1 = MagicMock()
    sentence1.get_spans = MagicMock(
        return_value=[MagicMock(text="TIS"), MagicMock(text="TIS")]
    )
    sentence2 = MagicMock()
    sentence2.get_spans = MagicMock(
        return_value=[MagicMock(text="INTEC"), MagicMock(text="TIS")]
    )

    sentences = [sentence1, sentence2]

    # Act
    result = utils.entity_ratio(sentences)

    # Assert
    assert result == float(2)


def test_load_plain_text(fixture_path: Path) -> None:
    """Test load_plain_text function"""
    # Arrange
    sentence1 = "Germany imported 47,600 sheep from Britain last year , nearly half of total imports ."
    file_path = fixture_path / "conll/plain_dataset.txt"

    # Act
    sentences = utils.load_plain_text(file_path)

    # Assert
    assert sentences[1].to_plain_string() == sentence1
