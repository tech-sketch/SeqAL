from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest
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


def test_entity_ratio_return_0(unlabeled_sentences: List[Sentence]) -> None:
    """Test entity_ratio if no entities"""
    # Act
    result = utils.entity_ratio(unlabeled_sentences)

    # Assert
    assert result == 0


def test_entity_ratio() -> None:
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


def test_count_tokens(unlabeled_sentences: List[Sentence]) -> None:
    """Test entity_ratio if no entities"""
    # Act
    result = utils.count_tokens(unlabeled_sentences[:2])

    # Assert
    assert result == 11


def test_bilou2bioes() -> None:
    """Test bilou2bioes conversion"""
    # Arrange
    tags = ["B-X", "I-X", "L-X", "U-X", "O"]
    expected = ["B-X", "I-X", "E-X", "S-X", "O"]

    # Act
    result = utils.bilou2bioes(tags)

    # Assert
    assert result == expected


def test_bilou2bio() -> None:
    """Test bilou2bio conversion"""
    # Arrange
    tags = ["B-X", "I-X", "L-X", "U-X", "O"]
    expected = ["B-X", "I-X", "I-X", "B-X", "O"]

    # Act
    result = utils.bilou2bio(tags)

    # Assert
    assert result == expected


def test_bioes2bio() -> None:
    """Test bioes2bio conversion"""
    # Arrange
    tags = ["B-X", "I-X", "E-X", "S-X", "O"]
    expected = ["B-X", "I-X", "I-X", "B-X", "O"]

    # Act
    result = utils.bioes2bio(tags)

    # Assert
    assert result == expected


@pytest.mark.parametrize(
    "tags,expected",
    [
        (["B-X", "I-X", "I-X", "B-X", "O"], ["B-X", "I-X", "E-X", "S-X", "O"]),
        (["B-X", "I-X", "I-X", "B-X"], ["B-X", "I-X", "E-X", "S-X"]),
    ],
)
def test_bio2bioes(tags: List[str], expected: List[str]) -> None:
    """Test bio2bioes conversion"""
    # Act
    result = utils.bio2bioes(tags)

    # Assert
    assert result == expected
