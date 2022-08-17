from typing import List
from unittest.mock import MagicMock

import pytest

from seqal.aligner import Aligner


class TestAligner:
    @pytest.mark.parametrize(
        "tags,expected",
        [
            (["B-LOC", "I-LOC", "I-LOC", "I-LOC"], "B-LOC"),
            (["I-LOC", "I-LOC", "I-LOC"], "I-LOC"),
            (["O", "O"], "O"),
        ],
    )
    def test_concat_tags(self, tags: List[str], expected: str) -> None:
        """Test concat_tags"""
        # Arrage
        aligner = Aligner()

        # Act
        result = aligner.concat_tags(tags)

        # Assert
        assert result == expected

    @pytest.mark.parametrize(
        "sentence,tags,expected_sentence,expected_tags",
        [
            (
                [
                    "T",
                    "o",
                    "k",
                    "y",
                    "o",
                    " ",
                    "i",
                    "s",
                    " ",
                    "a",
                    " ",
                    "c",
                    "i",
                    "t",
                    "y",
                    ".",
                ],
                [
                    "B-LOC",
                    "I-LOC",
                    "I-LOC",
                    "I-LOC",
                    "I-LOC",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                ],
                ["Tokyo", "is", "a", "city", "."],
                ["B-LOC", "O", "O", "O", "O"],
            ),
            (
                [
                    "T",
                    "o",
                    "k",
                    "y",
                    "o",
                    ",",
                    "L",
                    "A",
                    " ",
                    "a",
                    " ",
                    "c",
                    "i",
                    "t",
                    "y",
                    ".",
                ],
                [
                    "B-LOC",
                    "I-LOC",
                    "I-LOC",
                    "I-LOC",
                    "I-LOC",
                    "O",
                    "B-LOC",
                    "I-LOC",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                ],
                ["Tokyo", ",", "LA", "a", "city", "."],
                ["B-LOC", "O", "B-LOC", "O", "O", "O"],
            ),
            (
                [
                    "L",
                    "o",
                    "s",
                    " ",
                    "a",
                    "n",
                    "g",
                    "e",
                    "l",
                    "e",
                    "s",
                    " ",
                    "i",
                    "s",
                    " ",
                    "a",
                    " ",
                    "c",
                    "i",
                    "t",
                    "y",
                ],
                [
                    "B-LOC",
                    "I-LOC",
                    "I-LOC",
                    "I-LOC",
                    "I-LOC",
                    "I-LOC",
                    "I-LOC",
                    "I-LOC",
                    "I-LOC",
                    "I-LOC",
                    "I-LOC",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                ],
                ["Los", "angeles", "is", "a", "city"],
                ["B-LOC", "I-LOC", "O", "O", "O"],
            ),
        ],
    )
    def test_align_spaced_language(
        self,
        sentence: List[str],
        tags: List[str],
        expected_sentence: List[str],
        expected_tags: List[str],
    ) -> None:
        """Test align_spaced_language"""
        # Arrage
        aligner = Aligner()

        # Act
        result_sentence, result_tags = aligner.align_spaced_language(sentence, tags)

        # Assert
        assert result_sentence == expected_sentence
        assert result_tags == expected_tags

    def test_align_non_spaced_language(self) -> None:
        """Test align_non_spaced_language"""
        # Arrage
        aligner = Aligner()
        sentence = ["ロ", "ン", "ド", "ン", "は", "都", "市", "で", "す"]
        tags = ["B-LOC", "I-LOC", "I-LOC", "I-LOC", "O", "B-NONE", "I-NONE", "O", "O"]
        expected_sentence = ["ロンドン", "は", "都市", "です"]
        expected_tags = ["B-LOC", "O", "B-NONE", "O"]

        doc = [
            MagicMock(text="ロンドン", idx=0),
            MagicMock(text="は", idx=4),
            MagicMock(text="都市", idx=5),
            MagicMock(text="です", idx=7),
        ]
        spacy_model = MagicMock(return_value=doc)

        # Act
        result_sentence, result_tags = aligner.align_non_spaced_language(
            sentence, tags, spacy_model
        )

        # Assert
        assert result_sentence == expected_sentence
        assert result_tags == expected_tags
