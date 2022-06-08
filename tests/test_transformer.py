from unittest.mock import MagicMock

from flair.data import Sentence

from seqal.transformer import Transformer


class TestTransformer:
    """Test Transformer class"""

    def test_to_char_for_non_space_language(self) -> None:
        """Test Transform.to_char to convert subword form to character form for non-spaced language"""
        # Arrange
        nlp = MagicMock()
        sentence = Sentence(["ロンドン", "は", "大都市", "です"])
        sentence[0].set_label("ner", "B-LOC")
        sentence[1].set_label("ner", "O")
        sentence[2].set_label("ner", "O")
        sentence[3].set_label("ner", "O")

        expected = ["B-LOC", "I-LOC", "I-LOC", "E-LOC", "O", "O", "O", "O", "O", "O"]

        # Act
        transformer = Transformer(nlp)
        new_sentence = transformer.to_char(sentence)
        token_tag = [token.get_tag("ner").value for token in new_sentence]

        # Assert
        assert expected == token_tag
