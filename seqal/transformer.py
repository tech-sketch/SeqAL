from flair.data import Sentence
from spacy.language import Language


class Transformer:
    """Transform sentence to character or subword form"""

    def __init__(self, nlp: Language) -> None:
        self.lang = nlp.lang
        self.nlp = nlp

    def to_subword(self, sentence: str) -> Sentence:
        """Convert non-space language to subword form.

        Args:
            sentence (str): Sentence string.

        Returns:
            Sentence: Sentence class with subword form.
        """
        doc = self.nlp(sentence)
        new_sentence = Sentence([token.text for token in doc])
        return new_sentence

    def to_char(self, sentence: Sentence, tag_type: str = "ner") -> Sentence:
        """Convert subword form to character form.

        Args:
            sentence (Sentence): Sentence class tagged with subword form
            tag_type (str, optional): Tag type. Defaults to "ner".

        Returns:
            Sentence: Sentence class tagged with character form
        """
        new_sentence = Sentence(list(sentence.to_original_text().replace(" ", "")))
        char_idx_in_setence = 0
        for token in sentence:
            tag = token.get_tag("ner").value
            if tag == "O":
                for _ in token.text:
                    new_sentence[char_idx_in_setence].set_label(tag_type, tag)
                    char_idx_in_setence += 1
            else:
                for i, _ in enumerate(token.text):
                    if i == 0:
                        new_sentence[char_idx_in_setence].set_label(tag_type, tag)
                    else:
                        tag_list = list(tag)
                        if i == len(token.text) - 1:
                            tag_list[0] = "E"
                        else:
                            tag_list[0] = "I"
                        new_sentence[char_idx_in_setence].set_label(
                            tag_type, "".join(tag_list)
                        )
                    char_idx_in_setence += 1

        return new_sentence
