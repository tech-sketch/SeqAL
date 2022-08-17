import string
from typing import List, Tuple

from flair.data import Sentence
from spacy.language import Language

from seqal import utils


class Aligner:
    """Align tokens and tags."""

    def concat_tags(self, tags: List[str]) -> str:
        """Concat tags togeter

        Args:
            tags (List[str]): List of tags. The schema should be BIO

        Examples:
            ["O", "O"]
            ->
            O

            ['B-LOC', 'I-LOC', 'I-LOC', 'I-LOC']
            ->
            B-LOC

            ['I-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'I-LOC']
            ->
            ['I-LOC']

        Returns:
            str: tag
        """
        return tags[0]

    def align_spaced_language(
        self, sentence: List[str], tags: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Align token and tags for spaced language.

        Args:
            sentence (List[str]): A list of character.
            tags (List[str]): A list of tags

        Returns:
            sentence (List[str]): A list of token.
            tags (List[str]): A list of tags.
        """
        final_sentence = []
        final_tags = []

        tokens = []
        token_tags = []
        for i, character in enumerate(sentence):
            if i == len(sentence) - 1 and character not in string.punctuation:
                tokens.append(character)
                token_tags.append(tags[i])
                final_sentence.append("".join(tokens))
                final_tags.append(self.concat_tags(token_tags))

            elif character in string.punctuation:
                final_sentence.append("".join(tokens))
                final_tags.append(self.concat_tags(token_tags))
                tokens = []
                token_tags = []
                final_sentence.append(character)
                final_tags.append(tags[i])

            elif character != " ":
                tokens.append(character)
                token_tags.append(tags[i])

            else:
                final_sentence.append("".join(tokens))
                final_tags.append(self.concat_tags(token_tags))
                tokens = []
                token_tags = []

        return final_sentence, final_tags

    def align_non_spaced_language(
        self, sentence: List[str], tags: List[str], spacy_model: Language
    ) -> Tuple[List[str], List[str]]:
        """Align token and tags for non-spaced language.

        Args:
            sentence (List[str]): A list of character
            tags (List[str]): A list of tags
            spacy_model (Language): Spacy language model

        Returns:
            sentence (List[str]): A list of token
            tags (List[str]): A list of tags
        """
        doc = spacy_model("".join(sentence))

        final_sentence = []
        final_tags = []

        for token in doc:
            final_sentence.append(token.text)
            token_start = token.idx
            token_end = token.idx + len(token.text)

            token_tags = tags[token_start:token_end]
            final_tags.append(self.concat_tags(token_tags))

        return final_sentence, final_tags

    def to_subword_spaced_language(
        self,
        sentence: List[str],
        tags: List[str],
        input_schema: str = "BIO",
        output_schema: str = "BIO",
    ) -> Tuple[List[str], List[str]]:
        """Convert character form to token/subword form for sentence and tags.

        Args:
            sentence (List[str]): A list of character.
                Example of spaced language: ['T', 'o', 'k', 'y', 'o', ' ', 'c', 'i', 't', 'y']
            tags (List[str]): A list of tags
                Example of spaced language: ["B-LOC", "I-LOC", "I-LOC", "I-LOC", "I-LOC", 'O', 'O', 'O', 'O', 'O']
            input_schema (str, optional): Input tag shema. Defaults to "BIO". Support "BIO", "BILOU", "BIOES"
            output_schema (str, optional): Output tag shema. Defaults to "BIO". Support "BIO", "BIOES"
                                           Flair don't support "BILOU", so we don't output this schema.

        Returns:
            sentence (List[str]): A list of token.
                Example of spaced language: ['Tokyo', 'is', 'a', 'city']
            tags (List[str]): A list of tags.
                Example of spaced language: ["B-LOC", "O", "O", "O"]
        """
        if input_schema == "BILOU":
            tags = utils.bilou2bio(tags)
        elif input_schema == "BIOES":
            tags = utils.bioes2bio(tags)

        final_sentence, final_tags = self.align_spaced_language(
            sentence=sentence, tags=tags
        )

        if output_schema == "BIOES":
            tags = utils.bio2bioes(tags)

        return final_sentence, final_tags

    def to_subword_non_spaced_language(
        self,
        sentence: List[str],
        tags: List[str],
        spacy_model: Language,
        input_schema: str = "BIO",
        output_schema: str = "BIO",
    ) -> Tuple[List[str], List[str]]:
        """Convert character form to token/subword form for sentence and tags.

        Args:
            sentence (List[str]): A list of character.
                Example of non-spaced language: ['ロ', 'ン', 'ド', 'ン', 'は', '都', '市', 'で', 'す']
            tags (List[str]): A list of tags
                Example of non-spaced language: ["B-LOC", "I-LOC", "I-LOC", "I-LOC", "O", "O", "O", "O", "O"]
            spacy_model (Language): Spacy language model
            input_schema (str, optional): Input tag shema. Defaults to "BIO". Support "BIO", "BILOU", "BIOES"
            output_schema (str, optional): Output tag shema. Defaults to "BIO". Support "BIO", "BIOES"
                                           Flair don't support "BILOU", so we don't output this schema.

        Returns:
            sentence (List[str]): A list of token.
                Example of non-spaced language: ['ロンドン', 'は', '都市', 'です']
            tags (List[str]): A list of tags.
                Example of non-spaced language: ["B-LOC", "O", "O", "O"]
        """
        if input_schema == "BILOU":
            tags = utils.bilou2bio(tags)
        elif input_schema == "BIOES":
            tags = utils.bioes2bio(tags)

        final_sentence, final_tags = self.align_non_spaced_language(
            sentence=sentence, tags=tags, spacy_model=spacy_model
        )

        if output_schema == "BIOES":
            tags = utils.bio2bioes(tags)

        return final_sentence, final_tags

    def add_tags_on_char_spaced_language(
        self,
        labled_data: List[dict],
        input_schema: str = "BIO",
        output_schema: str = "BIO",
        tag_type: str = "ner",
    ) -> List[Sentence]:
        """Add tags to sentence on character based

        Args:
            labled_data (List[dict]): A list of labeled data.
                Example of spaced language:
                    [
                        {
                            "text": ['T', 'o', 'k', 'y', 'o', ' ', 'c', 'i', 't', 'y'],
                            "labels": ["B-LOC", "I-LOC", "I-LOC", "I-LOC", "I-LOC", 'O', 'O', 'O', 'O', 'O']
                        },
                    ]
            input_schema (str, optional): Input tag shema. Defaults to "BIO". Support "BIO", "BILOU", "BIOES"
            output_schema (str, optional): Output tag shema. Defaults to "BIO". Support "BIO", "BIOES"
                                           Flair don't support "BILOU", so we don't output this schema.
            tag_type (str, optional): Tag type. Defaults to "ner".

        Returns:
            annotated_sentence (List[Sentence]): A list of sentence.
        """
        annotated_sentence = []
        for sample in labled_data:
            sentence = sample["text"]
            tags = sample["labels"]

            subword_sentence, subword_tags = self.to_subword_spaced_language(
                sentence=sentence,
                tags=tags,
                input_schema=input_schema,
                output_schema=output_schema,
            )

            sentence = Sentence(subword_sentence)
            for i, tag in enumerate(subword_tags):
                sentence[i].add_tag(tag_type, tag)
            annotated_sentence.append(sentence)

        return annotated_sentence

    def add_tags_on_char_non_spaced_language(
        self,
        labled_data: List[dict],
        input_schema: str = "BIO",
        output_schema: str = "BIO",
        tag_type: str = "ner",
        spacy_model: Language = None,
    ) -> List[Sentence]:
        """Add tags to sentence on character based

        Args:
            labled_data (List[dict]): A list of labeled data.
                Example of non-spaced language:
                    [
                        {
                            "text": ['ロ', 'ン', 'ド', 'ン', 'は', '都', '市', 'で', 'す'],
                            "labels": ["B-LOC", "I-LOC", "I-LOC", "E-LOC", "O", "O", "O", "O", "O"]
                        }
                    ]
            input_schema (str, optional): Input tag shema. Defaults to "BIO". Support "BIO", "BILOU", "BIOES"
            output_schema (str, optional): Output tag shema. Defaults to "BIO". Support "BIO", "BIOES"
                                        Flair don't support "BILOU", so we don't output this schema.
            tag_type (str, optional): Tag type. Defaults to "ner".
            spacy_model (Language): Spacy language model

        Returns:
            annotated_sentence (List[Sentence]): A list of sentence.
        """
        annotated_sentence = []
        for sample in labled_data:
            sentence = sample["text"]
            tags = sample["labels"]

            subword_sentence, subword_tags = self.to_subword_non_spaced_language(
                sentence=sentence,
                tags=tags,
                spacy_model=spacy_model,
                input_schema=input_schema,
                output_schema=output_schema,
            )

            sentence = Sentence(subword_sentence)
            for i, tag in enumerate(subword_tags):
                sentence[i].add_tag(tag_type, tag)
            annotated_sentence.append(sentence)

        return annotated_sentence

    def add_tags_on_token(
        self, labled_data: List[dict], tag_type: str = "ner"
    ) -> List[Sentence]:
        """Add tags to sentence on token based

        Args:
            labled_data (List[dict]): A list of labeled data.
                Example of spaced language:
                    [
                        {
                            "text": ['Tokyo', 'is', 'a', 'city'],
                            "labels": ['B-LOC', 'O', 'O', 'O']
                        }
                    ]
                Example of non-spaced language:
                    [
                        {
                            "text": ['ロンドン', 'は', '都市', 'です'],
                            "labels": ['B-LOC', 'O', 'O', 'O']
                        }
                    ]
            tag_type (str, optional): Tag type. Defaults to "ner".

        Returns:
            annotated_sentence (List[Sentence]): A list of sentence.
        """
        annotated_sentence = []
        for sample in labled_data:
            sentence = Sentence(sample["text"])
            for i, tag in enumerate(sample["labels"]):
                sentence[i].add_tag(tag_type, tag)
            annotated_sentence.append(sentence)

        return annotated_sentence
