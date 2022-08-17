import json
from collections import defaultdict
from typing import List

from flair.data import Corpus, Sentence


def assign_id_corpus(corpus: Corpus) -> Corpus:
    """Assign ids to corpus

    Args:
        corpus (Corpus): The Corpus class in flair

    Returns:
        Corpus: The corpus with ids
    """
    id = 0
    for sentence in corpus.train:
        sentence.id = id
        id += 1
    for sentence in corpus.dev:
        sentence.id = id
        id += 1
    for sentence in corpus.test:
        sentence.id = id
        id += 1

    return corpus


def output_labeled_data(
    sentences: List[Sentence],
    file_path: str,
    file_format: str = "conll",
    tag_type: str = "ner",
) -> None:
    """Output dataset as conll format.

    Args:
        sentences (List[Sentence]): List of sentences.
        file_path (str): Path to save file.
        file_format (str, optional): Output file format. Defaults to "conll". Or "json"
        tag_type (str, optional): Output tag type. Defaults to "ner".
    """
    if file_format == "conll":
        with open(file_path, "w", encoding="utf-8") as file:
            for sent in sentences:
                for token in sent:
                    line = (
                        f"{token.text}\t{token.get_tag(tag_type).value}\n"  # noqa: E731
                    )
                    file.write(line)
                file.write("\n")
    elif file_format == "json":
        data = []
        for sent in sentences:
            sent_dict = {"text": [], "labels": []}
            for token in sent:
                sent_dict["text"].append(token.text)
                sent_dict["labels"].append(token.get_tag(tag_type).value)
            data.append(sent_dict)

        json_object = json.dumps(data, indent=4)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(json_object)
    else:
        raise NameError("The file_format must be 'conll' or 'json'.")


def add_tags(query_labels: List[dict]) -> List[Sentence]:
    """Add tags to create sentences.

    Args:
        query_labels (List[dict]): Each dictionary contians text and labels.
                                   Example:
                                    [
                                        {
                                            "text": "I love Berlin .",
                                            "labels": [
                                            {
                                                "start_pos": 7,
                                                "text": "Berlin",
                                                "label": "B-LOC"
                                            }
                                            ]
                                        }
                                    ]

    Returns:
        List[Sentence]: A list of sentences.
    """
    annotated_sentences = []
    for sent in query_labels:
        sentence = Sentence(sent["text"])
        sent_labels = sent["labels"]
        if len(sent_labels) != 0:
            for token in sentence:
                for token_label_info in sent_labels:
                    if (
                        token.start_pos == token_label_info["start_pos"]
                        and token.text == token_label_info["text"]
                    ):
                        token.add_tag("ner", token_label_info["label"])
                        break
                    token.add_tag("ner", "O")
        else:
            for token in sentence:
                token.add_tag("ner", "O")
        annotated_sentences.append(sentence)

    return annotated_sentences


def entity_ratio(sentences: List[Sentence], tag_type: str = "ner") -> float:
    """Calculate entity ratio of a dataset

    https://arxiv.org/abs/1701.02877

    Args:
        sentences (List[Sentence]): Sentence class list
        tag_type (str, optional): Tag type. Defaults to "ner".

    Returns:
        float: Entity ratio.
    """
    entity_counter = defaultdict(int)

    for sent in sentences:
        for span in sent.get_spans(tag_type):
            entity_counter[span.text] += 1

    if not list(entity_counter.keys()):
        return float(0)

    return float(sum(entity_counter.values()) / len(entity_counter.keys()))


def load_plain_text(file_path: str) -> List[Sentence]:
    """Load plain dataset"""
    sentences = []
    with open(file_path, mode="r", encoding="utf-8") as f:
        for line in f:
            sentences.append(Sentence(line))
    return sentences


def count_tokens(sentences: List[Sentence]) -> int:
    """Count tokens in sentences"""
    return sum(len(s.tokens) for s in sentences)


def bilou2bioes(tags: List[str]) -> List[str]:
    """Convert BILOU format to BIOES format

    Args:
        tags (List[str]): List of tags with BILOU format.

    Returns:
        List[str]: List of tags with BIOES format.
    """
    new_tags = []
    for tag in tags:
        tag_list = list(tag)
        if tag[0] == "L":
            tag_list[0] = "E"
        elif tag[0] == "U":
            tag_list[0] = "S"
        tag = "".join(tag_list)
        new_tags.append(tag)
    return new_tags


def bilou2bio(tags: List[str]) -> List[str]:
    """Convert BILOU format to BIO format

    Args:
        tags (List[str]): List of tags with BILOU format.

    Returns:
        List[str]: List of tags with BIO format.
    """
    new_tags = []
    for tag in tags:
        tag_list = list(tag)
        if tag[0] == "L":
            tag_list[0] = "I"
        elif tag[0] == "U":
            tag_list[0] = "B"
        tag = "".join(tag_list)
        new_tags.append(tag)
    return new_tags


def bioes2bio(tags: List[str]) -> List[str]:
    """Convert BIOES format to BIO format

    Args:
        tags (List[str]): List of tags with BIOES format.

    Returns:
        List[str]: List of tags with BIO format.
    """
    new_tags = []
    for tag in tags:
        tag_list = list(tag)
        if tag[0] == "E":
            tag_list[0] = "I"
        elif tag[0] == "S":
            tag_list[0] = "B"
        tag = "".join(tag_list)
        new_tags.append(tag)
    return new_tags


def bio2bioes(tags: List[str]) -> List[str]:
    """Convert BIOES format to BIO format

    Args:
        tags (List[str]): List of tags with BIOES format.

    Returns:
        List[str]: List of tags with BIO format.
    """
    new_tags = []
    for i, tag in enumerate(tags):
        tag_list = list(tag)
        if tag[0] == "B":
            if (i == len(tags) - 1) or ("I-" not in tags[i + 1]):
                tag_list[0] = "S"
        elif tag[0] == "I":
            if (i == len(tags) - 1) or ("I-" not in tags[i + 1]):
                tag_list[0] = "E"

        tag = "".join(tag_list)
        new_tags.append(tag)
    return new_tags
