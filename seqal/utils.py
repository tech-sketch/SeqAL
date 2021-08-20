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


def output_conll_format(sents: List[Sentence], save_path: str) -> None:
    """Output dataset as conll format.

    Args:
        sents (List[Sentence]): [description]
        save_path (str): [description]
    """
    with open(save_path, "w") as f:
        for sent in sents:
            for token in sent:
                line = f"{token.text}\t{token.get_tag('pos').value}\t{token.get_tag('ner').value}\n"  # noqa: E731
                f.write(line)
            f.write("\n")


def add_tags(query_labels: List[dict]) -> List[Sentence]:
    """Add tags to create sentences.

    Args:
        query_labels (List[dict]): Each dictionary contians text and labels.

    Returns:
        List[Sentence]: A list of sentences.
    """
    annotated_sents = []
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
        annotated_sents.append(sentence)

    return annotated_sents
