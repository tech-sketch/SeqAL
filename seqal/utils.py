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
