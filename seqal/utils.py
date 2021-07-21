from flair.data import Corpus


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
