import random
from typing import List, Tuple

import numpy as np
from flair.data import Sentence
from torch.nn import Module


def predict_data_pool(sents: List[Sentence], estimator: Module) -> None:
    """Predict on data pool for query.

    Args:
        corpus (Corpus): Corpus contains train(labeled data), dev, test (data pool)
        estimator (Module): Trained model.
    """
    for sent in sents:
        estimator.predict(sent)


def remove_query_samples(sents: List[Sentence], query_idx: List[int]) -> None:
    """Remove queried data from data pool.

    Args:
        corpus (Corpus): Corpus contains train(labeled data), dev, test (data pool)
        query_idx (List[int]): Index list of queried data.
    """
    new_sents = []
    query_sents = []
    for i, sent in enumerate(sents):
        if i in query_idx:
            query_sents.append(sent)
        else:
            new_sents.append(sent)
    return new_sents, query_sents


def random_sampling(
    sents: List[Sentence], estimator=None, query_number=0, seed=0
) -> Tuple[int, Sentence]:
    """Random select data from pool.

    Args:
        corpus (Corpus): Corpus contains train(labeled data), dev, test (data pool)
        estimator (None): Random sampling does not need estimator. Here is a placeholder.
        seed (int, optional): [description]. Defaults to 0.

    Returns:
        Tuple[int, Sentence]: The queried id and instance.
    """
    random.seed(seed)

    n_samples = len(sents)
    if query_number == 0:
        query_idx = random.choice(range(n_samples))
        query_idx = [query_idx]
    else:
        if query_number > len(sents):
            query_idx = random.sample(range(len(sents)), len(sents))
        else:
            query_idx = random.sample(range(len(sents)), query_number)

    sents, query_samples = remove_query_samples(sents, query_idx)

    return sents, query_samples


def ls_sampling(
    sents: List[Sentence], estimator: Module, query_number: int = 0
) -> Tuple[int, Sentence]:
    """Least confidence sampling.

    Args:
        corpus (Corpus): Corpus contains train(labeled data), dev, test (data pool)
        estimator (Module): Sequence tagger.

    Returns:
        Tuple[int, Sentence]: The queried id and instance.
    """
    # Predict on data pool
    predict_data_pool(sents, estimator)

    # Select on data pool
    probs = np.ones(len(sents)) * float("Inf")

    for i, sent in enumerate(sents):
        scores = [entity.score for entity in sent.get_spans("ner")]
        if scores != []:
            probs[i] = 1 - max(scores)

    ascending_indices = list(np.argsort(probs))

    if query_number == 0:
        query_idx = ascending_indices[0]
        query_idx = [query_idx]
    else:
        if query_number > len(sents):
            query_idx = ascending_indices
        else:
            query_idx = ascending_indices[:query_number]

    # Remove selected sample from data pool
    sents, query_samples = remove_query_samples(sents, query_idx)

    return sents, query_samples


def mnlp_sampling(
    sents: List[Sentence], estimator: Module, query_number: int = 0
) -> Tuple[int, Sentence]:
    """Least confidence sampling.

    Args:
        corpus (Corpus): Corpus contains train(labeled data), dev, test (data pool)
        estimator (Module): Sequence tagger.

    Returns:
        Tuple[int, Sentence]: The queried id and instance.
    """
    # Predict on data pool
    predict_data_pool(sents, estimator)

    # Select on data pool
    probs = np.ones(len(sents)) * float("-Inf")

    for i, sent in enumerate(sents):
        scores = [entity.score for entity in sent.get_spans("ner")]
        if scores != []:
            probs[i] = max(scores) / len(sent)

    descend_indices = np.argsort(-probs)

    if query_number == 0:
        query_idx = descend_indices[0]
        query_idx = [query_idx]
    else:
        if query_number > len(sents):
            query_idx = descend_indices
        else:
            query_idx = descend_indices[:query_number]

    # Remove selected sample from data pool
    sents, query_samples = remove_query_samples(sents, query_idx)

    return sents, query_samples
