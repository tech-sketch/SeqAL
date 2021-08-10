import random
from typing import List

import numpy as np
from flair.data import Sentence


def random_sampling(
    sents: List[Sentence],
    tag_type: str = None,
    query_number=0,
    token_based=False,
    seed=0,
) -> List[int]:
    """Random select data from pool.

    Args:
        sents (List[Sentence]): Sentences in data pool.
        tag_type (str): Tag type to predict. This is a placeholder for random sampling method.
        query_number (int, optional): Batch query number. Defaults to 0.
        token_based (bool, optional): If true, using query number as token number to query data.
                                      If false, using query number as sentence number to query data.
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        List[int]:
            query_idx: The index of queried samples in sents.
    """
    random.seed(seed)

    if token_based is True:
        # Shuffle index
        random_idx = list(range(len(sents)))
        random.shuffle(random_idx)

        queried_tokens = 0
        query_idx = []
        for indx in random_idx:
            sent = sents[indx]
            if queried_tokens < query_number:
                queried_tokens += len(sent.tokens)
                query_idx.append(indx)
    else:
        n_samples = len(sents)
        if query_number == 0:
            query_idx = random.choice(range(n_samples))
            query_idx = [query_idx]
        else:
            if query_number > len(sents):
                query_idx = random.sample(range(len(sents)), len(sents))
            else:
                query_idx = random.sample(range(len(sents)), query_number)

    return query_idx


def ls_sampling(
    sents: List[Sentence],
    tag_type: str,
    query_number: int = 0,
    token_based: bool = False,
) -> List[int]:
    """Least confidence sampling.

    Args:
        sents (List[Sentence]): Sentences in data pool.
        tag_type (str): Tag type to predict.
        query_number (int, optional): Batch query number. Defaults to 0.
        token_based (bool, optional): If true, using query number as token number to query data.
                                      If false, using query number as sentence number to query data.

    Returns:
        List[int]:
            query_idx: The index of queried samples in sents.
    """

    # Select on data pool
    probs = np.ones(len(sents)) * float("Inf")

    for i, sent in enumerate(sents):
        scores = [entity.score for entity in sent.get_spans(tag_type)]
        if scores != []:
            probs[i] = 1 - max(scores)

    ascending_indices = list(np.argsort(probs))

    if token_based is True:
        queried_tokens = 0
        query_idx = []
        for indx in ascending_indices:
            sent = sents[indx]
            if queried_tokens < query_number:
                queried_tokens += len(sent.tokens)
                query_idx.append(indx)
    else:
        if query_number == 0:
            query_idx = ascending_indices[0]
            query_idx = [query_idx]
        else:
            if query_number > len(sents):
                query_idx = ascending_indices
            else:
                query_idx = ascending_indices[:query_number]

    return query_idx


def mnlp_sampling(
    sents: List[Sentence],
    tag_type: str,
    query_number: int = 0,
    token_based: bool = False,
) -> List[int]:
    """Least confidence sampling.

    Args:
        sents (List[Sentence]): Sentences in data pool.
        tag_type (str): Tag type to predict.
        query_number (int, optional): Batch query number. Defaults to 0.
        token_based (bool, optional): If true, using query number as token number to query data.
                                      If false, using query number as sentence number to query data.

    Returns:
        List[int]:
            query_idx: The index of queried samples in sents.
    """
    # Select on data pool
    probs = np.ones(len(sents)) * float("-Inf")

    for i, sent in enumerate(sents):
        scores = [entity.score for entity in sent.get_spans(tag_type)]
        if scores != []:
            probs[i] = max(scores) / len(sent)

    descend_indices = np.argsort(-probs)

    if token_based is True:
        queried_tokens = 0
        query_idx = []
        for indx in descend_indices:
            sent = sents[indx]
            if queried_tokens < query_number:
                queried_tokens += len(sent.tokens)
                query_idx.append(indx)
    else:
        if query_number == 0:
            query_idx = descend_indices[0]
            query_idx = [query_idx]
        else:
            if query_number > len(sents):
                query_idx = descend_indices
            else:
                query_idx = descend_indices[:query_number]

    return query_idx
