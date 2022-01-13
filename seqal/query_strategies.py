import random
from typing import List

import numpy as np
from flair.data import Sentence
from torch import nn


def random_sampling(
    sents: List[Sentence],
    tag_type: str = None,
    query_number=0,
    token_based=False,
    seed=0,
    **kwargs
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
    **kwargs
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

    descend_indices = list(np.argsort(-probs))

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


def mnlp_sampling(
    sents: List[Sentence],
    tag_type: str,
    query_number: int = 0,
    token_based: bool = False,
    **kwargs
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

    ascend_indices = np.argsort(probs)

    if token_based is True:
        queried_tokens = 0
        query_idx = []
        for indx in ascend_indices:
            sent = sents[indx]
            if queried_tokens < query_number:
                queried_tokens += len(sent.tokens)
                query_idx.append(indx)
    else:
        if query_number == 0:
            query_idx = ascend_indices[0]
            query_idx = [query_idx]
        else:
            if query_number > len(sents):
                query_idx = ascend_indices
            else:
                query_idx = ascend_indices[:query_number]

    return query_idx


def similarity_sampling(
    sents: List[Sentence],
    tag_type: str,
    query_number: int = 0,
    token_based: bool = False,
    **kwargs
) -> List[int]:
    label_names = kwargs["label_names"]
    if "O" in label_names:
        label_names.remove("O")
    embeddings = kwargs["embeddings"]
    embedding_dim = None

    # Get entities in each class, each entity has {sent_idx, token_idx, token_text, token_embedding}
    label_entity_list = {label: [] for label in label_names}
    for label in label_names:
        for sent_idx, sent in enumerate(sents):
            if len(sent.get_spans("ner")) != 0:
                embeddings.embed(sent)
                for token_idx, token in enumerate(sent):
                    tag = token.get_tag("ner")
                    if (
                        tag.value == "O"
                    ):  # Skip if the "O" label. tag.value is the label name
                        continue
                    tag_info = {
                        "sent_idx": sent_idx,
                        "token_idx": token_idx,
                        "token_text": token.text,
                        "token_embedding": token.embedding,
                    }
                    if embedding_dim is None:
                        embedding_dim = len(token.embedding.shape) - 1
                    label_entity_list[tag.value].append(tag_info)

    # Calculate similarity of entity pair
    # (entity1, entity2, similarity)
    # {"ORG": [({sent_idx, token_idx, token_text, token_embedding}, {sent_idx, ...}, 0.98), (), ... ]}
    label_entity_pair_similarity = {label: [] for label in label_names}
    cos = nn.CosineSimilarity(dim=embedding_dim)
    for label, entity_list in label_entity_list.items():
        length = len(entity_list)
        for i in range(length - 1):
            for j in range(i + 1, length):
                cosine_score = cos(
                    entity_list[i]["token_embedding"], entity_list[j]["token_embedding"]
                )
                triple = (entity_list[i], entity_list[j], cosine_score)
                label_entity_pair_similarity[label].append(triple)

    # Reorder entity pair from low to high based on cosine_score.
    # If cosine_score is low, it means two entity are not similar, and the pair diversity is high
    for label, entity_pair_similarity_list in label_entity_pair_similarity.items():
        label_entity_pair_similarity[label] = sorted(
            entity_pair_similarity_list, key=lambda x: x[2]
        )

    sentence_score = [0] * len(sents)
    for label, entity_pair_similarity_list in label_entity_pair_similarity.items():
        for entity_pair in entity_pair_similarity_list:
            cosine_score = entity_pair[2]

            entity1 = entity_pair[0]
            sentence_score[entity1["sent_idx"]] += cosine_score

            entity2 = entity_pair[1]
            sentence_score[entity2["sent_idx"]] += cosine_score

    ascending_indices = np.argsort(sentence_score)

    # Query data
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
