import random
from typing import List

import numpy as np
from flair.data import Sentence
from torch import nn


def random_sampling(
    sents: List[Sentence], tag_type: str = None, seed: int = 0, **kwargs
) -> List[int]:
    """Random select data from pool.

    Args:
        sents (List[Sentence]): Sentences in data pool.
        tag_type (str): Tag type to predict. This is a placeholder for random sampling method.
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        List[int]:
            query_idx: The index of queried samples in sents.
    """
    random.seed(seed)

    random_idx = list(range(len(sents)))
    random.shuffle(random_idx)

    return random_idx


def lc_sampling(sents: List[Sentence], tag_type: str, **kwargs) -> List[int]:
    """Least confidence sampling.

    https://dl.acm.org/doi/10.5555/1619410.1619452

    Args:
        sents (List[Sentence]): Sentences in data pool.
        tag_type (str): Tag type to predict.

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

    return descend_indices


def mnlp_sampling(sents: List[Sentence], tag_type: str, **kwargs) -> List[int]:
    """Maximum Normalized Log-Probability sampling.

    https://arxiv.org/abs/1707.05928

    Args:
        sents (List[Sentence]): Sentences in data pool.
        tag_type (str): Tag type to predict.

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

    return ascend_indices


def similarity_sampling(sents: List[Sentence], tag_type: str, **kwargs) -> List[int]:
    """Similarity sampling

    We create similarity sampling as a kind of diversity sampling method.
    Different with most of sampling methods that are based on sentence level,
    Similarity sampling method is implemented on entity level.
    We calculate the similarity between entity pair, the low similarity pair means high diversity.

    Args:
        sents (List[Sentence]): flair sentences
        tag_type (str): label type, e.g. "ner"
    kwargs:
        label_names (List[str]): label name of all dataset
        embeddings: the embeddings method

    Returns:
        List[int]:
            query_idx: The index of queried samples in sents.
    """
    label_names = kwargs["label_names"]
    if "O" in label_names:
        label_names.remove("O")
    embeddings = kwargs["embeddings"]
    embedding_dim = None

    # Get entities in each class, each entity has {sent_idx, token_idx, token_text, token_embedding}
    label_entity_list = {label: [] for label in label_names}
    for label in label_names:
        for sent_idx, sent in enumerate(sents):
            if len(sent.get_spans(tag_type)) != 0:
                embeddings.embed(sent)
                for token_idx, token in enumerate(sent):
                    tag = token.get_tag("ner")
                    if tag.value == "O":  # tag.value is the label name
                        continue  # Skip the "O" label
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

    ascend_indices = np.argsort(sentence_score)

    return ascend_indices
