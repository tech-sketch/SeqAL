import random
from typing import List

import numpy as np
from flair.data import Sentence
from sklearn.cluster import KMeans
from torch import nn, stack, tensor


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
    tagger = kwargs["tagger"]
    probs = tagger.log_probability(sents).exp()
    indices = (1 - probs).argsort().cpu().tolist()
    return indices


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
    tagger = kwargs["tagger"]
    log_probs = tagger.log_probability(sents)
    lengths = tensor([len(sent) for sent in sents], device=log_probs.device)
    normed_log_probs = log_probs / lengths
    indices = normed_log_probs.argsort().cpu().tolist()
    return indices


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


def cluster_sampling(sents: List[Sentence], tag_type: str, **kwargs) -> List[int]:
    """Cluster sampling.

    We create cluster sampling as a kind of diversity sampling method.
    Different with most of sampling methods that are based on sentence level,
    Cluster sampling method is implemented on entity level.
    Cluster sampling classify all entity into cluster, and find the centen in each cluster.
    We calculate the similarity between center and entity in the same cluster,
    the low similarity pair means high diversity.

    Args:
        sents (List[Sentence]): [description]
        tag_type (str): [description]

    Returns:
        List[int]: [description]
    """
    label_names = kwargs["label_names"]
    if "O" in label_names:
        label_names.remove("O")
    embeddings = kwargs["embeddings"]
    embedding_dim = None

    # Get entities in each class, each entity has {sent_idx, token_idx, token_text, token_embedding}
    label_entity_list = []

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
                label_entity_list.append(tag_info)

    # Get all entity embedding matrix
    entity_embedding_matrix = [tag["token_embedding"] for tag in label_entity_list]
    if entity_embedding_matrix == []:
        return random_sampling(sents)
    else:
        entity_embedding_matrix = stack(entity_embedding_matrix)

    # Clustering
    kmeans = KMeans(n_clusters=len(label_names))
    kmeans.fit(entity_embedding_matrix)

    cluster_centers_matrix = kmeans.cluster_centers_
    entity_labels = kmeans.labels_

    # Find the center in matrix
    center_cluster_num = {}  # {center_num_in_cluster: center_index_in_matrix}
    for i, token_matrix in enumerate(entity_embedding_matrix):
        for center_matrix in cluster_centers_matrix:
            if center_matrix == token_matrix:
                center_num_in_cluster = entity_labels[i]
                center_cluster_num[center_num_in_cluster] = i

    # Find the entity in each cluster
    label_entity_cluster = {
        cluster_num: {"cluster_center_idx": 0, "cluster_member_idx": []}
        for cluster_num in center_cluster_num.keys()
    }
    for cluster_num in label_entity_cluster.keys():
        label_entity_cluster[cluster_num]["cluster_center"] = center_cluster_num[
            cluster_num
        ]
        for i, entity_cluster_num in enumerate(entity_labels):
            if entity_cluster_num == cluster_num:
                label_entity_cluster[cluster_num]["cluster_member_idx"].append(i)

    # Calculate each the similarity between center and entities
    for cluster_num, cluster_info in label_entity_cluster.items():
        center_idx = cluster_info["cluster_center_idx"]
        scores = []
        for member_idx in cluster_info["cluster_member_idx"]:
            cos = nn.CosineSimilarity(dim=embedding_dim)
            cosine_score = cos(
                entity_embedding_matrix[center_idx], entity_embedding_matrix[member_idx]
            )
            scores.append(cosine_score)
        label_entity_cluster["sim_scores"] = scores

    # Used for debug the order
    for cluster_num, cluster_info in label_entity_cluster.items():
        cluster_member_idx = cluster_info["cluster_member_idx"]
        sim_scores = cluster_info["sim_scores"]

        cluster_info["sim_scores"] = [
            x for _, x in sorted(zip(sim_scores, cluster_member_idx))
        ]
        cluster_info["cluster_member_idx"] = sorted(sim_scores)

    # Flat the entity score
    entity_scores = [0] * len(label_entity_list)
    for cluster_num, cluster_info in label_entity_cluster.items():
        for i, member_idx in enumerate(cluster_info["cluster_member_idx"]):
            entity_scores[member_idx] += cluster_info["sim_scores"][i]

    # Reorder the sentence index
    sentence_scores = [99] * len(sents)
    for entity_idx, entity_info in enumerate(label_entity_list):
        sent_idx = entity_info["sent_idx"]
        sentence_scores[sent_idx] += entity_scores[entity_idx]

    ascend_indices = np.argsort(sentence_scores)

    return ascend_indices
