import random
from dataclasses import dataclass
from pickletools import float8
from typing import List

import numpy as np
import torch
from flair.data import Sentence
from flair.embeddings import StackedEmbeddings
from sklearn.cluster import KMeans


@dataclass
class Entity:
    id: int
    sent_id: int
    text: str
    embeddings: List[torch.Tensor]

    @property
    def vector(self) -> torch.Tensor:
        return torch.mean(torch.stack(self.embeddings), dim=0)


def sim_matrix(a: torch.tensor, b: torch.tensor, eps: float8 = 1e-8) -> torch.tensor:
    """Calculate similarity bewteen matrix

    Args:
        a (torch.tensor): Matrix of embedding. shape=(entity_count, embedding_dim)
        b (torch.tensor): Matrix of embedding. shape=(entity_count, embedding_dim)
        eps (float8, optional): Eps for numerical stability. Defaults to 1e-8.

    Returns:
        torch.tensor: similarity of matrix. shape=(entity_count, entity_count)
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def get_entity_list(
    sents: List[Sentence], embeddings: StackedEmbeddings, tag_type: str = "ner"
) -> List:
    """Get all entities from sentences

    Args:
        sents (List[Sentence]): Flair sentences
        tag_type (str): Label type, e.g. "ner"
        embeddings: The embeddings method

    Returns:
        List: Get all entities. Each entity has {sent_idx, token_idx, token_text, token_embedding}
    """
    label_entity_list = []
    for sent_idx, sent in enumerate(sents):
        if len(sent.get_spans(tag_type)) != 0:
            embeddings.embed(sent)
            for token_idx, token in enumerate(sent):
                tag = token.get_tag(tag_type)
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
                label_entity_list.append(tag_info)
    return label_entity_list


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
    log_probs = tagger.log_probability(sents)
    # to get descending order of "(1 - log_probs).argsort()"
    # we change it to "(log_probs - 1).argsort()"
    indices = (np.exp(log_probs) - 1).argsort()
    return indices.tolist()


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
    lengths = np.array([len(sent) for sent in sents])
    normed_log_probs = log_probs / lengths
    indices = normed_log_probs.argsort()
    return indices.tolist()


def similarity_sampling(
    sents: List[Sentence], tag_type: str = "ner", **kwargs
) -> List[int]:
    """Similarity sampling

    We create similarity sampling as a kind of diversity sampling method.
    Different with most of sampling methods that are based on sentence level,
    Similarity sampling method is implemented on entity level.
    We calculate the similarity between entity pair, the low similarity pair means high diversity.

    Args:
        sents (List[Sentence]): Flair sentences
        tag_type (str): Label type, e.g. "ner"
    kwargs:
        label_names (List[str]): Label name of all dataset
        embeddings: The embeddings method

    Returns:
        List[int]:
            query_idx: The index of queried samples in sents.
    """
    label_names = kwargs["label_names"]
    if "O" in label_names:
        label_names.remove("O")
    embeddings = kwargs["embeddings"]

    # Get entities in each class, each entity has {sent_idx, token_idx, token_text, token_embedding}
    label_entity_list = {label: [] for label in label_names}
    for sent_id, sent in enumerate(sents):
        if len(sent.get_spans(tag_type)) == 0:
            continue
        embeddings.embed(sent)
        for token_id, token in enumerate(sent):
            tag = token.get_tag("ner")
            if tag.value == "O":
                continue
            entity = Entity(token_id, sent_id, token.text, token.embedding)
            label_entity_list[tag.value].append(entity)

    # Assign similarity score to entity pair
    label_entity_pair_similarity = {label: [] for label in label_names}
    for label, entity_list in label_entity_list.items():
        class_entity_embedding_matrix = [tag["token_embedding"] for tag in entity_list]
        if class_entity_embedding_matrix == []:
            continue
        # Calculate similarity of entity pair
        class_entity_embedding_matrix = torch.stack(class_entity_embedding_matrix)
        class_entity_embedding_sim_matrix = sim_matrix(
            class_entity_embedding_matrix, class_entity_embedding_matrix
        )
        length = len(entity_list)
        for i in range(length - 1):
            for j in range(i + 1, length):
                cosine_score = class_entity_embedding_sim_matrix[i][j]
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
            sentence_score[entity1["sent_idx"]] -= cosine_score

            entity2 = entity_pair[1]
            sentence_score[entity2["sent_idx"]] -= cosine_score

    ascend_indices = np.argsort(sentence_score)

    return ascend_indices.tolist()


def cluster_sampling(
    sents: List[Sentence], tag_type: str = "ner", **kwargs
) -> List[int]:
    """Cluster sampling.

    We create cluster sampling as a kind of diversity sampling method.
    Different with most of sampling methods that are based on sentence level,
    Cluster sampling method is implemented on entity level.
    Cluster sampling classify all entity into cluster, and find the centen in each cluster.
    We calculate the similarity between center and entity in the same cluster,
    the low similarity pair means high diversity.

    Args:
        sents (List[Sentence]): Flair sentences
        tag_type (str): Label type, e.g. "ner"

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
    label_entity_list = get_entity_list(sents, embeddings, tag_type=tag_type)

    # Get all entity embedding matrix
    entity_embedding_matrix = [tag["token_embedding"] for tag in label_entity_list]
    if entity_embedding_matrix == []:
        return random_sampling(sents)
    entity_embedding_matrix = torch.stack(
        entity_embedding_matrix
    )  # e.g. shape=(36, 100)

    if embedding_dim is None:  # for nn.consine similarity
        embedding_dim = len(entity_embedding_matrix[0].shape) - 1

    # Clustering
    kmeans = KMeans(n_clusters=len(label_names))
    kmeans.fit(entity_embedding_matrix)

    # If the algorithm stops before fully converging (see tol and max_iter), these will not be consistent with labels_.
    # Which means the cluster_centers_ is not the real entity
    cluster_centers_matrix = kmeans.cluster_centers_  # e.g. shape is (4, 100)
    entity_labels = (
        kmeans.labels_
    )  # e.g. [0, 2, 3, 1, ...], the number is the indices of index in cluster_centers_matrix

    # Find the entity in each cluster
    center_cluster_index = list(range(cluster_centers_matrix.shape[0]))
    label_entity_cluster = {
        cluster_num: {"cluster_member_idx": []} for cluster_num in center_cluster_index
    }
    for cluster_num in label_entity_cluster.keys():
        for i, entity_cluster_num in enumerate(entity_labels):
            if entity_cluster_num == cluster_num:
                label_entity_cluster[cluster_num]["cluster_member_idx"].append(i)

    # Calculate each the similarity between center and entity
    for cluster_num, cluster_dict in label_entity_cluster.items():
        center_matrix = cluster_centers_matrix[cluster_num]  # shape (100,)
        center_matrix = center_matrix.reshape(1, len(center_matrix))  # shape (1, 100)
        member_matrix = entity_embedding_matrix[
            cluster_dict["cluster_member_idx"]
        ]  # shape (8, 100)
        scores = (
            sim_matrix(center_matrix, member_matrix).reshape(-1).tolist()
        )  # shape(8)
        label_entity_cluster[cluster_num]["sim_scores"] = scores

    # Flat the entity score
    entity_scores = [0] * len(label_entity_list)
    for cluster_num, cluster_dict in label_entity_cluster.items():
        for i, member_idx in enumerate(cluster_dict["cluster_member_idx"]):
            entity_scores[member_idx] += cluster_dict["sim_scores"][i]

    # Reorder the sentence index
    sentence_scores = [0] * len(sents)
    for entity_idx, entity_info in enumerate(label_entity_list):
        sent_idx = entity_info["sent_idx"]
        # The entity_scores is smaller, means similarity is low, the diversity is high
        # After minus, the sentence_scores is bigger
        sentence_scores[sent_idx] -= entity_scores[entity_idx]

    ascend_indices = np.argsort(sentence_scores)

    return ascend_indices.tolist()
