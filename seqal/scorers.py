import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from flair.data import Sentence
from flair.embeddings import Embeddings
from sklearn.cluster import KMeans

from seqal.base_scorer import BaseScorer
from seqal.data import Entities, Entity
from seqal.tagger import SequenceTagger


class LeastConfidenceScorer(BaseScorer):
    """Least confidence scorer

    https://dl.acm.org/doi/10.5555/1619410.1619452

    Args:
        BaseScorer: BaseScorer class.
    """

    def __call__(
        self,
        sentences: List[Sentence],
        tag_type: str,
        query_number: int,
        token_based: bool = False,
        **kwargs
    ) -> List[int]:
        """Least confidence sampling workflow

        Args:
            sentences (List[Sentence]): Sentences in data pool.
            tag_type (str): Tag type to predict.
            query_number (int): batch query number.
            token_based (bool, optional): If true, using query number as token number to query data.
                                          If false, using query number as sentence number to query data.

        kwargs:
            tagger: The tagger after training
            label_names (List[str]): Label name of all dataset
            embeddings: The embeddings method

        Returns:
            List[int]: Queried sentence ids.
        """
        tagger = kwargs["tagger"]
        self.predict(sentences, tagger)
        scores = self.score(sentences, tagger)
        sorted_sent_ids = self.sort(scores, order="descend")
        queried_sent_ids = self.query(
            sentences, sorted_sent_ids, query_number, token_based
        )
        return queried_sent_ids

    def score(self, sentences: List[Sentence], tagger: SequenceTagger) -> np.ndarray:
        """Calculate score for each sentence"""
        log_probs = tagger.log_probability(sentences)
        scores = 1 - np.exp(log_probs)
        return scores


class MaxNormLogProbScorer(BaseScorer):
    """Maximum Normalized Log-Probability scorer

    https://arxiv.org/abs/1707.05928

    Args:
        BaseScorer: BaseScorer class.
    """

    def __call__(
        self,
        sentences: List[Sentence],
        tag_type: str,
        query_number: int,
        token_based: bool = False,
        **kwargs
    ) -> List[int]:
        """Maximum Normalized Log-Probability sampling workflow

        Args:
            sentences (List[Sentence]): Sentences in data pool.
            tag_type (str): Tag type to predict.
            query_number (int): batch query number.
            token_based (bool, optional): If true, using query number as token number to query data.
                                          If false, using query number as sentence number to query data.

        kwargs:
            tagger: The tagger after training
            label_names (List[str]): Label name of all dataset
            embeddings: The embeddings method

        Returns:
            List[int]: Queried sentence ids.
        """
        tagger = kwargs["tagger"]
        self.predict(sentences, tagger)
        scores = self.score(sentences, tagger)
        sorted_sent_ids = self.sort(scores, order="ascend")
        queried_sent_ids = self.query(
            sentences, sorted_sent_ids, query_number, token_based
        )
        return queried_sent_ids

    def score(self, sentences: List[Sentence], tagger: SequenceTagger) -> np.ndarray:
        """Calculate score for each sentence"""
        log_probs = tagger.log_probability(sentences)
        lengths = np.array([len(sent) for sent in sentences])
        normed_log_probs = log_probs / lengths
        return normed_log_probs


class DistributeSimilarityScorer(BaseScorer):
    """Distribute similarity scorer

    We create distribute similarity sampling as a kind of diversity sampling method.
    Different with most of sampling methods that are based on sentence level,
    Distribute similarity sampling method is implemented on token level.
    We calculate the similarity between entity pair, the low similarity pair means high diversity.

    Args:
        BaseScorer: BaseScorer class.
    """

    def __call__(
        self,
        sentences: List[Sentence],
        tag_type: str,
        query_number: int,
        token_based: bool = False,
        **kwargs
    ) -> List[int]:
        """Distribute similarity sampling workflow

        Args:
            sentences (List[Sentence]): Sentences in data pool.
            tag_type (str): Tag type to predict.
            query_number (int): batch query number.
            token_based (bool, optional): If true, using query number as token number to query data.
                                          If false, using query number as sentence number to query data.

        kwargs:
            tagger: The tagger after training
            label_names (List[str]): Label name of all dataset
            embeddings: The embeddings method

        Returns:
            List[int]: Queried sentence ids.
        """
        tagger = kwargs["tagger"]
        embeddings = kwargs["embeddings"]
        self.predict(sentences, tagger)
        entities = self.get_entities(sentences, embeddings, tag_type)

        # If no entities, return random indices
        if not entities.entities:
            sent_ids = list(range(len(sentences)))
            random.seed(0)
            random_sent_ids = random.sample(sent_ids, len(sent_ids))
            queried_sent_ids = self.query(
                sentences, random_sent_ids, query_number, token_based
            )
            return queried_sent_ids

        scores = self.score(entities, tagger)
        sorted_sent_ids = self.sort(scores, order="ascend")
        queried_sent_ids = self.query(
            sentences, sorted_sent_ids, query_number, token_based
        )
        return queried_sent_ids

    def score(self, sentences: List[Sentence], entities: Entities) -> np.ndarray:
        """Calculate score for each sentence"""
        sentence_scores = [0] * len(sentences)
        diversities_per_sent = self.sentence_diversities(entities)
        for sent_id, score in diversities_per_sent.items():
            sentence_scores[sent_id] = score

        return np.array(sentence_scores)

    def get_entities(
        self, sentences: List[Sentence], embeddings: Embeddings, tag_type: str
    ) -> Entities:
        """Get entity list of each class"""
        entities = Entities()
        for sent_id, sent in enumerate(sentences):
            labeled_entities = sent.get_spans(tag_type)
            if labeled_entities == []:  # Skip non-entity sentence
                continue
            _ = embeddings.embed(sent)  # Add embeddings internal
            for entity_id, span in enumerate(labeled_entities):
                entity = Entity(entity_id, sent_id, span)
                entities.add(entity)

        if not entities.entities:
            token = sentences[0][0]
            label = token.get_tag(tag_type)
            if label.value == "" and label.score == 1:
                raise TypeError(
                    "Entities are empty. Sentences have not been predicted."
                )
        return entities

    def sentence_diversities(self, entities: Entities) -> Dict[int, float]:
        """Get diversity score of each sentence"""
        entities_per_sentence = entities.group_by_sentence
        entities_per_label = entities.group_by_label
        return {
            sent_id: self.calculate_diversity(entities, entities_per_label)
            for sent_id, entities in entities_per_sentence.items()
        }

    def calculate_diversity(
        self,
        sentence_entities: List[Entity],
        entities_per_label: Dict[str, List[Entity]],
    ) -> float:
        """Calculate diversity score for a sentence"""
        scores = []
        for entity in sentence_entities:
            vectors = torch.stack(
                [entity.vector for entity in entities_per_label[entity.label]]
            )
            similarities = self.similarity_matrix(torch.stack([entity.vector]), vectors)
            score = torch.min(similarities)
            scores.append(float(score))
        return sum(scores) / len(sentence_entities)


class ClusterSimilarityScorer(BaseScorer):
    """Distribute similarity scorer

    We create cluster sampling as a kind of diversity sampling method.
    Different with most of sampling methods that are based on sentence level,
    Cluster sampling method is implemented on entity level.
    Cluster sampling classify all entity into cluster, and find the centen in each cluster.
    We calculate the similarity between center and entity in the same cluster,
    the low similarity pair means high diversity.

    Args:
        BaseScorer: BaseScorer class.
    """

    def __call__(
        self,
        sentences: List[Sentence],
        tag_type: str,
        query_number: int,
        token_based: bool = False,
        **kwargs
    ) -> List[int]:
        """Distribute similarity sampling workflow

        Args:
            sentences (List[Sentence]): Sentences in data pool.
            tag_type (str): Tag type to predict.
            query_number (int): batch query number.
            token_based (bool, optional): If true, using query number as token number to query data.
                                          If false, using query number as sentence number to query data.

        kwargs:
            tagger: The tagger after training
            embeddings: The embeddings method
            kmeans_params (dict): Parameters for clustering, detail on sklearn.cluster.KMeans.
                                  e.g. {"n_clusters": 8, "n_init": 10, "random_state": 0}
                                  "n_clusters": The number of cluster (label types except "O")
                                  "n_init": Number of time the k-means algorithm
                                            will be run with different centroid seeds.
                                  "random_state": Determines random number generation for centroid initialization.

        Returns:
            List[int]: Queried sentence ids.
        """
        tagger = kwargs["tagger"]
        embeddings = kwargs["embeddings"]
        kmeans_params = kwargs["kmeans_params"]
        self.predict(sentences, tagger)
        entities = self.get_entities(sentences, embeddings, tag_type)

        # If no entities, return random indices
        if not entities.entities:
            sent_ids = list(range(len(sentences)))
            random.seed(0)
            random_sent_ids = random.sample(sent_ids, len(sent_ids))
            queried_sent_ids = self.query(
                sentences, random_sent_ids, query_number, token_based
            )
            return queried_sent_ids

        scores = self.score(entities, tagger, kmeans_params)
        sorted_sent_ids = self.sort(scores, order="ascend")
        queried_sent_ids = self.query(
            sentences, sorted_sent_ids, query_number, token_based
        )
        return queried_sent_ids

    def score(
        self, sentences: List[Sentence], entities: Entities, kmeans_params: dict
    ) -> np.ndarray:
        """Calculate score for each sentence"""
        sentence_scores = [0] * len(sentences)
        cluster_centers_matrix, entity_cluster_nums = self.kmeans(
            entities.entities, kmeans_params
        )
        entities = self.assign_cluster(entities, entity_cluster_nums)
        diversities_per_sent = self.sentence_diversities(
            entities, cluster_centers_matrix
        )
        for sent_id, score in diversities_per_sent.items():
            sentence_scores[sent_id] = score

        return np.array(sentence_scores)

    def sentence_diversities(
        self, entities: Entities, cluster_centers_matrix: np.ndarray
    ) -> Dict[int, float]:
        """Get diversity score of each sentence"""
        entities_per_cluster = entities.group_by_cluster
        entities_per_sentence = entities.group_by_sentence
        return {
            sent_id: self.calculate_diversity(
                sent_entities, entities_per_cluster, cluster_centers_matrix
            )
            for sent_id, sent_entities in entities_per_sentence.items()
        }

    def assign_cluster(
        self, entities: Entities, entity_cluster_nums: np.ndarray
    ) -> Entities:
        """Assign cluster number to Entity"""
        new_entities = Entities()
        for i, entity in enumerate(entities.entities):
            entity.cluster = entity_cluster_nums[i]
            new_entities.add(entity)
        return new_entities

    def calculate_diversity(
        self,
        sentence_entities: List[Entity],
        entities_per_cluster: Dict[int, List[Entity]],
        cluster_centers_matrix: np.ndarray,
    ) -> float:
        """Calculate diversity score for a sentence"""
        scores = []
        cluster_centers_matrix = torch.tensor(cluster_centers_matrix)
        for entity in sentence_entities:
            cluster_center_vector = cluster_centers_matrix[entity.cluster]
            vectors = torch.stack(
                [entity.vector for entity in entities_per_cluster[entity.cluster]]
            )
            similarities = self.similarity_matrix(
                torch.stack([cluster_center_vector]), vectors
            )
            score = torch.min(similarities)
            scores.append(float(score))
        return sum(scores) / len(sentence_entities)

    def kmeans(
        self, entities: List[Entity], kmeans_params: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """K-Means cluster to get cluster centers and entity cluster"""
        if "n_clusters" not in kmeans_params:
            raise KeyError("n_clusters is not found.")
        if "random_state" not in kmeans_params:
            kmeans_params["random_state"] = 0

        kmeans = KMeans(**kmeans_params)
        entity_embedding_matrix = [entity.vector for entity in entities]
        entity_embedding_matrix = torch.stack(
            entity_embedding_matrix
        )  # e.g. shape is (36, 100)
        kmeans.fit(entity_embedding_matrix)
        cluster_centers_matrix = kmeans.cluster_centers_  # e.g. shape is (4, 100)
        entity_cluster_nums = (
            kmeans.labels_
        )  # e.g. [0, 2, 3, 1, ...], the number is the indices of index in cluster_centers_matrix

        return cluster_centers_matrix, entity_cluster_nums

    def get_entities(
        self, sentences: List[Sentence], embeddings: Embeddings, tag_type: str
    ) -> Entities:
        """Get entity list of each class"""
        entities = Entities()
        for sent_id, sent in enumerate(sentences):
            labeled_entities = sent.get_spans(tag_type)
            if labeled_entities == []:  # Skip non-entity sentence
                continue
            _ = embeddings.embed(sent)  # Add embeddings internal
            for entity_id, span in enumerate(labeled_entities):
                entity = Entity(entity_id, sent_id, span)
                entities.add(entity)

        if not entities.entities:
            token = sentences[0][0]
            label = token.get_tag(tag_type)
            if label.value == "" and label.score == 1:
                raise TypeError(
                    "Entities are empty. Sentences have not been predicted."
                )
        return entities
