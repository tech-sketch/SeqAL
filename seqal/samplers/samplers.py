import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from flair.data import Sentence
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from seqal.data import Entities, Entity
from seqal.tagger import SequenceTagger

from .base import BaseSampler


class RandomSampler(BaseSampler):
    """Random sampling method"""

    def __call__(
        self,
        sentences: List[Sentence],
        tag_type: str,
        query_number: int,
        token_based: bool = False,
        **kwargs,
    ) -> List[int]:
        """Random sampling workflow

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
        random.seed(0)
        sent_ids = list(range(len(sentences)))
        random_sent_ids = random.sample(sent_ids, len(sent_ids))
        queried_sent_ids = self.query(
            sentences, random_sent_ids, query_number, token_based
        )
        return queried_sent_ids


class LeastConfidenceSampler(BaseSampler):
    """Least confidence sampler

    https://dl.acm.org/doi/10.5555/1619410.1619452

    Args:
        BaseSampler: BaseSampler class.
    """

    def __call__(
        self,
        sentences: List[Sentence],
        tag_type: str,
        query_number: int,
        token_based: bool = False,
        **kwargs,
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
        sorted_sent_ids = self.sort(-scores, order="ascend")
        queried_sent_ids = self.query(
            sentences, sorted_sent_ids, query_number, token_based
        )
        return queried_sent_ids

    def score(
        self,
        sentences: List[Sentence],
        tagger: SequenceTagger,
        kwargs: Optional[dict] = None,
    ) -> np.ndarray:
        """Calculate score for each sentence"""
        log_probs = tagger.log_probability(sentences)
        scores = 1 - np.exp(log_probs)
        return scores


class MaxNormLogProbSampler(BaseSampler):
    """Maximum Normalized Log-Probability sampler

    https://arxiv.org/abs/1707.05928

    Args:
        BaseSampler: BaseSampler class.
    """

    def __call__(
        self,
        sentences: List[Sentence],
        tag_type: str,
        query_number: int,
        token_based: bool = False,
        **kwargs,
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

    def score(
        self,
        sentences: List[Sentence],
        tagger: SequenceTagger,
        kwargs: Optional[dict] = None,
    ) -> np.ndarray:
        """Calculate score for each sentence"""
        log_probs = tagger.log_probability(sentences)
        lengths = np.array([len(sent) for sent in sentences])
        normed_log_probs = log_probs / lengths
        return normed_log_probs


class StringNGramSampler(BaseSampler):
    """The StringNGramSampler class

    https://aclanthology.org/C10-1096.pdf

    Args:
        BaseSampler: BaseSampler class.
    """

    def __call__(
        self,
        sentences: List[Sentence],
        tag_type: str,
        query_number: int,
        token_based: bool = False,
        **kwargs,
    ) -> List[int]:
        """StringNGram similarity sampling workflow

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
            random_sampler = RandomSampler()
            return random_sampler(sentences, tag_type, query_number, token_based)

        scores = self.score(sentences, entities)
        sorted_sent_ids = self.sort(scores, order="ascend")
        queried_sent_ids = self.query(
            sentences, sorted_sent_ids, query_number, token_based
        )
        return queried_sent_ids

    def score(
        self,
        sentences: List[Sentence],
        entities: Entities,
        kwargs: Optional[dict] = None,
    ) -> np.ndarray:
        """Calculate score for each sentence"""
        sentence_scores = [0] * len(sentences)
        diversities_per_sent = self.sentence_diversities(entities)
        for sent_id, score in diversities_per_sent.items():
            sentence_scores[sent_id] = score

        return np.array(sentence_scores)

    def trigram(self, entity: Entity) -> List[str]:
        """Get trigram of a entity

        Args:
            entity (Entity): Entity contains text

        Returns:
            List[str]: Entity trigram with ordinal number
                       e.g. "Peter" will return ['$$P1', '$Pe1', 'Pet1', 'ete1', 'ter1', 'er$1', 'r$$1']
                       e.g. "prepress" will return ['$$p1', '$pr1', 'pre1', 'rep1', 'epr1',
                                                    'pre2', 'res1', 'ess1', 'ss$1', 's$$1']
        """
        counter = defaultdict(int)
        entity = "$$" + entity.text + "$$"
        entity = entity.replace(" ", "_")
        trigrams = []
        for i in range(len(entity) - 3 + 1):
            span = entity[i : i + 3]  # noqa: E203
            counter[span] += 1
            trigrams.append(span + str(counter[span]))
        return trigrams

    def sentence_diversities(self, entities: Entities) -> Dict[int, float]:
        """Get diversity score of each sentence"""
        entities_per_label = entities.group_by_label
        entities_per_sentence = entities.group_by_sentence

        # Calculate similarities of all entities in one label
        similarities_per_label = self.similarity_matrix_per_label(
            entities_per_label
        )  # {"ORG": matrix, "PER": matrix}

        # Create index map
        # entity_id_map[label][sent_id][entity_id] = entity_id_in_label_entity_list
        sentence_count = max(entities_per_sentence.keys()) + 1
        max_entity_count = max(
            [len(entities) for entities in entities_per_sentence.values()]
        )
        entity_id_map = self.get_entity_id_map(
            entities_per_label, sentence_count, max_entity_count
        )

        # Calculate diversity for each sentence
        sentence_scores = self.calculate_diversity(
            entities_per_sentence, entity_id_map, similarities_per_label
        )

        return sentence_scores

    def calculate_diversity(
        self,
        entities_per_sentence: Dict[str, List[Entity]],
        entity_id_map: dict,
        similarities_per_label: dict,
    ) -> float:
        """Calculate diversity score for a sentence"""
        sentence_scores = {}
        for sent_id, sentence_entities in entities_per_sentence.items():
            scores = []
            for entity in sentence_entities:
                entity_id_in_label_list = entity_id_map[entity.label][entity.sent_id][
                    entity.id
                ]
                similarities = similarities_per_label[entity.label][
                    int(entity_id_in_label_list)
                ]
                scores.append(float(similarities.min()))
            sentence_score = sum(scores) / len(sentence_entities)
            sentence_scores[sent_id] = sentence_score
        return sentence_scores

    def get_entity_id_map(
        self,
        entities_per_label: Dict[str, List[Entity]],
        sentence_count: int,
        max_entity_count: int,
    ) -> Dict[str, np.ndarray]:
        """Get index map of entity from sentence id to the id in entity_per_label"""
        entity_id_map = {}
        for label, label_entities in entities_per_label.items():
            if label not in entity_id_map:
                entity_id_map[label] = np.ones((sentence_count, max_entity_count))
            for i, entity in enumerate(
                label_entities
            ):  # entity id in label entities list
                print(i, entity, entity.sent_id, entity.id)
                entity_id_map[label][entity.sent_id][entity.id] = i
            print(entity_id_map)
        return entity_id_map

    def similarity_matrix_per_label(
        self, entities_per_label: Dict[str, List[Entity]]
    ) -> Dict[str, np.ndarray]:
        """Calculate similarity matrix of entities in each label"""
        similarity_matrix_per_label = {}
        for label, label_entities in entities_per_label.items():
            entities_trigrams = [self.trigram(e) for e in label_entities]
            similarity_matrix = []
            for i, entity in enumerate(label_entities):
                entity_trigrams = self.trigram(entity)
                similarities = [
                    self.trigram_cosine_similarity(
                        entity_trigrams, entities_trigrams[i]
                    )
                    for i in range(len(label_entities))
                ]
                similarity_matrix.append(similarities)
            similarity_matrix_per_label[label] = np.array(similarity_matrix)
        return similarity_matrix_per_label

    def trigram_cosine_similarity(
        self, entity_trigram1: List[str], entity_trigram2: List[str]
    ) -> float:
        """Calculate trigram consine similarity"""
        similarity = len(set(entity_trigram1) & set(entity_trigram2)) / math.sqrt(
            len(entity_trigram1) * len(entity_trigram2)
        )
        return similarity


class DistributeSimilaritySampler(BaseSampler):
    """Distribute similarity sampler

    We create distribute similarity sampling as a kind of diversity sampling method.
    Different with most of sampling methods that are based on sentence level,
    Distribute similarity sampling method is implemented on token level.
    We calculate the similarity between entity pair, the low similarity pair means high diversity.

    Args:
        BaseSampler: BaseSampler class.
    """

    def __call__(
        self,
        sentences: List[Sentence],
        tag_type: str,
        query_number: int,
        token_based: bool = False,
        **kwargs,
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
            random_sampler = RandomSampler()
            return random_sampler(sentences, tag_type, query_number, token_based)

        scores = self.score(sentences, entities)
        sorted_sent_ids = self.sort(scores, order="ascend")
        queried_sent_ids = self.query(
            sentences, sorted_sent_ids, query_number, token_based
        )
        return queried_sent_ids

    def score(
        self,
        sentences: List[Sentence],
        entities: Entities,
        kwargs: Optional[dict] = None,
    ) -> np.ndarray:
        """Calculate score for each sentence"""
        sentence_scores = [0] * len(sentences)
        diversities_per_sent = self.sentence_diversities(entities)
        for sent_id, score in diversities_per_sent.items():
            sentence_scores[sent_id] = score

        return np.array(sentence_scores)

    def sentence_diversities(self, entities: Entities) -> Dict[int, float]:
        """Get diversity score of each sentence"""
        entities_per_label = entities.group_by_label
        entities_per_sentence = entities.group_by_sentence

        # Calculate similarities of all entities in one label
        similarities_per_label = self.similarity_matrix_per_label(
            entities_per_label
        )  # {"ORG": matrix, "PER": matrix}

        # Create index map
        # entity_id_map[label][sent_id][entity_id] = entity_id_in_label_entity_list
        sentence_count = max(entities_per_sentence.keys()) + 1
        max_entity_count = max(
            [len(entities) for entities in entities_per_sentence.values()]
        )
        entity_id_map = self.get_entity_id_map(
            entities_per_label, sentence_count, max_entity_count
        )

        # Calculate diversity for each sentence
        sentence_scores = self.calculate_diversity(
            entities_per_sentence, entity_id_map, similarities_per_label
        )

        return sentence_scores

    def calculate_diversity(
        self,
        entities_per_sentence: Dict[str, List[Entity]],
        entity_id_map: dict,
        similarities_per_label: dict,
    ) -> float:
        """Calculate diversity score for a sentence"""
        sentence_scores = {}
        for sent_id, sentence_entities in entities_per_sentence.items():
            scores = []
            for entity in sentence_entities:
                entity_id_in_label_list = entity_id_map[entity.label][entity.sent_id][
                    entity.id
                ]
                similarities = similarities_per_label[entity.label][
                    int(entity_id_in_label_list)
                ]
                scores.append(float(similarities.min()))
            sentence_score = sum(scores) / len(sentence_entities)
            sentence_scores[sent_id] = sentence_score
        return sentence_scores

    def get_entity_id_map(
        self,
        entities_per_label: Dict[str, List[Entity]],
        sentence_count: int,
        max_entity_count: int,
    ) -> Dict[str, np.ndarray]:
        """Get index map of entity from sentence id to the id in entity_per_label

        Args:
            entities_per_label (Dict[str, List[Entity]]): Entity list in each label.
            sentence_count (int): Sentences count number, used for create matrix.
            max_entity_count (int): Max entities count in every sentence, used for create matrix.

        Returns:
            Dict[str, np.ndarray]: An index map convert entity id from sentence id to the id label entities list
                                   e.g. map[label][sent_id][entity_id] = entity_id_in_label_entities_list
        """
        entity_id_map = {}
        for label, label_entities in entities_per_label.items():
            if label not in entity_id_map:
                entity_id_map[label] = np.ones((sentence_count, max_entity_count))
            for i, entity in enumerate(
                label_entities
            ):  # entity id in label entities list
                print(i, entity, entity.sent_id, entity.id)
                entity_id_map[label][entity.sent_id][entity.id] = i
            print(entity_id_map)
        return entity_id_map

    def similarity_matrix_per_label(
        self, entities_per_label: Dict[str, List[Entity]]
    ) -> Dict[str, np.ndarray]:
        """Calculate similarity matrix of entities in each label"""
        similarity_matrix_per_label = {}
        for label, label_entities in entities_per_label.items():
            vectors = torch.stack([entity.vector for entity in label_entities])
            similarities = self.similarity_matrix(vectors, vectors)
            similarity_matrix_per_label[label] = (
                similarities.cpu().detach().numpy().copy()
            )
        return similarity_matrix_per_label


class ClusterSimilaritySampler(BaseSampler):
    """Distribute similarity sampler

    We create cluster sampling as a kind of diversity sampling method.
    Different with most of sampling methods that are based on sentence level,
    Cluster sampling method is implemented on entity level.
    Cluster sampling classify all entity into cluster, and find the centen in each cluster.
    We calculate the similarity between center and entity in the same cluster,
    the low similarity pair means high diversity.

    Args:
        BaseSampler: BaseSampler class.
    """

    def __call__(
        self,
        sentences: List[Sentence],
        tag_type: str,
        query_number: int,
        token_based: bool = False,
        **kwargs,
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
        self.predict(sentences, tagger)
        entities = self.get_entities(sentences, embeddings, tag_type)

        # If no entities, return random indices
        if not entities.entities:
            random_sampler = RandomSampler()
            return random_sampler(sentences, tag_type, query_number, token_based)

        scores = self.score(sentences, entities, kwargs)
        sorted_sent_ids = self.sort(scores, order="ascend")
        queried_sent_ids = self.query(
            sentences, sorted_sent_ids, query_number, token_based
        )
        return queried_sent_ids

    def score(
        self,
        sentences: List[Sentence],
        entities: Entities,
        kwargs: Optional[dict] = None,
    ) -> np.ndarray:
        """Calculate score for each sentence"""
        kmeans_params = self.get_kmeans_params(kwargs)

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

    def get_kmeans_params(self, kwargs: dict) -> bool:
        """Check the sampler type is availabel or not."""
        if "kmeans_params" not in kwargs or "n_clusters" not in kwargs["kmeans_params"]:
            output = (
                "You have to provide 'kmeans_params' parameter to use ClusterSimilaritySampler."
                " 'kmeans_params' must contain 'n_clusters', which means number of label types in dataset except 'O'."
                " For example, kmeans_params={'n_clusters': 8, 'n_init': 10, 'random_state': 0}}"
            )
            raise NameError(output)

        kmeans_params = kwargs["kmeans_params"]
        return kmeans_params

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


class CombinedMultipleSampler(BaseSampler):
    """Multiple similarity sampler

    Uncertainty-based samplers do not take full advantage of entity information.
    The proposed token-level diversity based sampler can fully utilize the entity information.
    So we combine diversity sampler and uncertainty-based sampler together to improve the active learning performance.

    Args:
        BaseSampler: BaseSampler class.
    """

    @property
    def available_sampler_types(self):
        """Available samplers"""
        available_sampler_types = ["lc_ds", "lc_cs", "mnlp_ds", "mnlp_cs"]
        return available_sampler_types

    @property
    def available_combined_types(self):
        """Available combined type"""
        available_combined_types = ["series", "parallel"]
        return available_combined_types

    def __call__(
        self,
        sentences: List[Sentence],
        tag_type: str,
        query_number: int,
        token_based: bool = False,
        **kwargs,
    ) -> List[int]:
        """Combined multiple sampler sampling workflow

        Args:
            sentences (List[Sentence]): Sentences in data pool.
            tag_type (str): Tag type to predict.
            query_number (int): batch query number.
            token_based (bool, optional): If true, using query number as token number to query data.
                                          If false, using query number as sentence number to query data.

        kwargs:
            sampler_type (str):  Which kind of sampler to use.
                                Available types are "lc_ds", "lc_cs", "mnlp_ds", "mnlp_cs"
                                - "lc_ds" means LeastConfidenceSampler and DistributeSimilaritySampler.
                                - "lc_cs" means LeastConfidenceSampler and ClusterSimilaritySampler.
                                - "mnlp_ds" means MaxNormLogProbSampler and DistributeSimilaritySampler.
                                - "mnlp_cs" means MaxNormLogProbSampler and ClusterSimilaritySampler.

            combined_type (str): The combined method of different samplers.
                                 Available types are "series", "parallel"
                                 - "series" means run one sampler first and then run the second one.
                                 - "parallel" means run two samplers together.
                                 If sampler_type is "lc_ds", it means first run lc and then run ds.
                                 If reverse parameter is provided, it runs ds first and then lc.
            reverse (bool): The running order when combined type is "series"
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
        sampler_type = self.get_sampler_type(kwargs)
        combined_type = self.get_combined_type(kwargs)
        scaler = self.get_scaler(kwargs)

        # Get samplers
        uncertainty_sampler, diversity_sampler = self.get_samplers(sampler_type)

        # Combine scores
        if combined_type == "series":
            uncertainty_sampler_queried_sent_ids = uncertainty_sampler(
                sentences, tag_type, 2 * query_number, token_based, **kwargs
            )
            uncertainty_sampler_queried_sents = [
                sentences[i] for i in uncertainty_sampler_queried_sent_ids
            ]
            queried_sent_ids = diversity_sampler(
                uncertainty_sampler_queried_sents,
                tag_type,
                query_number,
                token_based,
                **kwargs,
            )
            return queried_sent_ids

        # The combine_type == "parallel"
        tagger = kwargs["tagger"]
        embeddings = kwargs["embeddings"]

        self.predict(sentences, tagger)
        entities = self.get_entities(sentences, embeddings, tag_type)

        # If no entities, return random indices
        if not entities.entities:
            random_sampler = RandomSampler()
            return random_sampler(sentences, tag_type, query_number, token_based)

        # Calculate scores
        uncertainty_scores = uncertainty_sampler.score(sentences, tagger)
        diversity_scores = diversity_sampler.score(sentences, entities, kwargs)

        # Normalize scores
        if "lc" in sampler_type:  # reverse lc order for ascend setup below
            scores = self.normalize_scores(
                -uncertainty_scores, diversity_scores, scaler
            )
        scores = self.normalize_scores(uncertainty_scores, diversity_scores, scaler)

        sorted_sent_ids = self.sort(scores, order="ascend")
        queried_sent_ids = self.query(
            sentences, sorted_sent_ids, query_number, token_based
        )
        return queried_sent_ids

    def normalize_scores(
        self,
        uncertainty_scores: np.ndarray,
        diversity_scores: np.ndarray,
        scaler: BaseEstimator,
    ) -> np.ndarray:
        """Normalize two kinds of scores

        Args:
            uncertainty_scores (np.ndarray): Scores calculated by uncertainty_sampler
            diversity_scores (np.ndarray): Scores calculated by diversity_sampler

        Returns:
            np.ndarray: Normalized score
        """
        concatenate_scores = np.stack([uncertainty_scores, diversity_scores])
        normalized_scores = scaler.fit_transform(np.transpose(concatenate_scores))
        return normalized_scores.sum(axis=1)

    def get_samplers(self, sampler_type: str) -> Tuple[BaseSampler, BaseSampler]:
        """Get specific samplers"""
        if sampler_type == "lc_ds":
            uncertainty_sampler, diversity_sampler = (
                LeastConfidenceSampler(),
                DistributeSimilaritySampler(),
            )
        elif sampler_type == "lc_cs":
            uncertainty_sampler, diversity_sampler = (
                LeastConfidenceSampler(),
                ClusterSimilaritySampler(),
            )
        elif sampler_type == "mnlp_ds":
            uncertainty_sampler, diversity_sampler = (
                MaxNormLogProbSampler(),
                DistributeSimilaritySampler(),
            )
        elif sampler_type == "mnlp_cs":
            uncertainty_sampler, diversity_sampler = (
                MaxNormLogProbSampler(),
                ClusterSimilaritySampler(),
            )
        else:
            uncertainty_sampler, diversity_sampler = (
                LeastConfidenceSampler(),
                DistributeSimilaritySampler(),
            )

        return uncertainty_sampler, diversity_sampler

    def get_sampler_type(self, kwargs: dict) -> bool:
        """Check the sampler type is availabel or not."""
        if "sampler_type" not in kwargs:
            sampler_type = "lc_ds"
            print("sampler_type is not found. Default use 'lc_ds' sampler type")
            return sampler_type

        sampler_type = kwargs["sampler_type"]
        if sampler_type not in self.available_sampler_types:
            raise NameError(
                f"sampler_type is not found. sampler_type must be one of {self.available_sampler_types}"
            )
        return sampler_type

    def get_combined_type(self, kwargs: dict) -> bool:
        """Check the combined type is availabel or not."""
        if "combined_type" not in kwargs:
            combined_type = "parallel"
            print("combined_type is not found. Default use 'parallel' combined type")
            return combined_type

        combined_type = kwargs["combined_type"]
        if combined_type not in self.available_combined_types:
            raise NameError(
                f"combined_type is not found. combined_type must be one of {self.available_combined_types}"
            )
        return combined_type

    def get_scaler(self, kwargs: dict) -> bool:
        """Get scaler"""
        if "scaler" not in kwargs:
            scaler = MinMaxScaler()
            print("scaler is not found. Default use 'MinMaxScaler()'")
            return scaler

        return kwargs["scaler"]
