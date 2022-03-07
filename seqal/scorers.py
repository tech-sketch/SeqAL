import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from flair.data import Sentence
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from seqal.base_scorer import BaseScorer
from seqal.data import Entities, Entity
from seqal.tagger import SequenceTagger


class RandomScorer(BaseScorer):
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
            random_scorer = RandomScorer()
            return random_scorer(sentences, tag_type, query_number, token_based)

        scores = self.score(sentences, entities)
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
        kmeans_params = kwargs["kmeans_params"]
        self.predict(sentences, tagger)
        entities = self.get_entities(sentences, embeddings, tag_type)

        # If no entities, return random indices
        if not entities.entities:
            random_scorer = RandomScorer()
            return random_scorer(sentences, tag_type, query_number, token_based)

        scores = self.score(sentences, entities, kmeans_params)
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


class CombinedMultipleScorer(BaseScorer):
    """Multiple similarity scorer

    Uncertainty-based scorers do not take full advantage of entity information.
    The proposed token-level diversity based scorer can fully utilize the entity information.
    So we combine diversity scorer and uncertainty-based scorer together to improve the active learning performance.

    Args:
        BaseScorer: BaseScorer class.
    """

    @property
    def available_scorer_types(self):
        available_scorer_types = ["lc_ds", "lc_cs", "mnlp_ds", "mnlp_cs"]
        return available_scorer_types

    @property
    def available_combined_types(self):
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
        """Combined multiple scorer sampling workflow

        Args:
            sentences (List[Sentence]): Sentences in data pool.
            tag_type (str): Tag type to predict.
            query_number (int): batch query number.
            token_based (bool, optional): If true, using query number as token number to query data.
                                          If false, using query number as sentence number to query data.

        kwargs:
            scorer_type (str):  Which kind of scorer to use.
                                Available types are "lc_ds", "lc_cs", "mnlp_ds", "mnlp_cs"
                                - "lc_ds" means LeastConfidenceScorer and DistributeSimilarityScorer.
                                - "lc_cs" means LeastConfidenceScorer and ClusterSimilarityScorer.
                                - "mnlp_ds" means MaxNormLogProbScorer and DistributeSimilarityScorer.
                                - "mnlp_cs" means MaxNormLogProbScorer and ClusterSimilarityScorer.

            combined_type (str): The combined method of different scorers.
                                 Available types are "series", "parallel"
                                 - "series" means run one scorer first and then run the second one.
                                 - "parallel" means run two scorers together.
                                 If scorer_type is "lc_ds", it means first run lc and then run ds.
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
        self.check_scorer_type(kwargs)
        self.check_combined_type(kwargs)
        scorer_type = kwargs["scorer_type"]
        combined_type = kwargs["combined_type"]

        # Get scorers
        uncertainty_scorer, diversity_scorer = self.get_scorers(scorer_type)

        # Combine scores
        if combined_type == "series":
            uncertainty_scorer_queried_sent_ids = uncertainty_scorer(
                sentences, tag_type, 2 * query_number, token_based, **kwargs
            )
            uncertainty_scorer_queried_sents = [
                sentences[i] for i in uncertainty_scorer_queried_sent_ids
            ]
            queried_sent_ids = diversity_scorer(
                uncertainty_scorer_queried_sents,
                tag_type,
                query_number,
                token_based,
                **kwargs,
            )
            return queried_sent_ids

        # The combine_type == "parallel"
        tagger = kwargs["tagger"]
        embeddings = kwargs["embeddings"]
        if "kmeans_params" in kwargs:
            kmeans_params = kwargs["kmeans_params"]

        self.predict(sentences, tagger)
        entities = self.get_entities(sentences, embeddings, tag_type)

        # If no entities, return random indices
        if not entities.entities:
            random_scorer = RandomScorer()
            return random_scorer(sentences, tag_type, query_number, token_based)

        # Calculate scores
        uncertainty_scores = uncertainty_scorer.score(sentences, tagger)
        if "kmeans_params" in kwargs:
            diversity_scores = diversity_scorer.score(
                sentences, entities, kmeans_params
            )
        diversity_scores = diversity_scorer.score(sentences, entities)

        # Normalize scores
        if "lc" in scorer_type:  # reverse lc order for ascend setup below
            scores = self.normalize_scores(-uncertainty_scores, diversity_scores)
        scores = self.normalize_scores(uncertainty_scores, diversity_scores)

        sorted_sent_ids = self.sort(scores, order="ascend")
        queried_sent_ids = self.query(
            sentences, sorted_sent_ids, query_number, token_based
        )
        return queried_sent_ids

    def normalize_scores(
        self, uncertainty_scores: np.ndarray, diversity_scores: np.ndarray
    ) -> np.ndarray:
        """Normalize two kinds of scores

        Args:
            uncertainty_scores (np.ndarray): Scores calculated by uncertainty_scorer
            diversity_scores (np.ndarray): Scores calculated by diversity_scorer

        Returns:
            np.ndarray: Normalized score
        """
        scaler = MinMaxScaler()
        concatenate_scores = np.stack([uncertainty_scores, diversity_scores])
        normalized_scores = scaler.fit_transform(np.transpose(concatenate_scores))
        return normalized_scores.sum(axis=1)

    def get_scorers(self, scorer_type: str) -> Tuple[BaseScorer, BaseScorer]:
        """Get specific scorers"""
        if scorer_type == "lc_ds":
            uncertainty_scorer, diversity_scorer = (
                LeastConfidenceScorer(),
                DistributeSimilarityScorer(),
            )
        elif scorer_type == "lc_cs":
            uncertainty_scorer, diversity_scorer = (
                LeastConfidenceScorer(),
                ClusterSimilarityScorer(),
            )
        elif scorer_type == "mnlp_ds":
            uncertainty_scorer, diversity_scorer = (
                MaxNormLogProbScorer(),
                DistributeSimilarityScorer(),
            )
        elif scorer_type == "mnlp_cs":
            uncertainty_scorer, diversity_scorer = (
                MaxNormLogProbScorer(),
                ClusterSimilarityScorer(),
            )

        return uncertainty_scorer, diversity_scorer

    def check_scorer_type(self, kwargs: dict) -> bool:
        """Check the scorer type is availabel or not."""
        if "scorer_type" not in kwargs:
            raise KeyError("scorer_type is not found.")

        scorer_type = kwargs["scorer_type"]
        if scorer_type not in self.available_scorer_types:
            raise NameError(
                f"scorer_type is not found. scorer_type must be one of {self.available_scorer_types}"
            )
        return True

    def check_combined_type(self, kwargs: dict) -> bool:
        """Check the combined type is availabel or not."""
        if "combined_type" not in kwargs:
            raise KeyError("combined_type is not found.")

        combined_type = kwargs["combined_type"]
        if combined_type not in self.available_combined_types:
            raise NameError(
                f"combined_type is not found. scorer_type must be one of {self.available_combined_types}"
            )
        return True
