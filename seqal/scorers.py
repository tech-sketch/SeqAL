import random
from typing import Dict, List

import numpy as np
import torch
from flair.data import Sentence
from flair.embeddings import Embeddings

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
        sents: List[Sentence],
        tag_type: str,
        query_number: int,
        token_based: bool = False,
        **kwargs
    ) -> List[int]:
        """Least confidence sampling workflow

        Args:
            sents (List[Sentence]): Sentences in data pool.
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
        self.predict(sents, tagger)
        scores = self.score(sents, tagger)
        sorted_sent_ids = self.sort(scores, order="descend")
        queried_sent_ids = self.query(sents, sorted_sent_ids, query_number, token_based)
        return queried_sent_ids

    def score(self, sents: List[Sentence], tagger: SequenceTagger) -> np.ndarray:
        log_probs = tagger.log_probability(sents)
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
        sents: List[Sentence],
        tag_type: str,
        query_number: int,
        token_based: bool = False,
        **kwargs
    ) -> List[int]:
        """Maximum Normalized Log-Probability sampling workflow

        Args:
            sents (List[Sentence]): Sentences in data pool.
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
        self.predict(sents, tagger)
        scores = self.score(sents, tagger)
        sorted_sent_ids = self.sort(scores, order="ascend")
        queried_sent_ids = self.query(sents, sorted_sent_ids, query_number, token_based)
        return queried_sent_ids

    def score(self, sents: List[Sentence], tagger: SequenceTagger) -> np.ndarray:
        log_probs = tagger.log_probability(sents)
        lengths = np.array([len(sent) for sent in sents])
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
        sents: List[Sentence],
        tag_type: str,
        query_number: int,
        token_based: bool = False,
        **kwargs
    ) -> List[int]:
        """Distribute similarity sampling workflow

        Args:
            sents (List[Sentence]): Sentences in data pool.
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
        self.predict(sents, tagger)
        entities = self.get_entities(sents, embeddings, tag_type)

        # If no entities, return random indices
        if entities.entities == []:
            sent_ids = list(range(len(sents)))
            random.seed(0)
            random_sent_ids = random.sample(sent_ids, len(sent_ids))
            queried_sent_ids = self.query(
                sents, random_sent_ids, query_number, token_based
            )
            return queried_sent_ids

        scores = self.score(entities, tagger)
        sorted_sent_ids = self.sort(scores, order="ascend")
        queried_sent_ids = self.query(sents, sorted_sent_ids, query_number, token_based)
        return queried_sent_ids

    def score(self, sents: List[Sentence], entities: Entities) -> List[float]:
        """Calculate score for each sentence"""
        sentence_scores = [0] * len(sents)
        diversities_per_sent = self.sentence_diversities(entities)
        for sent_id, score in diversities_per_sent.items():
            sentence_scores[sent_id] = score

        return sentence_scores

    def get_entities(
        self, sents: List[Sentence], embeddings: Embeddings, tag_type: str
    ) -> Entities:
        """Get entity list of each class"""
        entities = Entities()
        for sent_id, sent in enumerate(sents):
            labeled_entities = sent.get_spans(tag_type)
            if labeled_entities == []:  # Skip non-entity sentence
                continue
            _ = embeddings.embed(sent)  # Add embeddings internal
            for entity_id, span in enumerate(labeled_entities):
                entity = Entity(entity_id, sent_id, span)
                entities.add(entity)

        if entities.entities == []:
            token = sents[0][0]
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
        self, entities: List[Entity], entities_per_label: Dict[str, List[Entity]]
    ) -> float:
        """Calculate diversity score for a sentence"""
        scores = []
        for entity in entities:
            vectors = torch.stack(
                [entity.vector for entity in entities_per_label[entity.label]]
            )
            similarities = self.similarity_matrix(torch.stack([entity.vector]), vectors)
            score = torch.min(similarities)
            scores.append(float(score))
        return sum(scores) / len(entities)
