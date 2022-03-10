from pickletools import float8
from typing import List

import numpy as np
import torch
from flair.data import Sentence
from flair.embeddings import Embeddings

from seqal.data import Entities, Entity
from seqal.tagger import SequenceTagger


class BaseSampler:
    """BaseSampler class

    This is a base class to inherit for active learning sampling method.
    Each sampling method class should inherit this class.
    """

    def __call__(self):
        """Run active learning workflow

        This function in every sampling method should follow a specific workflow:
        - Predict on dataset
        - Get entities from dataset
        - Calculate score for each data
        - Sort data by scores
        - Query data id

        Raises:
            NotImplementedError: This method must be implemented
        """
        raise NotImplementedError

    def predict(self, sents: List[Sentence], tagger: SequenceTagger) -> None:
        """Predict unlabel data

        Args:
            sents (List[Sentence]): Sentences in data pool.
            tagger (Module): Trained model.
        """
        tagger.predict(sents, mini_batch_size=32)

    def sort(self, sent_scores: np.ndarray, order: str = "ascend") -> List[int]:
        """Sort sentence id based on sentence scores

        Args:
            sent_scores (np.ndarray): Sentence scores.
            order (str, optional): Sentence id order for reutrn . Defaults to "ascend".

        Raises:
            TypeError: if sent_scores is not np.ndarray
            ValueError: if order is not available

        Returns:
            List[int]: Sentence id by order
        """
        if isinstance(sent_scores, np.ndarray) is False:
            raise TypeError("'sent_scores' must be ndarray")

        if order == "ascend":
            ordered_indices = np.argsort(sent_scores)
        elif order == "descend":
            ordered_indices = np.argsort(-sent_scores)
        else:
            raise ValueError("Order option only accepts 'ascend' or 'descend'")

        return ordered_indices.tolist()

    def query(
        self,
        sents: List[Sentence],
        ordered_indices: list,
        query_number: int = 0,
        token_based: bool = False,
    ) -> List[int]:
        """Query data based on ordered indices.

        Args:
            sents (List[Sentence]): Sentences in data pool.
            ordered_indices (list): Ordered indices.
            query_number (int, optional): Batch query number. Defaults to 0.
            token_based (bool, optional): If true, using query number as token number to query data.
                                        If false, using query number as sentence number to query data.
        Returns:
            List[int]: The index of queried samples in sents.
        """
        if query_number <= 0:
            raise ValueError("query_number must be bigger than 0")

        if token_based is True:
            queried_tokens = 0
            queried_sent_id = []
            for sent_id in ordered_indices:
                sent = sents[sent_id]
                if queried_tokens < query_number:
                    queried_tokens += len(sent.tokens)
                    queried_sent_id.append(sent_id)
        else:
            if query_number == 0:
                queried_sent_id = ordered_indices[0]
                queried_sent_id = [queried_sent_id]
            else:
                if query_number > len(sents):
                    queried_sent_id = ordered_indices
                else:
                    queried_sent_id = ordered_indices[:query_number]

        return queried_sent_id

    def similarity_matrix(
        self, mat1: torch.Tensor, mat2: torch.Tensor, eps: float8 = 1e-8
    ) -> torch.Tensor:
        """Calculate similarity bewteen matrix

        https://en.wikipedia.org/wiki/Cosine_similarity

        Args:
            mat1 (torch.Tensor): Multiple entity embedding vectors. shape: (entity_count, embedding_dim)
            mat2 (torch.Tensor): Multiple entity embedding vectors. shape: (entity_count, embedding_dim)
            eps (float8, optional): Eps for numerical stability. Defaults to 1e-8.

        Returns:
            torch.Tensor: similarity of matrix. shape: (1, entity_count)
        """
        if not torch.is_tensor(mat1) or not torch.is_tensor(mat2):
            raise TypeError("Input type is not torch.Tensor")

        mat1 = mat1.double()
        mat2 = mat2.double()

        numerator = torch.mm(mat1, mat2.transpose(0, 1))
        mat1_n, mat2_n = (
            torch.linalg.norm(mat1, dim=1, keepdims=True),
            torch.linalg.norm(mat2, dim=1, keepdims=True),
        )
        norm_mul = torch.mm(mat1_n, mat2_n.transpose(0, 1))
        denominator = torch.max(norm_mul, torch.full_like(norm_mul, eps))
        sim_mt = numerator / denominator

        return sim_mt

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
