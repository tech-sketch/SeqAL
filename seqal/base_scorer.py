from pickletools import float8
from typing import List

import numpy as np
import torch

from seqal.datasets import Sentence


class BaseScorer:
    def score(self):
        raise NotImplementedError

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
        self, a: torch.tensor, b: torch.tensor, eps: float8 = 1e-8
    ) -> torch.tensor:
        """Calculate similarity bewteen matrix

        https://en.wikipedia.org/wiki/Cosine_similarity

        Args:
            a (torch.tensor): Matrix of embedding. shape=(entity_count, embedding_dim)
            b (torch.tensor): Matrix of embedding. shape=(entity_count, embedding_dim)
            eps (float8, optional): Eps for numerical stability. Defaults to 1e-8.

        Returns:
            torch.tensor: similarity of matrix. shape=(entity_count, entity_count)
        """
        if torch.is_tensor(a) is False or torch.is_tensor(b) is False:
            raise TypeError("Input matrix type is not torch.tensor")
        if a.dtype != torch.float32:
            a = a.type(torch.float32)
        if b.dtype != torch.float32:
            b = b.type(torch.float32)

        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))

        return sim_mt

    def similarity_matrix(
        self, a: torch.tensor, b: torch.tensor, eps: float8 = 1e-8
    ) -> torch.tensor:
        """Calculate similarity bewteen matrix

        https://en.wikipedia.org/wiki/Cosine_similarity

        Args:
            a (torch.tensor): Matrix of embedding. shape=(entity_count, embedding_dim)
            b (torch.tensor): Matrix of embedding. shape=(entity_count, embedding_dim)
            eps (float8, optional): Eps for numerical stability. Defaults to 1e-8.

        Returns:
            torch.tensor: similarity of matrix. shape=(entity_count, entity_count)
        """
        if torch.is_tensor(a) is False or torch.is_tensor(b) is False:
            raise TypeError("Input matrix type is not torch.tensor")
        if a.dtype != torch.float32:
            a = a.type(torch.float32)
        if b.dtype != torch.float32:
            b = b.type(torch.float32)

        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))

        return sim_mt

    def normalize_score(self):
        # TODO: This is used for combined scorer

        pass