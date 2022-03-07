import functools
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from flair.data import Span


@dataclass
class Entity:
    """Entity with predicted information"""

    id: int  # Entity id in the same sentence
    sent_id: int  # Sentence id
    span: Span  # Entity span
    cluster: Optional[int] = None  # Cluster number

    @property
    def vector(self) -> torch.Tensor:
        """Get entity embeddings"""
        embeddings = [token.embedding for token in self.span.tokens]
        if not any([e.nelement() != 0 for e in embeddings]):
            raise TypeError(
                "Tokens have no embeddings. Make sure embedding sentence first."
            )
        return torch.mean(torch.stack(embeddings), dim=0)

    @property
    def label(self) -> str:
        """Get entity label"""
        return self.span.tag

    @property
    def text(self) -> str:
        """Get entity text"""
        return self.span.text


class Entities:
    """Entity list"""

    def __init__(self):
        self.entities = []

    def add(self, entity: Entity):
        """Add entity to list"""
        self.entities.append(entity)

    @functools.cached_property
    def group_by_sentence(self) -> Dict[int, List[Entity]]:
        """Group entities by sentence"""
        entities_per_sentence = defaultdict(list)
        for entity in self.entities:
            entities_per_sentence[entity.sent_id].append(entity)
        return entities_per_sentence

    @functools.cached_property
    def group_by_label(self) -> Dict[str, List[Entity]]:
        """Group entities by label"""
        entities_per_label = defaultdict(list)
        for entity in self.entities:
            entities_per_label[entity.label].append(entity)
        return entities_per_label

    @functools.cached_property
    def group_by_cluster(self) -> Dict[str, List[Entity]]:
        """Group entities by cluster"""
        entities_per_cluster = defaultdict(list)
        for entity in self.entities:
            entities_per_cluster[entity.cluster].append(entity)
        return entities_per_cluster
