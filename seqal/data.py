import functools
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import torch
from flair.data import Span


@dataclass
class Entity:
    id: int  # Entity id in the same sentence
    sent_id: int  # Sentence id
    span: Span  # Entity span

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
        return self.span.tag


class Entities:
    def __init__(self):
        self.entities = []

    def add(self, entity: Entity):
        self.entities.append(entity)

    @functools.cached_property
    def group_by_sentence(self) -> Dict[int, List[Entity]]:
        entities_per_sentence = defaultdict(list)
        for entity in self.entities:
            entities_per_sentence[entity.sent_id].append(entity)
        return entities_per_sentence

    @functools.cached_property
    def group_by_label(self) -> Dict[str, List[Entity]]:
        entities_per_label = defaultdict(list)
        for entity in self.entities:
            entities_per_label[entity.label].append(entity)
        return entities_per_label
