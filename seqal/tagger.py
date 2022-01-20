from typing import List, Tuple, Union

import flair.data
import torch
from flair.data import Sentence
from flair.datasets import DataLoader, SentenceDataset
from flair.models import SequenceTagger as FlairSequenceTagger
from flair.models.sequence_tagger_model import pad_tensors


class SequenceTagger(FlairSequenceTagger):
    def log_probability(
        self, sentences: List[Sentence], batch_szie: int = 32
    ) -> torch.tensor:
        """Calculate probability of each sentence.

        Args:
            sentences (List[Sentence]): Sentences must be predicted.
            batch_szie (int, optional): Defaults to 32.

        Returns:
            [type]: [description]
        """
        scores = []
        dataloader = DataLoader(
            dataset=SentenceDataset(sentences), batch_size=batch_szie
        )
        with torch.no_grad():
            for batch in dataloader:
                features = self.forward(batch)
                batch_scores, _ = self._calculate_loss(
                    features, batch, reduction="none"
                )
                scores.extend(batch_scores.neg().tolist())

        return torch.tensor(scores)

    def _calculate_loss(
        self, features: torch.tensor, sentences: List[Sentence], reduction: str = "sum"
    ) -> Tuple[Union[float, torch.Tensor], int]:
        """Overided _calculate_loss with reduction parameter

        Args:
            features (torch.tensor): features after forward
            sentences (List[Sentence]): sentence after prediction
            reduction (str, optional): reduction method. Defaults to "sum".

        Returns:
            Tuple[Union[float, torch.Tensor], int]: scores and token_count
        """

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

        tag_list: List = []
        token_count = 0
        for s_id, sentence in enumerate(sentences):
            # get the tags in this sentence
            tag_idx: List[int] = [
                self.tag_dictionary.get_idx_for_item(token.get_tag(self.tag_type).value)
                for token in sentence
            ]
            token_count += len(tag_idx)
            # add tags as tensor
            tag = torch.tensor(tag_idx, device=flair.device)
            tag_list.append(tag)

        if self.use_crf:
            # pad tags if using batch-CRF decoder
            tags, _ = pad_tensors(tag_list)

            forward_score = self._forward_alg(features, lengths)
            gold_score = self._score_sentence(features, tags, lengths)

            score = forward_score - gold_score
            if reduction == "sum":
                return score.sum(), token_count
            elif reduction == "none":
                return score, token_count
            elif reduction == "mean":
                return score.mean(), token_count
            else:
                raise ValueError("Invalid.")

        else:
            score = 0
            for sentence_feats, sentence_tags, sentence_length in zip(
                features, tag_list, lengths
            ):
                sentence_feats = sentence_feats[:sentence_length]
                score += torch.nn.functional.cross_entropy(
                    sentence_feats,
                    sentence_tags,
                    weight=self.loss_weights,
                    reduction=reduction,
                )

            return score, token_count
