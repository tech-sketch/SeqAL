from typing import List, Tuple, Union

import flair.data
import numpy as np
import torch
from flair.data import Sentence
from flair.datasets import DataLoader, SentenceDataset
from flair.models import SequenceTagger as FlairSequenceTagger
from flair.models.sequence_tagger_model import pad_tensors


class SequenceTagger(FlairSequenceTagger):
    def log_probability(
        self, sentences: List[Sentence], batch_szie: int = 32
    ) -> np.array:
        """Calculate probability of each sentence.

        Args:
            sentences (List[Sentence]): Sentences must be predicted.
            batch_szie (int, optional): Defaults to 32.

        Returns:
            [np.array]: The log probability of each sentences
        """
        scores = []
        dataloader = DataLoader(
            dataset=SentenceDataset(sentences), batch_size=batch_szie
        )
        with torch.no_grad():
            for batch in dataloader:
                features = self.forward(batch)
                batch_loss, _ = self._calculate_loss(features, batch, reduction="none")
                scores.extend(batch_loss.neg().tolist())

        return np.array(scores)

    def _calculate_loss(
        self, features: torch.Tensor, sentences: List[Sentence], reduction: str = "sum"
    ) -> Tuple[Union[float, torch.Tensor], int]:
        """Overided FlairSequenceTagger._calculate_loss with reduction parameter

        Args:
            features (torch.Tensor): features after forward
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

    @staticmethod
    def _init_model_with_state_dict(state):

        rnn_type = "LSTM" if "rnn_type" not in state.keys() else state["rnn_type"]
        use_dropout = 0.0 if "use_dropout" not in state.keys() else state["use_dropout"]
        use_word_dropout = (
            0.0 if "use_word_dropout" not in state.keys() else state["use_word_dropout"]
        )
        use_locked_dropout = (
            0.0
            if "use_locked_dropout" not in state.keys()
            else state["use_locked_dropout"]
        )

        train_initial_hidden_state = (
            False
            if "train_initial_hidden_state" not in state.keys()
            else state["train_initial_hidden_state"]
        )
        beta = 1.0 if "beta" not in state.keys() else state["beta"]
        weights = None if "weight_dict" not in state.keys() else state["weight_dict"]
        reproject_embeddings = (
            True
            if "reproject_embeddings" not in state.keys()
            else state["reproject_embeddings"]
        )
        if "reproject_to" in state.keys():
            reproject_embeddings = state["reproject_to"]

        model = SequenceTagger(
            hidden_size=state["hidden_size"],
            embeddings=state["embeddings"],
            tag_dictionary=state["tag_dictionary"],
            tag_type=state["tag_type"],
            use_crf=state["use_crf"],
            use_rnn=state["use_rnn"],
            rnn_layers=state["rnn_layers"],
            dropout=use_dropout,
            word_dropout=use_word_dropout,
            locked_dropout=use_locked_dropout,
            train_initial_hidden_state=train_initial_hidden_state,
            rnn_type=rnn_type,
            beta=beta,
            loss_weights=weights,
            reproject_embeddings=reproject_embeddings,
        )
        model.load_state_dict(state["state_dict"])
        return model
