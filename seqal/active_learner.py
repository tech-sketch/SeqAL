import json
from collections import namedtuple
from typing import Callable, List, Tuple

from flair.data import Sentence
from flair.trainers import ModelTrainer

from seqal.datasets import Corpus
from seqal.tagger import SequenceTagger

LabelInfo = namedtuple("LabelInfo", "idx text label")


def get_label_names(corpus: Corpus, label_type: str) -> List[str]:
    """Get all label names from corpus

    Args:
        corpus (Corpus): Corpus contains train, valid, test data.

    Returns:
        List: label name list.
    """
    data = corpus.obtain_statistics(label_type=label_type)
    data = json.loads(data)
    label_names = []
    for value in data.values():
        label_names.extend(value["number_of_tokens_per_tag"].keys())
    label_names = list(set(label_names))

    return label_names


def remove_queried_samples(sents: List[Sentence], queried_idx: List[int]) -> None:
    """Remove queried data from data pool.

    Args:
        sents (List[Sentence]): Sentences in data pool.
        queried_idx (List[int]): Index list of queried data.
    """
    new_sents = []
    query_sents = []
    for i, sent in enumerate(sents):
        if i in queried_idx:
            query_sents.append(sent)
        else:
            new_sents.append(sent)
    return new_sents, query_sents


def save_label_info(sents: List[Sentence]) -> List[List[LabelInfo]]:
    """Save label information before prediction in case of overwriting.

    Args:
        sents (List[Sentence]): Sentences in data pool.

    Returns:
        List[List[LabelInfo]]: Labels information in each sentence.
    """
    labels_info = []

    for sent in sents:
        sent_label_info = []
        if len(sent.get_spans("ner")) != 0:
            for token in sent:
                tag = token.get_tag("ner")
                sent_label_info.append(LabelInfo(token.idx, token.text, tag.value))
        labels_info.append(sent_label_info)

    return labels_info


def load_label_info(
    sents: List[Sentence], labels_info: List[List[LabelInfo]]
) -> List[Sentence]:
    """Load label infomation after prediction.

    Args:
        sents (List[Sentence]): Sentences in data pool.
        labels_info (List[List[LabelInfo]]): Labels information in each sentence.

    Returns:
        List[Sentence]: Sentences in data pool.
    """
    for idx_sent, sent_label_info in enumerate(labels_info):
        if len(sent_label_info) != 0:
            for idx_token, token_label_info in enumerate(sent_label_info):
                sents[idx_sent][idx_token].add_tag("ner", token_label_info.label)

    return sents


class ActiveLearner:
    """Active learning workflow class.

    Args:
        corpus: Corpus contains train(labeled data), dev, test (data pool).
        query_strategy: Function providing the query strategy for the active learning loop,
            for instance, seqal.uncertainty.uncertainty_sampling.
        tagger_params: Parameters for model.
        trainer_params: Parameters for training process.
    Attributes:
        corpus: The corpus to be used in active learning loop.
        query_strategy: Sampler providing the query strategy for the active learning loop.
        tagger_params: Parameters for model.
        trainer_params: Parameters for training process.
        trained_tagger: The tagger to be used in the active learning loop.
        label_names: Labels

    """

    def __init__(
        self,
        corpus: Corpus,
        query_strategy: Callable,
        tagger_params: dict,
        trainer_params: dict,
    ) -> None:
        assert callable(query_strategy), "query_strategy must be callable"
        self.corpus = corpus
        self.query_strategy = query_strategy
        self.tagger_params = tagger_params
        self.trainer_params = trainer_params
        self.trained_tagger = None
        self.label_names = None

    def initialize(self, dir_path: str = "output/init_train") -> None:
        """Train model on labeled data.

        Args:
            dir_path (str, optional): Directory path to save log and model. Defaults to "output/init_train".
        """
        # Initialize sequence tagger
        tag_type = self.tagger_params["tag_type"]
        self.tagger_params["tag_dictionary"] = self.corpus.make_tag_dictionary(
            tag_type=tag_type
        )
        self.label_names = get_label_names(self.corpus, tag_type)

        tagger = SequenceTagger(**self.tagger_params)

        trainer = ModelTrainer(tagger, self.corpus)
        trainer.train(dir_path, **self.trainer_params)
        self.trained_tagger = tagger

    def query(
        self,
        sents: List[Sentence],
        query_number: int,
        token_based: bool = False,
        research_mode: bool = False,
    ) -> Tuple[List[Sentence], List[Sentence]]:
        """Query data from pool (sents).

        Args:
            sents (List[Sentence]): Data pool that consist of sentences.
            query_number (int): batch query number.
            token_based (bool, optional): If true, using query number as token number to query data.
                                          If false, using query number as sentence number to query data.
            research_mode (bool, optional): If ture, sents contains real NER tags.
                                            If false, sents do not contains NER tags.

        Returns:
            Tuple[List[Sentence], List[Sentence]]:
                sents: The data pool after removing query samples.
                queried_samples: Query samples.
        """
        tag_type = self.tagger_params["tag_type"]
        embeddings = self.tagger_params["embeddings"]

        if research_mode is True:
            # Save labels information before prediction in case of overwriting real NER tags.
            # This is because Flair will assign NER tags to token after prediction
            labels_info = save_label_info(sents)

        queried_sent_ids = self.query_strategy(
            sents,
            tag_type,
            query_number,
            token_based,
            tagger=self.trained_tagger,
            label_names=self.label_names,
            embeddings=embeddings,
        )

        if research_mode is True:
            # Reload the real NER labels
            sents = load_label_info(sents, labels_info)

        # Remove queried data from sents and create a new list to store queried data
        sents_after_remove, queried_samples = remove_queried_samples(
            sents, queried_sent_ids
        )

        return queried_samples, sents_after_remove

    def teach(
        self,
        queried_samples: List[Sentence],
        resume: bool = False,
        dir_path: str = "output/retrain",
    ) -> None:
        """Retrain model on new labeled dataset.

        Args:
            queried_samples (Sentence): new labeled data.
            resume (bool, optional): If true, train model on new labeled data.
                                     If false, train a new model on all labeled data.
            dir_path (str, optional): Directory path to save log and model. Defaults to "output/retrain".
        """
        if resume is True:
            self.resume(queried_samples, dir_path)
        else:
            for sample in queried_samples:
                self.corpus.train.sentences.append(sample)
            self.initialize(dir_path)

    def resume(
        self, queried_samples: List[Sentence], dir_path: str = "output/retrain"
    ) -> None:
        """Train model on the new labeled data"""
        self.corpus.train.sentences = queried_samples
        trainer = ModelTrainer(self.trained_tagger, self.corpus)
        trainer.train(dir_path, **self.trainer_params)
