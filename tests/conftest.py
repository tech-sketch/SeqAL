from pathlib import Path
from typing import List

import pytest
from flair.data import Sentence
from flair.embeddings import BytePairEmbeddings, StackedEmbeddings

from seqal.active_learner import ActiveLearner
from seqal.datasets import ColumnCorpus, Corpus
from seqal.samplers import RandomSampler
from seqal.tagger import SequenceTagger


@pytest.fixture
def fixture_path() -> Path:
    """Path to save file"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def unlabeled_sentences() -> List[Sentence]:
    """Unlabeled sentences for test

    These sentence are from fixture conll/eng.train
    """
    s1 = Sentence("EU rejects German call to boycott British lamb .")
    s2 = Sentence("Peter Blackburn")
    s3 = Sentence("BRUSSELS 1996-08-22")
    s4 = Sentence(
        "The European Commission said on Thursday it disagreed with German advice to consumers to shun "
        "British lamb until scientists determine whether mad cow disease can be transmitted to sheep ."
    )
    s5 = Sentence(
        "Germany 's representative to the European Union 's veterinary committee Werner Zwingmann "
        "said on Wednesday consumers should buy sheepmeat from countries other than Britain until "
        "the scientific advice was clearer ."
    )
    s6 = Sentence(
        """ We do n't support any such recommendation because we do n't see any grounds for it , """
        """" the Commission 's chief spokesman Nikolaus van der Pas told a news briefing ."""
    )
    s7 = Sentence(
        "He said further scientific study was required and if it was found that action"
        "was needed it should be taken by the European Union ."
    )
    s8 = Sentence(
        "He said a proposal last month by EU Farm Commissioner Franz Fischler to ban sheep brains , "
        "spleens and spinal cords from the human "
        "and animal food chains was a highly specific and precautionary move to protect human health ."
    )
    s9 = Sentence(
        "Fischler proposed EU-wide measures after reports from Britain and France that under "
        "laboratory conditions sheep could contract Bovine Spongiform Encephalopathy ( BSE ) -- mad cow disease ."
    )
    s10 = Sentence(
        "But Fischler agreed to review his proposal after the EU 's standing veterinary committee , "
        "mational animal health officials , questioned if such action was justified as there was only "
        "a slight risk to human health ."
    )

    sents = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
    return sents


@pytest.fixture
def trained_tagger(
    fixture_path: Path, corpus: Corpus, embeddings: StackedEmbeddings
) -> SequenceTagger:
    """A trained tagger used for test"""
    tagger_params = {}
    tagger_params["tag_type"] = "ner"  # what tag do we want to predict?
    tagger_params["hidden_size"] = 256
    tagger_params["embeddings"] = embeddings

    trainer_params = {}
    trainer_params["max_epochs"] = 1
    trainer_params["learning_rate"] = 0.1
    trainer_params["train_with_dev"] = True
    trainer_params["train_with_test"] = True
    random_sampler = RandomSampler()
    learner = ActiveLearner(corpus, random_sampler, tagger_params, trainer_params)

    save_path = fixture_path / "output"
    learner.initialize(save_path)

    return learner.trained_tagger


@pytest.fixture
def predicted_sentences(
    unlabeled_sentences: List[Sentence], trained_tagger: SequenceTagger
) -> List[Sentence]:
    """Sentences after prediction for test"""
    trained_tagger.predict(unlabeled_sentences)
    return unlabeled_sentences


@pytest.fixture
def corpus(fixture_path: Path) -> Corpus:
    """A corpus used for test"""
    columns = {0: "text", 1: "pos", 3: "ner"}
    data_folder = fixture_path / "conll"
    corpus = ColumnCorpus(
        data_folder,
        columns,
        train_file="eng.train",
        test_file="eng.testb",
        dev_file="eng.testa",
    )
    return corpus


@pytest.fixture
def labeled_sentences_after_prediction(
    corpus: Corpus, trained_tagger: SequenceTagger
) -> List[Sentence]:
    """Labeled sentences after being predicted

    The labels in sentences should be updated
    """
    label_sents = corpus.train.sentences  # Labeled sentences
    trained_tagger.predict(label_sents)
    return label_sents


@pytest.fixture
def embeddings(fixture_path: Path) -> StackedEmbeddings:
    """Embeddings used for test"""
    model_file_path = fixture_path / "embeddings/en.wiki.bpe.vs100000.model"
    embedding_file_path = fixture_path / "embeddings/en.wiki.bpe.vs100000.d50.w2v.bin"
    byte_embedding = BytePairEmbeddings(
        model_file_path=model_file_path, embedding_file_path=embedding_file_path
    )
    embedding_types = [byte_embedding]
    embeddings = StackedEmbeddings(embeddings=embedding_types)

    return embeddings


@pytest.fixture
def learner(corpus: Corpus, embeddings: StackedEmbeddings) -> ActiveLearner:
    """A not trained learner for test"""
    tagger_params = {}
    tagger_params["tag_type"] = "ner"  # what tag do we want to predict?
    tagger_params["hidden_size"] = 256
    tagger_params["embeddings"] = embeddings

    trainer_params = {}
    trainer_params["max_epochs"] = 1
    trainer_params["learning_rate"] = 0.1
    trainer_params["train_with_dev"] = True
    trainer_params["train_with_test"] = True
    random_sampler = RandomSampler()
    learner = ActiveLearner(corpus, random_sampler, tagger_params, trainer_params)

    return learner


@pytest.fixture
def trained_learner(
    fixture_path: Path, corpus: Corpus, embeddings: StackedEmbeddings
) -> SequenceTagger:
    """A trained tagger used for test"""
    tagger_params = {}
    tagger_params["tag_type"] = "ner"  # what tag do we want to predict?
    tagger_params["hidden_size"] = 256
    tagger_params["embeddings"] = embeddings

    trainer_params = {}
    trainer_params["max_epochs"] = 1
    trainer_params["learning_rate"] = 0.1
    trainer_params["train_with_dev"] = True
    trainer_params["train_with_test"] = True
    random_sampler = RandomSampler()

    trained_learner = ActiveLearner(
        corpus, random_sampler, tagger_params, trainer_params
    )

    save_path = fixture_path / "output"
    trained_learner.initialize(save_path)

    return trained_learner
