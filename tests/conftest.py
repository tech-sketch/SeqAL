from pathlib import Path

import pytest
from flair.embeddings import BytePairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger

from seqal.active_learner import ActiveLearner
from seqal.datasets import ColumnCorpus, Corpus
from seqal.query_strategies import random_sampling


@pytest.fixture
def fixture_path() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def corpus(fixture_path: Path) -> Corpus:
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
def embeddings(fixture_path: Path) -> StackedEmbeddings:
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
    tagger_params = {}
    tagger_params["tag_type"] = "ner"  # what tag do we want to predict?
    tagger_params["hidden_size"] = 256
    tagger_params["embeddings"] = embeddings

    trainer_params = {}
    trainer_params["max_epochs"] = 1
    trainer_params["learning_rate"] = 0.1
    trainer_params["train_with_dev"] = True
    trainer_params["train_with_test"] = True
    learner = ActiveLearner(tagger_params, random_sampling, corpus, trainer_params)

    return learner


@pytest.fixture
def trained_tagger(
    fixture_path: Path, corpus: Corpus, embeddings: StackedEmbeddings
) -> SequenceTagger:
    tagger_params = {}
    tagger_params["tag_type"] = "ner"  # what tag do we want to predict?
    tagger_params["hidden_size"] = 256
    tagger_params["embeddings"] = embeddings

    trainer_params = {}
    trainer_params["max_epochs"] = 1
    trainer_params["learning_rate"] = 0.1
    trainer_params["train_with_dev"] = True
    trainer_params["train_with_test"] = True
    learner = ActiveLearner(tagger_params, random_sampling, corpus, trainer_params)

    save_path = fixture_path / "output"
    learner.fit(save_path)

    return learner.trained_tagger
