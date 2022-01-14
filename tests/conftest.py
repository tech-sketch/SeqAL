from pathlib import Path
from typing import List

import pytest
from flair.data import Sentence
from flair.embeddings import BytePairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger

from seqal.active_learner import ActiveLearner
from seqal.datasets import ColumnCorpus, Corpus
from seqal.query_strategies import random_sampling


class FakeEntity:
    def __init__(self, x: float):
        self.score = x


class FakeSentence:
    def __init__(self, entity_list: List[FakeEntity]):
        self.entity_list = entity_list

    def __len__(self):
        return len(self.entity_list)

    def get_spans(self, tag_type: str):
        return self.entity_list


@pytest.fixture
def fake_sents() -> List[FakeSentence]:
    e1 = FakeEntity(0.25)
    e2 = FakeEntity(0.3)
    e3 = FakeEntity(0.289)
    e4 = FakeEntity(0.2988)
    e5 = FakeEntity(0.2971)
    e6 = FakeEntity(0.268)
    e7 = FakeEntity(0.2707)
    e8 = FakeEntity(0.2679)
    e9 = FakeEntity(0.2942)
    e10 = FakeEntity(0.2902)
    e11 = FakeEntity(0.2831)
    e12 = FakeEntity(0.2821)
    e13 = FakeEntity(0.2712)
    e14 = FakeEntity(0.2646)

    s1 = FakeSentence([e1])
    s2 = FakeSentence([e2, e3])
    s3 = FakeSentence([e4, e5])
    s4 = FakeSentence([e6])
    s5 = FakeSentence([e7])
    s6 = FakeSentence([e8])
    s7 = FakeSentence([e9, e10, e11])
    s8 = FakeSentence([e12])
    s9 = FakeSentence([e13])
    s10 = FakeSentence([e14])

    sents = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
    return sents


@pytest.fixture
def sents() -> List[Sentence]:
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
