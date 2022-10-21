import spacy
from flair.embeddings import FlairEmbeddings, StackedEmbeddings

from seqal.active_learner import ActiveLearner
from seqal.datasets import ColumnCorpus
from seqal.samplers import LeastConfidenceSampler
from seqal.transformer import Transformer

# 1. get the corpus
columns = {0: "text", 1: "ner"}
data_folder = "./data/sample_jp"
corpus = ColumnCorpus(
    data_folder,
    columns,
    train_file="train_seed.txt",
    dev_file="dev.txt",
    test_file="test.txt",
)

# 2. tagger params
tagger_params = {}
tagger_params["tag_type"] = "ner"
tagger_params["hidden_size"] = 256
embedding_types = [
    FlairEmbeddings("ja-forward"),
    FlairEmbeddings("ja-backward"),
]
embeddings = StackedEmbeddings(embeddings=embedding_types)
tagger_params["embeddings"] = embeddings
tagger_params["use_rnn"] = False

# 3. trainer params
trainer_params = {}
trainer_params["max_epochs"] = 1
trainer_params["mini_batch_size"] = 32
trainer_params["learning_rate"] = 0.1
trainer_params["patience"] = 5

# 4. setup active learner
sampler = LeastConfidenceSampler()
learner = ActiveLearner(corpus, sampler, tagger_params, trainer_params)

# 5. initialize active learner
learner.initialize(dir_path="output/init_train")

# 6. prepare data pool
pool_file = data_folder + "/labeled_data_pool.txt"
unlabeled_sentences = []
with open(pool_file, mode="r", encoding="utf-8") as f:
    for line in f:
        unlabeled_sentences.append(line)
nlp = spacy.load("ja_core_news_sm")
tokenizer = Transformer(nlp)
unlabeled_sentences = [
    tokenizer.to_subword(sentence) for sentence in unlabeled_sentences
]

# 7. query setup
query_number = 1
token_based = False
iterations = 4

# 8. iteration
for i in range(iterations):
    # 9. query unlabeled sentences
    queried_samples, unlabeled_sentences = learner.query(
        unlabeled_sentences, query_number, token_based=token_based, research_mode=False
    )

    # 10. retrain model, the queried_samples will be added to corpus.train
    learner.teach(queried_samples, dir_path=f"output/retrain_{i}")
