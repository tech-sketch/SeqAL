from flair.embeddings import WordEmbeddings
from xxxx import (  # User need to prepare this method to interact with annotation tool
    annotate_by_human,
)

from seqal.active_learner import ActiveLearner
from seqal.aligner import Aligner
from seqal.datasets import ColumnCorpus
from seqal.samplers import LeastConfidenceSampler
from seqal.utils import load_plain_text

# 0: prepare seed data, validation data, test data, and unlabeled data pool
# - labeled data:
#     - seed data: `train_seed.txt`
#     - validation data: `dev.txt`
#     - test data: `test.txt`
# - unlabeled data:
#     - unlabeled data pool: `train_datapool.txt`

# 1. get the corpus
columns = {0: "text", 1: "ner"}
data_folder = "./data/sample_bio"
corpus = ColumnCorpus(
    data_folder,
    columns,
    train_file="train_seed.txt",
    dev_file="dev.txt",
    test_file="test.txt",
)

# 2. tagger params
tagger_params = {}
tagger_params["tag_type"] = "ner"  # what tag do we want to predict?
tagger_params["hidden_size"] = 256
embeddings = WordEmbeddings("glove")
tagger_params["embeddings"] = embeddings

# 3. Trainer params
trainer_params = {}
trainer_params["max_epochs"] = 10
trainer_params["mini_batch_size"] = 32
trainer_params["learning_rate"] = 0.1
trainer_params["train_with_dev"] = True

# 4. initialize learner
sampler = LeastConfidenceSampler()
learner = ActiveLearner(corpus, sampler, tagger_params, trainer_params)

# 5. initial training
learner.initialize(dir_path="output/init_train")

# 6. prepare unlabeled data pool
file_path = "./data/sample_bio/train_datapool.txt"
unlabeled_sentences = load_plain_text(file_path)

# 7. query setup
query_number = 2
token_based = False
iterations = 5

# initialize the tool to read annotated data
aligner = Aligner()

# 8. iteration
for i in range(iterations):
    # 9. query unlabeled sentences
    queried_samples, unlabeled_sentences = learner.query(
        unlabeled_sentences, query_number, token_based=token_based, research_mode=False
    )

    # 10. convert sentence to plain text
    queried_texts = [{"text": sent.to_plain_string()} for sent in queried_samples]
    # queried_texts:
    # [
    #   {
    #     "text": "I love Berlin"
    #   },
    #   {
    #     "text": "Tokyo is a city"
    #   }
    # ]

    # 11. send queried_texts to annotation tool
    # annotator annotate the queried samples
    # 'annotate_by_human' method should be provide by user
    annotated_data = annotate_by_human(queried_texts)
    # annotated_data:
    # [
    #     {
    #         "text": ['I', 'love', 'Berlin'],
    #         "labels": ['O', 'O', 'B-LOC']
    #     }
    #     {
    #         "text": ['Tokyo', 'is', 'a', 'city'],
    #         "labels": ['B-LOC', 'O', 'O', 'O']
    #     }
    # ]

    # 12. convert data to sentence
    queried_samples = aligner.align_spaced_language(annotated_data)

    # 13. retrain model, the queried_samples will be added to corpus.train
    learner.teach(queried_samples, dir_path=f"output/retrain_{i}")
