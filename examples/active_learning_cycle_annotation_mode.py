import json

from flair.embeddings import WordEmbeddings

from seqal.active_learner import ActiveLearner
from seqal.datasets import ColumnCorpus
from seqal.samplers import LeastConfidenceSampler
from seqal.utils import add_tags, count_tokens, load_plain_text

# 1. get the corpus
columns = {0: "text", 1: "ner"}
data_folder = "./data/trivial_bioes"
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
trainer_params["max_epochs"] = 1
trainer_params["mini_batch_size"] = 32
trainer_params["learning_rate"] = 0.1
trainer_params["train_with_dev"] = True

# 4. initialize learner
sampler = LeastConfidenceSampler()
learner = ActiveLearner(corpus, sampler, tagger_params, trainer_params)

# 5. initial training
learner.initialize(dir_path="output/init_train")

# 6. prepare unlabeled data pool
file_path = "./data/trivial_bioes/train_datapool.txt"
unlabeled_sentences = load_plain_text(file_path)

# 7. calculate query how many sentences in each iteration based on tokens
query_percent = 0.02  # Query 2% data at each iteration
token_based = True
seed_stats = json.loads(corpus.obtain_statistics())
datapool_stats = count_tokens(unlabeled_sentences)
total_tokens = seed_stats["TRAIN"]["number_of_tokens"]["total"] + datapool_stats
query_number = tokens_each_iteration = max(int(total_tokens * query_percent), 1)


# 8. iteration
for i in range(25):
    print(f"Annotate {int((i+1)*query_percent*100)}% data:")
    print(f"Number of the unlabeled sentences: {len(unlabeled_sentences)}")

    unlabeled_sentences, queried_samples = learner.query(
        unlabeled_sentences, query_number, token_based=True, simulation_mode=True
    )

    # Convert sentence class to plain text
    queried_texts = [{"text": sent.to_plain_string()} for sent in queried_samples]
    # queried_texts:
    # [
    #   {
    #     "text": "I love Berlin"
    #   }
    # ]

    # Annotator annotate the queried samples
    # 'annotate_by_human' method should be provide by user
    labeled_texts = annotate_by_human(queried_texts)  # noqa: F821
    # labeled_texts:
    # [
    #   {
    #     "text": "I love Berlin .",
    #     "labels": [  # The labels created by annotators
    #       {
    #         "start_pos": 7,
    #         "text": "Berlin",
    #         "label": "S-LOC"
    #       }
    #     ]
    #   }
    # ]
    print(f"Number of queried sentence: {len(queried_samples)}")
    print(f"Number of the rest sentences: {len(unlabeled_sentences)}")

    # Add labels to Sentence class
    labeled_samples = add_tags(labeled_texts)

    # 'teach' method adds labeled_samples to corpus.train and retrain model
    learner.teach(labeled_samples, dir_path=f"output/retrain_{i}")
    print(f"Number of the labeled data: {len(corpus.train)}")

    print()
