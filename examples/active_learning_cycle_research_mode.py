import json

from flair.embeddings import WordEmbeddings

from seqal.active_learner import ActiveLearner
from seqal.datasets import ColumnCorpus, ColumnDataset
from seqal.samplers import LeastConfidenceSampler

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

# 6. prepare data pool
pool_file = data_folder + "/train_labeled_pool.txt"
data_pool = ColumnDataset(pool_file, columns)
unlabeled_sentences = data_pool.sentences

# 7. calculate query how many sentences in each iteration based on tokens
query_percent = 0.02  # Query 2% data at each iteration
token_based = True
seed_stats = json.loads(corpus.obtain_statistics())
datapool_stats = data_pool.obtain_statistics()
total_tokens = (
    seed_stats["TRAIN"]["number_of_tokens"]["total"]
    + datapool_stats["number_of_tokens"]["total"]
)
query_number = tokens_each_iteration = int(total_tokens * query_percent)


# 8. iteration
for i in range(25):
    print(f"Annotate {int((i+1)*query_percent*100)}% data:")
    print(f"Number of the unlabeled sentences: {len(unlabeled_sentences)}")

    unlabeled_sentences, queried_samples = learner.query(
        unlabeled_sentences, query_number, token_based=True, simulation_mode=True
    )
    print(f"Number of queried sentence: {len(queried_samples)}")
    print(f"Number of the rest sentences: {len(unlabeled_sentences)}")

    print(f"Number of the labeled data: {len(corpus.train)}")

    # queried_samples will be added to corpus.train
    learner.teach(queried_samples, dir_path=f"output/retrain_{i}")
    print()
