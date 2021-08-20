import json

from flair.embeddings import StackedEmbeddings, WordEmbeddings

from seqal.active_learner import ActiveLearner
from seqal.datasets import ColumnCorpus, ColumnDataset
from seqal.query_strategies import mnlp_sampling

# 1. get the corpus
columns = {0: "text", 1: "pos", 3: "ner"}
data_folder = "seqal/datasets/conll"
corpus = ColumnCorpus(
    data_folder,
    columns,
    train_file="eng.train_seed_tokens",
    test_file="eng.testb",
    dev_file="eng.testa",
)

# 2. tagger params
tagger_params = {}
tagger_params["tag_type"] = "ner"  # what tag do we want to predict?
tagger_params["hidden_size"] = 256
embedding_types = [WordEmbeddings("glove")]
embeddings = StackedEmbeddings(embeddings=embedding_types)
tagger_params["embeddings"] = embeddings

# 3. Trainer params
trainer_params = {}
trainer_params["max_epochs"] = 1
trainer_params["mini_batch_size"] = 32
trainer_params["learning_rate"] = 0.1
trainer_params["train_with_dev"] = True

# 4. initialize learner
learner = ActiveLearner(tagger_params, mnlp_sampling, corpus, trainer_params)

# 5. initial training
learner.fit(save_path="output/init_train")

# 6. prepare data pool
pool_columns = {0: "text", 1: "pos", 3: "ner"}
pool_file = data_folder + "/eng.train_pool_tokens"
data_pool = ColumnDataset(pool_file, pool_columns)
sents = data_pool.sentences

# 7. calculate query how many sentences in each iteration based on tokens
query_percent = 0.02
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
    # query
    sents, query_samples = learner.query(
        sents, query_number, token_based=True, simulation_mode=True
    )
    print(f"Number of the rest sentences: {len(sents)}")
    print(f"Number of queried sentence: {len(query_samples)}")
    # retrain, query_samples will be added to corpus.train
    learner.teach(query_samples, save_path=f"output/retrain_{i}")
    print()
