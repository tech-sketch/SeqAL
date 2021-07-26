from flair.embeddings import StackedEmbeddings, WordEmbeddings
from flair.models import SequenceTagger

from seqal.active_learner import ActiveLearner
from seqal.datasets import ColumnCorpus
from seqal.query_strategies import mnlp_sampling

# 1. get the corpus
columns = {0: "text", 1: "pos", 3: "ner"}
data_folder = "../conll"
corpus = ColumnCorpus(
    data_folder,
    columns,
    train_file="eng.train_seed",
    test_file="eng.testb",
    dev_file="eng.testa",
)
corpus_pool = ColumnCorpus(
    data_folder,
    columns,
    train_file="eng.train_pool",
    test_file="eng.testb",
    dev_file="eng.testa",
)

# 2. what tag do we want to predict?
tag_type = "ner"

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# 4. initialize embeddings
embedding_types = [WordEmbeddings("glove")]
embeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type=tag_type,
)


# 6. initialize learner
params = {}
params["max_epochs"] = 1
params["learning_rate"] = 0.1
params["train_with_dev"] = True
params["train_with_test"] = True

learner = ActiveLearner(tagger, mnlp_sampling, corpus, **params)

# 7. initial training
learner.fit(save_path="output/init_train")

# 8. iteration
sents = corpus_pool.train.sentences
percent = 0.2
query_number = int(len(corpus.train) * 0.2)  # 2297

for i in range(25):
    print(f"Annotate {(i+1)*2}% data:")
    # query
    sents, query_samples = learner.query(sents, query_number)
    # retrain
    learner.teach(query_samples, save_path=f"output/retrain_{i}")
    print()
