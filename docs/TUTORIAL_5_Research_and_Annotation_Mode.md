# Research and Annotation Mode

This tutorial shows explain what are research mode and annotation mode.

```python
from flair.embeddings import WordEmbeddings

from seqal.active_learner import ActiveLearner
from seqal.datasets import ColumnCorpus, ColumnDataset
from seqal.samplers import LeastConfidenceSampler
from seqal.utils import load_plain_text, add_tags

# 1. get the corpus
columns = {0: "text", 1: "ner"}
data_folder = "./data/conll"
corpus = ColumnCorpus(
    data_folder,
    columns,
    train_file="train_seed.txt",
    dev_file="valid.txt",
    test_file="test.txt",
)

# 2. tagger params
tagger_params = {}
tagger_params["tag_type"] = "ner"
tagger_params["hidden_size"] = 256
embeddings = WordEmbeddings("glove")
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
file_path = "./datasets/conll/train_pool.txt"
unlabeled_sentences = load_plain_text(file_path)

# 7. query setup
query_number = 10
token_based = False
iterations = 5

# 8. iteration
for i in range(iterations):
    # 9. query unlabeled sentences
    queried_samples, unlabeled_sentences = learner.query(
        unlabeled_sentences, query_number, token_based=token_based, research_mode=False
    )
```

The parameter `research_mode` controls which mode we use.

## Research mode

The research mode means we run the experiment for research purposes. When we are doing research, we already have a labeled dataset. So we do not need people to annotate the data. We just want to simulate the active learning cycle to see the performance of the model.

When the model predicts, predicted labels will overwrite the gold labels. But we assume that humans will assign glod labels. In the case of adding predicted labels to the training dataset, we set the `research_mode` as `True`.

Make sure that we load the labeled data pool.

```python
from seqal.datasets import ColumnDataset

# 1~5 steps can be found in Introduction

# 6. prepare data pool from conll format
columns = {0: "text", 1: "ner"}
pool_file = "./datasets/conll/train_pool.txt"
data_pool = ColumnDataset(pool_file, columns)
unlabeled_sentences = data_pool.sentences

# 7. query setup
query_number = 10
token_based = False
iterations = 5

# 8. iteration
for i in range(iterations):
    # 9. query unlabeled sentences
    queried_samples, unlabeled_sentences = learner.query(
        unlabeled_sentences, query_number, token_based=token_based, research_mode=True
    )
```

## Annotation mode

The annotation mode means we use SeqAL in a real annotation project, which means that the data pool does not contain labels. We set the `research_mode` as `False`.

```python
from seqal.datasets import ColumnDataset
from seqal.utils import load_plain_text

# 6. prepare data pool from conll format
columns = {0: "text"}
pool_file = "./datasets/conll/train_pool.txt"
data_pool = ColumnDataset(pool_file, columns)
unlabeled_sentences = data_pool.sentences

# 6. prepare data pool from plain text
file_path = "./datasets/conll/train_pool.txt"
unlabeled_sentences = load_plain_text(file_path)

# 7. query setup
query_number = 10
token_based = False
iterations = 5

# 8. iteration
for i in range(iterations):
    # 9. query unlabeled sentences
    queried_samples, unlabeled_sentences = learner.query(
        unlabeled_sentences, query_number, token_based=token_based, research_mode=False
    )
```
