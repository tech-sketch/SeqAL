# Tutorial 5: Research and Annotation Mode

This tutorial shows explain what are research mode and annotation mode.


```python
from flair.embeddings import WordEmbeddings

from seqal.active_learner import ActiveLearner
from seqal.datasets import ColumnCorpus, ColumnDataset
from seqal.samplers import LeastConfidenceSampler
from seqal.utils import load_plain_text, add_tags

# 1~7

# 8. iteration
for i in range(iterations):
    # 9. query unlabeled sentences
    queried_samples, unlabeled_sentences = learner.query(
        unlabeled_sentences, query_number, token_based=token_based, research_mode=False
    )
```

The parameter `research_mode` controls which mode we use.

## Research mode

The research mode means we run experiment for research purpose. When we are doing research, we already have a labeled dataset. So we do not need people to annotate the data. We just want to simulate the active learning cycle to see the performance of model.

When model predicts, predicted labels will overwrite the glod labels. But we assume that human will assign glod labels. In case of add predicted labels to training dataset, we set the `research_mode` is `True`.

Make sure that we load the labeled data pool.

```python
from seqal.datasets import ColumnDataset

# 1~5

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

The annotation mode means we use SeqAL in a real annotation project, which means that the data pool do not contain labels. We set the `research_mode` is `False`.

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
