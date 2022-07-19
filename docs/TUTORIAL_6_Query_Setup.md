# Tutorial 6: Query Setup

This tutorial shows how to make reasonable query.

```python
from flair.embeddings import WordEmbeddings

from seqal.active_learner import ActiveLearner
from seqal.datasets import ColumnCorpus, ColumnDataset
from seqal.samplers import LeastConfidenceSampler
from seqal.utils import load_plain_text, add_tags

# 1~6

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

The query setup above is to query 10 sentences in each iterations. Usually we will make more 'smarter' query setup.


## Query on Sentence

If we set `token_based` as `False`, this will count `query_number` as a sentence number. We prefer to give a percentage query number to query data instead of a fixed query number. 

```python

# 7. query setup
query_percent = 0.02
token_based = False
total_sentences = len(corpus.train.sentences) + len(data_pool.sentences)
query_number = int(total_sentences * query_percent)  # queried sentences in each iteration

# 8. iteration
for i in range(iterations):
    # 9. query unlabeled sentences
    queried_samples, unlabeled_sentences = learner.query(
        unlabeled_sentences, query_number, token_based=token_based, research_mode=False
    )
```

The `query_percent` could be `0.01` or `0.02`.


## Query on Token

If we set `token_based` as `True`, this will count `query_number` as a token number. In the real case, we usually query data based on the token. Because we don't know how many tokens a queried sentence has.


```python

from seqal.utils import count_tokens

# 7. query setup
query_percent = 0.02
token_based = True
total_tokens = count_tokens(corpus.train.sentences) + count_tokens(data_pool.sentences)
query_number = tokens_each_iteration = int(total_tokens * query_percent)  # queried tokens in each iteration

# 8. iteration
for i in range(iterations):
    # 9. query unlabeled sentences
    queried_samples, unlabeled_sentences = learner.query(
        unlabeled_sentences, query_number, token_based=token_based, research_mode=False
    )
```
