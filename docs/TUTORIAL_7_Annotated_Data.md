# Tutorial 7: Annotated Data

This tutorial shows how to receive the labeled data.

Below is the demo example.

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

    # 10. annotate data
    annotated_data = human_annotate(queried_samples)

    # 11. retrain model with newly added queried_samples
    queried_samples = add_tags(annotated_data)
    learner.teach(queried_samples, dir_path=f"output/retrain_{i}")
```

In the step 10, it receive the labeled data.

If `annotated_data` has below format, we can use the `add_tags()` to get a list of `flair.data.Sentence`.

```json
[
  {
    "text": "I love Berlin.",
    "labels": [
      {
        "start_pos": 7,
        "text": "Berlin",
        "label": "B-LOC"
      }
    ]
  },
  {
    "text": "Tokyo is a city",
    "labels": [
      {
        "start_pos": 0,
        "text": "Tokyo",
        "label": "B-LOC"
      }
    ]
  }
]
```
