# Welcome to SeqAL

SeqAL is a sequence labeling active learning framework based on Flair.

Please follow the tutorial to know the usage of SeqAL.

## Active Learning Cycle

To understand what SeqAL can do, we first introduce the pool-based active learning cycle.

![al_cycle](./images/al_cycle.png)

- Step 0: Prepare seed data (a small number of labeled data used for training)
- Step 1: Train the model with seed data
  - Step 2: Predict unlabeled data with the trained model
  - Step 3: Query informative samples based on predictions
  - Step 4: Annotator (Oracle) annotate the selected samples
  - Step 5: Input the new labeled samples to labeled dataset
  - Step 6: Retrain model
- Repeat step2~step6 until the f1 score of the model beyond the threshold or annotation budget is no left

## Active Learning Cycle Implementation

SeqAL can cover all steps except step 0 and step 4. Below is a simple script to demonstrate how to use SeqAL to implement the workflow.


```python
from seqal.active_learner import ActiveLearner
from seqal.samplers import LeastConfidenceSampler
from seqal.alinger import Alinger
from seqal.datasets import ColumnCorpus
from seqal.utils import load_plain_text
from xxxx import annotate_by_human  # User need to prepare this method


# Step 0: Preparation
## Prepare Seed data, valid data, and test data
columns = {0: "text", 1: "pos", 2: "syntactic_chunk", 3: "ner"}
data_folder = "./datasets/conll"
corpus = ColumnCorpus(
    data_folder,
    columns,
    train_file="train_seed.txt",
    dev_file="valid.txt",
    test_file="test.txt",
)

## Unlabeled data pool
file_path = "./datasets/conll/train_datapool.txt"
unlabeled_sentences = load_plain_text(file_path)

## Initilize ActiveLearner
learner = ActiveLearner(
  tagger_params=tagger_params,   # Model parameters (hidden size, embedding, etc.)
  query_strategy=LeastConfidenceSampler(),  # Query algorithm
  corpus=corpus,                 # Corpus contains training, validation, test data
  trainer_params=trainer_params  # Trainer parameters (epoch, batch size, etc.)
)

# Step 1: Initial training on model
learner.initialize()

# Step 2&3: Predict on unlabeled data and query informative data
_, queried_samples = learner.query(data_pool)
queried_samples = [{"text": sent.to_plain_string()} for sent in queried_samples]  # Convert sentence class to plain text
# queried_samples:
# [
#   {
#     "text": "Tokyo is a city"
#   }
# ]

# Step 4: Annotator annotate the selected samples
new_labels = annotate_by_human(queried_samples)
# new_labels:
# [
#   {
#     "text": ['Tokyo', 'is', 'a', 'city'],
#     "labels": ['B-LOC', 'O', 'O', 'O']
#   }
# ]

## Convert data to the suitable format
alinger = Alinger()
new_labeled_samples = alinger.add_tags_on_token(new_labels, 'ner')

# Step 5&6: Add new labeled samples to training and retrain model
learner.teach(new_labeled_samples)
```
