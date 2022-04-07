# Tutorial 2: Multiple Language Support

This tutorial shows how to use SeqAL for different language.

[Tutorial 1](./TUTORIAL_1_Introduction.md) show example on English. If you want to use SeqAL on other language, you only need to do two things. 

1. Prepare the data with IOB2 format on the language you used
2. Prepare the embedding on the same language

## Example 

Below is the simple example of active learning cycle on English.

```python
from seqal.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings
from seqal.active_learner import ActiveLearner
from seqal.samplers import LeastConfidenceSampler

## 1 Load data
columns = {0: "text", 3: "ner"}
data_folder = "./datasets/conll"
corpus = ColumnCorpus(
    data_folder,
    columns,
    train_file="train_seed.txt",
    dev_file="valid.txt",
    test_file="test.txt",
)  # Change to the dataset on the language you used

## 2 Initialize Active Learner
tagger_params = {}
tagger_params["tag_type"] = "ner" 
tagger_params["hidden_size"] = 256
embeddings = WordEmbeddings("glove")  # Prepare the embedding on the same language
tagger_params["embeddings"] = embeddings

trainer_params = {}
trainer_params["max_epochs"] = 1
trainer_params["mini_batch_size"] = 32
trainer_params["learning_rate"] = 0.1
trainer_params["train_with_dev"] = True

sampler = LeastConfidenceSampler()
learner = ActiveLearner(corpus, sampler, tagger_params, trainer_params)


## 3 Active Learning Cycle
query_number = 200

for i in range(5):
    unlabeled_sentences, queried_samples = learner.query(
        unlabeled_sentences, query_number, token_based=False
    )
    learner.teach(queried_samples, dir_path=f"output/retrain_{i}")
```

If you want to change the language, you just need change few lines. Below is an example on German.

First, changing the dataset on the language that you used.
```python
data_folder = "./datasets/conll_deu"
corpus = ColumnCorpus(
    data_folder,
    columns,
    train_file="train.txt",
    test_file="test.txt",
    dev_file="dev.txt",
)
```

Next, changing the embedding with the same language.
```python
embedding = WordEmbeddings("de")
```

Check out the full list of all word embeddings models [here](https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/CLASSIC_WORD_EMBEDDINGS.md), along with more explanations on the `WordEmbeddings` class.

We also can use the contextualized embedding.
```python
from flair.embeddings import BertEmbeddings

embedding = BertEmbeddings("bert-base-german-cased")
```

[Here](https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/TRANSFORMER_EMBEDDINGS.md) are more explanations on the `TransformerWordEmbeddings` class. We can find more language embeddings on [](https://huggingface.co/models)
