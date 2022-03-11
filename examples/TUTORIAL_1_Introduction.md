# Tutorial 1: Introduction

This tutorial shows how to use SeqAL to perform active learning for NER(named entity recognition).

## 1 Prepare data

### Data format

The labeled data format should follow the [IOB](https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)) syntax.

An example with IOB2 format:

```
   Alex         B-PER
   is           O
   going        O
   to           O
   Los          B-LOC
   Angeles      I-LOC
   in           O
   California   B-LOC
```

The unlabeled data should be one sample a line.

Two samples in a text:

```
EU rejects German call to boycott British lamb.
Germany 's representative to the European Union 's veterinary committee Werner Zwingmann said on Wednesday consumers should buy sheepmeat from countries other than Britain until the scientific advice was clearer .
```

### Load Data

**Load Corpus**

We take [CoNLL-2003 dataset](https://www.clips.uantwerpen.be/conll2003/ner/) as an example to demonstrate how to read labeled data.


The CoNLL-2003 dataset has four columns. The 4th column is the NER tag.

```
   U.N.         NNP  I-NP  I-ORG 
   official     NN   I-NP  O 
   Ekeus        NNP  I-NP  I-PER 
   heads        VBZ  I-VP  O 
   for          IN   I-PP  O 
   Baghdad      NNP  I-NP  I-LOC 
   .            .    O     O 
```

For such data, we import `ColumnCorpus` class and provide a `columns` variable to specify which column is the name entity tag. 

```python
from seqal.datasets import ColumnCorpus

columns = {0: "text", 3: "ner"}
data_folder = "./datasets/conll"
corpus = ColumnCorpus(
    data_folder,
    columns,
    train_file="train_seed.txt",
    dev_file="valid.txt",
    test_file="test.txt",
)
```

The `valid.txt` is the dataset used to give an estimate of model skill while tuning the model’s hyperparameters. The `test.txt` is the dataset used to give an unbiased estimate of the final tuned model. 

```python
# print the number of Sentences in the train split
print(len(corpus.train))

# print the number of Sentences in the test split
print(len(corpus.test))

# print the number of Sentences in the dev split
print(len(corpus.dev))
```

We can access one sentence in each dataset.
```python
# print the one Sentence in the training dataset
print(corpus.train[19])
```

This prints:
```
Sentence: "Germany imported 47,600 sheep from Britain last year , nearly half of total imports ."   [− Tokens: 15  − Token-Labels: "Germany <B-LOC> imported 47,600 sheep from Britain <B-LOC> last year , nearly half of total imports ."]
 ```

This sentence contains NER tags. We can print it with NER tags.

```python
print(corpus.train[19].to_tagged_string('ner'))
```

This prints:

```
Germany <B-LOC> imported 47,600 sheep from Britain <B-LOC> last year , nearly half of total imports .
```

The `seqal.datasets.ColumnCorpus` inherit from `flair.data.Corpus`. For the detail usage of `ColumnCorpus`, we recommend the flair tutorial about [Corpus](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_6_CORPUS.md), 


**Load Dataset**

If we just want to load labeled data instead of a corpus, we can use the `ColumnDataset` class. 


```python
from seqal.datasets import ColumnDataset

columns = {0: "text", 3: "ner"}
data_folder = "./datasets/conll"
pool_file = data_folder + "/train_datapool.txt"
data_pool = ColumnDataset(pool_file, columns)
unlabeled_sentences = data_pool.sentences
```

We can get sentences from data_pool by calling `sentences` property.

```python
print(data_pool.sentence[0])
```

This prints:

```
Sentence: "Taleban said its shipment of ammunition from Albania was evidence of Russian military support for Rabbani 's government ."   [− Tokens: 19  − Token-Labels: "Taleban <B-MISC> said its shipment of ammunition from Albania <B-LOC> was evidence of Russian <B-MISC> military support for Rabbani <B-PER> 's government ."]
```

**Load Unlabeled Dataset**

We can use `load_plain_text` to read the unlabeled dataset. This will create a list of `Sentence` objects.

```python
from seqal.utils import load_plain_text

file_path = "./datasets/conll/train_datapool.txt"
unlabeled_sentences = load_plain_text(file_path)
```

## 2 Initialize Active Learner

The model used for training is the Bi-LSTM CRF. We have to prepare the model parameters and training parameters in advance.

```python
from flair.embeddings import StackedEmbeddings, FlairEmbeddings

# model parameters
tagger_params = {}
tagger_params["tag_type"] = "ner"  # what kind of tag we want to predict?
tagger_params["hidden_size"] = 256
embeddings = StackedEmbeddings([FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')])
tagger_params["embeddings"] = embeddings

# training parameters
trainer_params = {}
trainer_params["max_epochs"] = 1
trainer_params["mini_batch_size"] = 32
trainer_params["learning_rate"] = 0.1
trainer_params["train_with_dev"] = True
```

When initializing the active learner, we have to provide the query strategy. Here we use `LeastConfidenceSampler`.

```python
from seqal.active_learner import ActiveLearner
from seqal.samplers import LeastConfidenceSampler

sampler = LeastConfidenceSampler()
learner = ActiveLearner(corpus, sampler, tagger_params, trainer_params)
```

Finally, we call `initialize()` method to train the model on seed data. We can provide path to save the model and trianing log.

```
learner.initialize(dir_path="output/init_train")
```


## 3 Active Learning Cycle

We assume that we need 1000 labeled data. So we run the active learning 5 iterations, and we query 200 samples in each iteration. These 200 samples are sent to workers for annotation. 

> The term 'sample' and 'data' are basically the seem. Both of them means one sentence. But we prefer to use 'sample' to represent the selected data.


```python
query_number = 200

for i in range(5):
    unlabeled_sentences, queried_samples = learner.query(
        unlabeled_sentences, query_number, token_based=False
    )
    learner.teach(queried_samples, dir_path=f"output/retrain_{i}")
```

The `token_based=False` parameter means we count on the sentence level. So we will get 200 sentences in `queried_samples`. If we set `token_based=True`, it means we count on the token level. If one sentence contains 10 tokens, we will get 10 sentences in `queried_samples`. 

The `queried_samples` contains the samples selected by query strategy. The `unlabeled_setence` contains the rest data.

`learner.teach()` add `queried_sampels` to seed data and retrain the model from scratch. 

In each iteration, the model will print the performance on different labels, like below:

```
Results:
- F-score (micro) 0.6969
- F-score (macro) 0.6603
- Accuracy 0.5495

By class:
              precision    recall  f1-score   support

         PER     0.8441    0.7934    0.8180      1617
         LOC     0.8431    0.6151    0.7113      1668
         ORG     0.7852    0.5105    0.6188      1661
        MISC     0.7943    0.3575    0.4931       702

   micro avg     0.8246    0.6034    0.6969      5648
   macro avg     0.8167    0.5692    0.6603      5648
weighted avg     0.8203    0.6034    0.6875      5648
 samples avg     0.5495    0.5495    0.5495      5648
```

