## Usage

### Prepare data

The tagging scheme is the IOB scheme.

```
    U.N. NNP I-ORG
official NN  O
   Ekeus NNP I-PER
   heads VBZ O
     for IN  O
 Baghdad NNP I-LOC
       . .   O
```

Each line contains three fields: the word, its part-of-speech (POS) tag and its named entity tag. Words tagged with O are outside of named entities. The POS tag is optional. 

### Example

Because SeqAL is based on flair, we heavily recommend to read the [tutorial](https://github.com/flairNLP/flair/blob/5c4231b30865bf4426ba8076eb91492d329c8a9b/resources/docs/TUTORIAL_1_BASICS.md) of flair first. 

```python
from flair.embeddings import StackedEmbeddings, WordEmbeddings

from seqal.active_learner import ActiveLearner
from seqal.datasets import ColumnCorpus, ColumnDataset
from seqal.query_strategies import mnlp_sampling

# 1. get the corpus
columns = {0: "text", 1: "pos", 2: "ner"}
data_folder = "../conll"
corpus = ColumnCorpus(
    data_folder,
    columns,
    train_file="seed.data",
    dev_file="dev.data",
    test_file="test.data",
)
```

First we need to create the corpus. `date_folder` is the directry path where we store datasets. `seed.data` contains NER labels, which usually just a small part of data (around 2% of total train data). `dev.data` and `test.data` should contains NER labels for evaluation. All three kinds of data should follow the IOB scheme. But if you have 4 columns, you can just change `columns` to specify the tag column.


```python
# 2. tagger params
tagger_params = {}
tagger_params["tag_type"] = "ner"  # what tag do we want to predict?
tagger_params["hidden_size"] = 256
embedding_types = [WordEmbeddings("glove")]
embeddings = StackedEmbeddings(embeddings=embedding_types)
tagger_params["embeddings"] = embeddings

# 3. Trainer params
trainer_params = {}
trainer_params["max_epochs"] = 10
trainer_params["mini_batch_size"] = 32
trainer_params["learning_rate"] = 0.01
trainer_params["train_with_dev"] = True

# 4. initialize learner
learner = ActiveLearner(tagger_params, mnlp_sampling, corpus, trainer_params)
```

This part is where we set the parameters for sequence tagger and trainer. The above setup can conver most of situations. If you want to add more paramters, we recommend to the read [SequenceTagger](https://github.com/flairNLP/flair/blob/master/flair/models/sequence_tagger_model.py#L68) and [ModelTrainer](https://github.com/flairNLP/flair/blob/master/flair/trainers/trainer.py#L42) in flair.

The embeddings will download by flair, you can check available embeddings in [List of All Word Embeddings](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md)

```python
# 5. Initial training
learner.fit(save_path="output/init_train")
```

The initial training will be trained on the seed data. We usually give a `save_path` to save the trained model and training logs.

```python
# 6. prepare data pool
pool_columns = {0: "text", 1: "pos"}
pool_file = data_folder + "/pool.data"
data_pool = ColumnDataset(pool_file, pool_columns)
sents = data_pool.sentences
```
Here we prepare the unlabeled data pool.

```python
# 7. query data
query_number = 20
sents, queried_samples = learner.query(sents, query_number, token_based=True)
```

We can query samples from data pool by the `learner.query()` method. `query_number` means how many sentence we want to query. 

If we set `token_based=True`, the `query_number` means how many tokens we want to query. We usually set `token_based=True` For the sequence labeling task. For examples, what have 5 sentences and each sentence have 4 tokens. If we set `query_number=15` and `token_based=True`, the returned `queried_samples` will contain 4 sentences (16 tokens > 15), and `sents` contains the rest of unqueried sentences (1 sentences).


`sents` and `queried_samples` are lists that contains Sentence class from flair. You can check more usaage of Sentence class on the flair [tutorial](https://github.com/flairNLP/flair/blob/master/resources/docs/HUNFLAIR_TUTORIAL_1_TAGGING.md) or [code]((https://github.com/flairNLP/flair/blob/c1e30faa63/flair/data.py#L621)). For example, we can get the plain text by the `to_plain_string()` method. We can 

```
In [1]: queried_samples[0].to_plain_string()
Out[1]: 'I love Berlin .'
```

Next we need to annotate the selected data.

```python
from seqal.utils import add_tags

# 8. obtaining labels for "queried_samples" by the human
query_labels = annotate_by_human(queried_samples)
# query_labels = [
#       {
#         "text": "I love Berlin .",
#         "labels": [{"start_pos": 7, "text": "Berlin", "label": "S-LOC"}]
#       },
#       {
#         "text": "This book is great.",
#         "labels": []
#       }
# ]


annotated_sents = add_tags(query_labels)
```
`query_labels` is the label information of a sentence after annotation by human. We use such information to create flair Sentence class by calling `add_tags()` method.

For more detail, see [Adding labels to sentences](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_1_BASICS.md#adding-labels-to-sentences)


```python
# 9. retrain model with new labeled data
learner.teach(annotated_sents, save_path=f"output/retrain")
```

Finally, we call `learner.teach()` to retrain the model. The `annotated_sents` will be added to the training data (`corpus.train`) automatically.

If you want to run the workflow in a loop, you can see examples in [`examples`](https://github.com/tech-sketch/SeqAL/tree/main/examples) folders.
