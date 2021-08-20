# SeqAL

<!-- <p align="center">
  <a href="https://github.com/BrambleXu/seqal/actions?query=workflow%3ACI">
    <img src="https://img.shields.io/github/workflow/status/BrambleXu/seqal/CI/main?label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://seqal.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/seqal.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/BrambleXu/seqal">
    <img src="https://img.shields.io/codecov/c/github/BrambleXu/seqal.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p> -->
<p align="center">
  <a href="https://github.com/BrambleXu/seqal/actions?query=workflow%3ACI">
    <img src="https://img.shields.io/github/workflow/status/BrambleXu/seqal/CI/main?label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://python-poetry.org/">
    <img src="https://img.shields.io/badge/packaging-poetry-299bd7?style=flat-square&logo=data:image/png" alt="Poetry">
  </a>
  <a href="https://github.com/ambv/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square" alt="black">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/seqal/">
    <img src="https://img.shields.io/pypi/v/seqal.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/seqal.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/seqal.svg?style=flat-square" alt="License">
</p>

SeqAL is a sequence labeling active learning framework based on Flair.

## Installation

Install this via pip (or your favourite package manager):

`pip install seqal`


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

Each line contains four fields: the word, its partof-speech tag and its named entity tag. Words tagged with O are outside of named entities. 

### Examples

Because SeqAL is based on flair, we heavily recommend to read the [tutorial](https://github.com/flairNLP/flair/blob/5c4231b30865bf4426ba8076eb91492d329c8a9b/resources/docs/TUTORIAL_1_BASICS.md) of flair first. 

```python
import json

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

This part is where we set the parameters for sequence tagger and trainer. The above setup can conver most of situations. If you want to add more paramters, I recommend to the read [SequenceTagger](https://github.com/flairNLP/flair/blob/master/flair/models/sequence_tagger_model.py#L68) and [ModelTrainer](https://github.com/flairNLP/flair/blob/master/flair/trainers/trainer.py#L42) in flair.


```python
# 5. initial training
learner.fit(save_path="output/init_train")
```

The initial training will be trained on the seed data.

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
query_number = 1
sents, query_samples = learner.query(sents, query_number, token_based=True)
```

We can query samples from data pool by the `learner.query()` method. `query_number` means how many sentence we want to query. But if we set `token_based=True`, the `query_number` means how many tokens we want to query. For the sequence labeling task, we usually set `token_based=True`.

`query_samples` is a list that contains queried sentences (the Sentence class in flair). `sents` contains the rest of unqueried sentences.

```
In [1]: query_samples[0].to_plain_string()
Out[1]: 'I love Berlin .'
```

We can get the text by calling `to_plain_strin()` method and put it into the interface for human annotation.


```python
# 8. obtaining labels for "query_samples" by the human
query_labels = [
      {
        "text": "I love Berlin .",
        "labels": [{"start_pos": 7, "text": "Berlin", "label": "S-LOC"}]
      },
      {
        "text": "This book is great.",
        "labels": []
      }
]


annotated_sents = assign_labels(query_labels)
```
`query_labels` is the label information of a sentence after annotation by human. We use such information to create Flair Sentence class by calling `assign_labels()` method.

For more detail, see [Adding labels to sentences](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_1_BASICS.md#adding-labels-to-sentences)


```python
# 9. retrain model with new labeled data
learner.teach(annotated_sents, save_path=f"output/retrain")
```

Finally, we call `learner.teach()` to retrain the model. The `annotated_sents` will be added to `corpus.train` automatically.

If you want to run the workflow in a loop, you can take a look at the `examples` folders.


## Construct envirement locally

If you want to make a PR or implement something locally, you can follow bellow instruction to construct the development envirement locally.

First we create a environment "seqal" based on the `environment.yml` file.

We use conda as envirement management tool, so install it first.

```
conda env create -f environment.yml
```

Then we activate the environment.

```
conda activate seqal
```

Install poetry for dependency management.

```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

Add poetry path in your shell configure file (`bashrc`, `zshrc`, etc.)
```
export PATH="$HOME/.poetry/bin:$PATH"
```

Installing dependencies from `pyproject.toml`.

```
poetry install
```

You can make development locally now.

If you want to delete the local envirement, run below command.
```
conda remove --name seqal --all
```

## Performance

See [performance.md](./docs/source/performance.md) for detail.


## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Credits

- [Cookiecutter](https://github.com/audreyr/cookiecutter)
- [browniebroke/cookiecutter-pypackage](https://github.com/browniebroke/cookiecutter-pypackage)
- [flairNLP/flair](https://github.com/flairNLP/flair)
- [modal](https://github.com/modAL-python/modAL)
