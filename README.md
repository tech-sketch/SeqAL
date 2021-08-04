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

Sequence labeling active learning framework for Python

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

```python
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from seqal.active_learner import ActiveLearner
from seqal.query_strategies import mnlp_sampling

# 1. get the corpus
corpus = Corpus(train="train.data", dev="dev.data", test="dev.data")

# 2. what tag do we want to predict?
tag_type = "ner"

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# 4. initialize embeddings
embedding_types = [WordEmbeddings("glove")]
embeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
tagger = SequenceTagger(hidden_size=256, embeddings=embeddings, tag_dictionary=tag_dictionary, tag_type=tag_type)

# 6. initialize learner
params = {}
params["max_epochs"] = 1
params["learning_rate"] = 0.1
params["train_with_dev"] = True

learner = ActiveLearner(tagger, mnlp_sampling, corpus, **params)

# 7. initial training
learner.fit(save_path="output/init_train")

# 8. query for labels
sents, query_samples = learner.query(sents, percent_count)

# ...obtaining new labels for "query_samples" from the Oracle...

# 9. retrain model with new labeled data
learner.teach(query_samples, save_path=f"output/retrain")
```

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
