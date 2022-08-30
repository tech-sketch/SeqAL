# Installation

## Installation from PyPI

To install SeqAL with pip, run the following command:

```bash
pip install seqal
```

## Construct Envirement Locally

If you want to make a PR or implement something locally, you can follow bellow instruction to construct the development envirement locally. It will install the latest SeqAL from the main branch.

We use conda as the envirement management tool, so install it first. Here is the [installation tutorial for conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#installing-conda-on-a-system-that-has-other-python-installations-or-packages). We recommend the install [Miniconda](https://docs.conda.io/en/latest/miniconda.html#macos-installers) due to it's small size.


First we create a environment `seqal` based on the `environment.yml` file.

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

This command will install all dependencies to `seqal` environment.

You can make development locally now.

If you want to delete the local envirement, run below command.
```
conda remove --name seqal --all
```
