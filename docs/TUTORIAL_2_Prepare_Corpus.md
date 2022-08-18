# Tutorial 2: Prepare Corpus

This tutorial shows how to prepare corpus.

```python
from seqal.datasets import ColumnCorpus

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
```

## Data format

Flair support Flair supports the [BIO schema and the BIOES schema](https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)). So we need our data to follow the BIO schema and BIOES schema.

If you want to change to BIO shema or BIOES shema, we provide the below methods.

```python
from seqal import utils

bilou_tags = ["B-X", "I-X", "L-X", "U-X", "O"]
bioes_tags = utils.bilou2bio(bilou_tags)
bio_tags = utils.bilou2bio(bilou_tags)
bio_tags = utils.bilou2bio(bioes_tags)
bioes_tags = utils.bio2bioes(bio_tags)
```

## Spaced Language

The spaced language means a sentence can split tokens by space. For example, `Tokyo is a city`.

An example with BIO format:

```
Alex I-PER
is O
going O
to O
Los I-LOC
Angeles I-LOC
in O
California I-LOC
```

An example with BIOES format:

```
Alex S-PER
is O
going O
with O
Marty B-PER
A. I-PER
Rick E-PER
to O
Los B-LOC
Angeles E-LOC
```

## Non-spaced Language

The non-spaced language means a sentence can not be split by space, for example, `東京は都市です`.

Usually, one character with a label.

```
東 B-LOC
京 I-LOC
は O
都 O
市 O
で O
す O
```

But this format cannot be trained by flair. So we have to tokenize the sentence and merge the tags like below.

An example with BIO format:

```
東京 B-LOC
は O
都市 O
です O
```

An example with BIOES format:

```
東京 S-LOC
は O
都市 O
です O
```

## Corpus Usage

We can access different dataset by below commands.


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

The `seqal.datasets.ColumnCorpus` inherit from `flair.data.Corpus`. For the detail usage of `ColumnCorpus`, we recommend the flair tutorial about [Corpus](https://github.com/flairNLP/flair/blob/v0.10/resources/docs/TUTORIAL_6_CORPUS.md), 