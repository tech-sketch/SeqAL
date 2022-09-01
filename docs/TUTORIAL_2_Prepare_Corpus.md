# Prepare Corpus

This tutorial shows how to prepare corpus.

We can load the custom dataset by below script.

```python
from seqal.datasets import ColumnCorpus

# 1. get the corpus
columns = {0: "text", 1: "ner"}
data_folder = "./data/sample_bio"
corpus = ColumnCorpus(
    data_folder,
    columns,
    train_file="train_seed.txt",
    dev_file="valid.txt",
    test_file="test.txt",
)
```

If we want to use the existing corpus in [flair datasets](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_6_CORPUS.md), we could use below script. 

Notice that we have to download the data first. For example, if we want to load CoNLL-03 corpus, we download CoNLL-03 from [homepage](https://www.clips.uantwerpen.be/conll2003/ner/) and put the `eng.testa`, `eng.testb`, `eng.train` to `data/conll_03` floder.


```python
import flair.datasets

floder_path = "../data/conll_03"
corpus = flair.datasets.CONLL_03(floder_path)
```


## Data format

Flair support Flair supports the [BIO schema and the BIOES schema](https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)). So we need our data to follow the BIO schema or BIOES schema.

If you want to change to BIO shema or BIOES shema, we provide the below methods.

```python
from seqal import utils

bilou_tags = ["B-X", "I-X", "L-X", "U-X", "O"]
bioes_tags = utils.bilou2bio(bilou_tags)
bioes_tags = utils.bio2bioes(bio_tags)
bio_tags = utils.bilou2bio(bilou_tags)
bio_tags = utils.bioes2bio(bioes_tags)
```

## Spaced Language


The spaced language means a sentence can split tokens by space, like English ( `"Tokyo is a city"`) and Spanish (`"Tokio es una ciudad"`).

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

The non-spaced language means a sentence can not be split by space, like Japanese ( `"東京は都市です"`) and Chinese (`"东京是都市"`). 

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

We can get labels from one sentence.

```python
sentence = corpus.train[19]
for entity in sentence.get_spans('ner'):
    print(entity.text, entity.tag)
```

This prints:

```
Germany LOC
Britain LOC
```

We also can get label of each token.

```python
for token in sentence:
    tag = token.get_tag('ner')
    print(token.text, tag.value, tag.score)
```

This prints:

```
Germany B-LOC 1.0
imported O 1.0
47,600 O 1.0
sheep O 1.0
from O 1.0
Britain B-LOC 1.0
last O 1.0
year O 1.0
, O 1.0
nearly O 1.0
half O 1.0
of O 1.0
total O 1.0
imports O 1.0
. O 1.0
```

The score is confidence score. Because we read the entities' labels from dataset, it assumes that the labels are glod annotations. The confidence score of glod annotaiotns is 1.0. If a sentence is predicted by model, the condidence score should be lower than 1.0. 

Below is an example that use a pre-trained model to predict a sentence.

```python
from flair.models import SequenceTagger

tagger = SequenceTagger.load('ner')

sentence = Sentence('George Washington went to Washington.')
tagger.predict(sentence)

for token in sentence:
    tag = token.get_tag('ner')
    print(token.text, tag.value, tag.score)
```

It prints:
```
George B-PER 0.9978131055831909
Washington E-PER 0.9999594688415527
went O 0.999995231628418
to O 0.9999998807907104
Washington S-LOC 0.9942096471786499
. O 0.99989914894104
```

The `seqal.datasets.ColumnCorpus` inherit from `flair.data.Corpus`. We recommend the flair tutorials for more detail. 

Related tutorials:
- [Tutorial 1: Basics](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_1_BASICS.md)
- [Tutorial 2: Tagging your Text](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md)
- [Tutorial 6: Loading a Dataset](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_6_CORPUS.md)
