# Multiple Language Support

This tutorial shows how to use SeqAL for different language.

[Introduction](./TUTORIAL_1_Introduction.md) shows example on English. If you want to use SeqAL on other language, you need to meet the following requirements.

1. Prepare the data with BIO or BIOES format on the language you used. More detail can be found in [Prepare Corpus](./TUTORIAL_2_Prepare_Corpus.md)
2. Prepare the embedding on the same language. We can use [`TransformerWordEmbeddings`](https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/TRANSFORMER_EMBEDDINGS.md) to read different languages' embedding from [HuggingFace](https://huggingface.co/models?sort=downloads).
3. If the language of dataset is a kind of non-spaced language, we have to use spacy model to tokenize the dataset.

We introduce the processing workflow for spaced language and non-spaced language below.

## Spaced Language

Below is the simple example of active learning cycle on English.

```python
from seqal.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings
from seqal.active_learner import ActiveLearner
from seqal.samplers import LeastConfidenceSampler

## 1 Load data
columns = {0: "text", 1: "ner"}
data_folder = "./data/sample_bio"
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
data_folder = "./data/conll_deu"
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

## Non-spaced Language

### Prepare Corpus

First we provide the corpus with CoNLL format like below:

```
東京 B-LOC
は O
都市 O
です O
```

Then we load the corpus just like the spaced language.
```python
columns = {0: "text", 1: "ner"}
data_folder = "./data/sample_jp"
corpus = ColumnCorpus(
    data_folder,
    columns,
    train_file="train_seed.txt",
    dev_file="valid.txt",
    test_file="test.txt",
)
```

You can see the examples in `./data/sample_jp` directory.

### Prepare Embeddings

We provide the proper embedding for specified language. Below is a Japances example.

```python
from flair.embeddings import StackedEmbeddings, FlairEmbeddings

embedding_types = [
    FlairEmbeddings('ja-forward'),
    FlairEmbeddings('ja-backward'),
]
embeddings = StackedEmbeddings(embeddings=embedding_types)
```

### Prepare Data Pool


**Research Mode**

If we just want to simulate the active learning cycle (the research mode), we can should load the labeled data pool by `ColumnDataset` and set the `research_mode` as `True`.

```python
from seqal.datasets import ColumnDataset

columns = {0: "text", 1: "ner"}
pool_file = "./data/sample_jp/labeled_data_pool.txt"
data_pool = ColumnDataset(pool_file, columns)
unlabeled_sentences = data_pool.sentences

# ...

for i in range(iterations):
    queried_samples, unlabeled_sentences = learner.query(
        unlabeled_sentences, query_number, token_based=token_based, research_mode=True
    )
    learner.teach(queried_samples, dir_path=f"output/retrain_{i}")

```

**Annotation Mode**

If we are dealing with a real annotation project, we usually load the data from plain text.

```python
file_path = "./data/sample_jp/unlabeled_data_pool.txt"

unlabeled_sentences = []
with open(file_path, mode="r", encoding="utf-8") as f:
    for line in f:
        unlabeled_sentences.append(line)
```

Because the loaded sentence is non-spaced, we have to tokenize the sentence. For examples, we tokenize the `東京は都市です` to `["東京", "は", "都市", "です"]`. 

There are two ways to tokenize the sentences. The first method is using the `Transformer.to_subword()`. 

```python
import spacy
from seqal.transformer import Transformer

nlp = spacy.load("ja_core_news_sm")
tokenizer = Transformer(nlp)
unlabeled_sentences = [tokenizer.to_subword(sentence) for sentence in unlabeled_sentences]
```

The second method is directly using the spacy tokenizer.

```python
import spacy
from flair.data import Sentence
from flair.tokenization import SpacyTokenizer

tokenizer = SpacyTokenizer("ja_core_news_sm")
unlabeled_sentences = [Sentence(sentence, use_tokenizer=tokenizer) for sentence in sentences]
```

And do not forget set the `research_mode` as `False`.

```python
for i in range(iterations):
    queried_samples, unlabeled_sentences = learner.query(
        unlabeled_sentences, query_number, token_based=token_based, research_mode=False
    )
    learner.teach(queried_samples, dir_path=f"output/retrain_{i}")
```


### Run Active Learning Cycle

Executing below command will run the active learning cycle on non-spaced language. The user can see the script for more detail.

```
python examples/active_learning_cycle_research_mode_non_spaced_language.py
```