# Tutorial 7: Annotated Data

This tutorial shows how to receive the labeled data.

## Annotated Data Based on Token

Below is the demo example.

```python
from flair.embeddings import WordEmbeddings

from seqal.active_learner import ActiveLearner
from seqal.datasets import ColumnCorpus, ColumnDataset
from seqal.samplers import LeastConfidenceSampler
from seqal.utils import load_plain_text, add_tags
from seqal.aligner import Aligner

# 1~7

# initialize Aligner
aligner = Aligner()

# 8. iteration
for i in range(iterations):
    # 9. query unlabeled sentences
    queried_samples, unlabeled_sentences = learner.query(
        unlabeled_sentences, query_number, token_based=token_based, research_mode=False
    )

    # 10. annotate data
    annotated_data = human_annotate(queried_samples)

    # 11. retrain model with newly added queried_samples
    queried_samples = aligner.add_tags_on_token(annotated_data)
    learner.teach(queried_samples, dir_path=f"output/retrain_{i}")
```

In the step 10, it receive the labeled data.

For token based annotated data, we can use the `aligner.add_tags_on_token()` to get a list of `flair.data.Sentence`.

The `annotated_data` should has below format.

Example of spaced language:

```json
[
    {
        "text": ['Tokyo', 'is', 'a', 'city'],
        "labels": ['B-LOC', 'O', 'O', 'O']
    }
]
```

Example of non-spaced language:
```json
    [
        {
            "text": ['ロンドン', 'は', '都市', 'です'],
            "labels": ['B-LOC', 'O', 'O', 'O']
        }
    ]
```

The detail of spaced language and non-spaced language can be found in [TUTORIAL_2_Prepare_Corpus](TUTORIAL_2_Prepare_Corpus.md).

## Annotated Data Based on Character

```python
# 1~7

# initialize Aligner
aligner = Aligner()
nlp = spacy.load("ja_core_news_sm")

# 8. iteration
for i in range(iterations):
    # 9. query unlabeled sentences
    queried_samples, unlabeled_sentences = learner.query(
        unlabeled_sentences, query_number, token_based=token_based, research_mode=False
    )

    # 10. annotate data
    annotated_data = human_annotate(queried_samples)

    # 11. retrain model with newly added queried_samples
    queried_samples = aligner.add_tags_on_char(annotated_data, spaced_language=False, spacy_model=nlp, input_schema="BIO", output_schema="BIO", tag_type="ner")

    # queried_samples = aligner.add_tags_on_char(annotated_data, spaced_language=True, input_schema="BIO", output_schema="BIO", tag_type="ner")

    learner.teach(queried_samples, dir_path=f"output/retrain_{i}")
```

For character based annotated data, we can use the `aligner.add_tags_on_char()` to get a list of `flair.data.Sentence`.

The parameters of `aligner.add_tags_on_char()`:
- `annotated_data`: A list of labeled data
- `spaced_language`: Input sentence is non-spaced language or not. If spaced_language is False, we have to provide a spacy model to tokenize the non-spaced language.
- `spacy_model`: Spacy language model
- `input_schema`: Input tag shema. Defaults to "BIO". Support "BIO", "BILOU", "BIOES"
- `output_schema`: Output tag shema. Defaults to "BIO". Support "BIO", "BIOES". Flair don't support "BILOU", so we don't output this schema.


The `annotated_data` example of spaced language:

```
[
    {
        "text": ['T', 'o', 'k', 'y', 'o', ' ', 'c', 'i', 't', 'y', '.'],
        "labels": ["B-LOC", "I-LOC", "I-LOC", "I-LOC", "E-LOC", 'O', 'O', 'O', 'O', 'O']
    },
]
```

The `annotated_data` example of non-spaced language:

```
[
    {
        "text": ['ロ', 'ン', 'ド', 'ン', 'は', '都', '市', 'で', 'す'],
        "labels": ["B-LOC", "I-LOC", "I-LOC", "E-LOC", "O", "O", "O", "O", "O"]
    }
]
```
