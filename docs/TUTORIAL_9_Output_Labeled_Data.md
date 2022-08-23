# Tutorial 9: Output Labeled Data

This tutorial shows how to output labeled data.

After annotation we need to get the labeled data. We can output it on CoNLL format or JSON format.


```python
from flair.embeddings import WordEmbeddings

from seqal.active_learner import ActiveLearner
from seqal.datasets import ColumnCorpus, ColumnDataset
from seqal.samplers import LeastConfidenceSampler
from seqal.utils import load_plain_text, add_tags, output_labeled_data

# 1~7

# 8. iteration
for i in range(iterations):
    # 9. query unlabeled sentences
    queried_samples, unlabeled_sentences = learner.query(
        unlabeled_sentences, query_number, token_based=token_based, research_mode=False
    )

    # 10. annotate data
    annotated_data = human_annotate(queried_samples)

    # 11. retrain model with newly added queried_samples
    queried_samples = add_tags(annotated_data)
    learner.teach(queried_samples, dir_path=f"output/retrain_{i}")

# 12. Output labeled data
output_labeled_data(corpus.train.sentences, file_path="labeled_data.txt", file_format="conll", tag_type='ner')
```

We should know that all newly labeled data are added to the training dataset, so we just need to output the training dataset. 

The `file_path` is the path to save the file, `file_format` is the output format, and the `tag_type` is the tag type we want to output.

If we want to output the JSON file, we change the use below code.


```python
output_labeled_data(corpus.train.sentences, file_path="labeled_data.json", file_format="json", tag_type='ner')
```
