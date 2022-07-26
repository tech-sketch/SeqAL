# Tutorial 8: Stopper

This tutorial shows how to use stoppers.

## Stop by Budget

Annotation costs a lot of money. Usually, we will have a budget. If we run out of budget, the active learning cycle will stop.

Below is a demo to show how to use `BudgetStopper`.

```python
from flair.embeddings import WordEmbeddings

from seqal.active_learner import ActiveLearner
from seqal.datasets import ColumnCorpus, ColumnDataset
from seqal.samplers import LeastConfidenceSampler
from seqal.utils import load_plain_text, add_tags, count_tokens
from seqal.stoppers import BudgetStopper

# 1~7

# Stopper setup
stopper = BudgetStopper(goal=200, unit_price=0.02)

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

    # 12. stop iteration early
    unit_count = count_tokens(corpus.train.sentences)
    if stopper.stop(unit_count):
        break
```

The `BudgetStopper(goal=200, unit_price=0.02)` initialize the budget stopper. The `goal` means how much money we have, here we say 200\$. The `unit_price` means annotation cost for each unit, here we say 0.02\$/unit. A unit could be a sentence or a token. Usually, it is a token.

## Stop by Metric

Another motivation to stop active learning cycle is model's performance is beyond our goal.

```python
from flair.embeddings import WordEmbeddings

from seqal.active_learner import ActiveLearner
from seqal.datasets import ColumnCorpus, ColumnDataset
from seqal.samplers import LeastConfidenceSampler
from seqal.utils import load_plain_text, add_tags, count_tokens
from seqal.stoppers import MetricStopper

# 1~6

# 7. query setup
query_percent = 0.02
token_based = True
total_tokens = count_tokens(corpus.train.sentences) + count_tokens(data_pool.sentences)
query_number = tokens_each_iteration = int(total_tokens * query_percent)

# performance recorder setup
performance_recorder = PerformanceRecorder()
accumulate_data = 0

# Stopper setup
stopper = MetricStopper(goal=0.9)

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

    # 12. stop iteration early
    result = learner.trained_tagger.evaluate(corpus.test, gold_label_type="ner")
    accumulate_data += query_percent
    performance_recorder.get_result(accumulate_data, result)
    iteration_performance = performance_recorder.performance_list[i]

    if stopper.stop(iteration_performance.micro_f1):
        break
```

The `MetricStopper(goal=0.9)` initialize the budget stopper. The `goal` means the f1 score we want to achieve. 

The `learner.trained_tagger.evaluate(corpus.test, gold_label_type="ner")` evaluate on test dataset and return the evaluation result.

We use `performance_recorder` to parse the evluation result. 

The `stopper.stop(iteration_performance.micro_f1)` compare the goal and the evaluation result on micro f1 score. We can also compare other metrics like macro f1, accuracy, etc. 