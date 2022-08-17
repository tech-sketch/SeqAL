# Tutorial 10: Performance Recorder

This tutorial shows how to use performance recorder.

```python
from flair.embeddings import WordEmbeddings

from seqal.active_learner import ActiveLearner
from seqal.datasets import ColumnCorpus, ColumnDataset
from seqal.samplers import LeastConfidenceSampler
from seqal.utils import load_plain_text, add_tags, count_tokens
from seqal.performance_recorder import PerformanceRecorder

# 1~6

# 7. query setup
query_percent = 0.02
token_based = True
total_tokens = count_tokens(corpus.train.sentences) + count_tokens(data_pool.sentences)
query_number = tokens_each_iteration = int(total_tokens * query_percent)

# performance recorder setup
performance_recorder = PerformanceRecorder()
accumulate_data = 0

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

    # 12. performance recorder get result
    result = learner.trained_tagger.evaluate(corpus.test, gold_label_type="ner")
    accumulate_data += query_percent
    performance_recorder.get_result(accumulate_data, result)
    iteration_performance = performance_recorder.performance_list[i]
    
    print(iteration_performance.data)
    print(iteration_performance.precision)
    print(iteration_performance.recall)
    print(iteration_performance.accuracy)
    print(iteration_performance.micro_f1)
    print(iteration_performance.macro_f1)
    print(iteration_performance.weighted_f1)
    print(iteration_performance.samples_f1)
    print(iteration_performance.label_scores)

performance_recorder.save("lc_performance.txt")
performance_recorder.plot(metric="micro_f1", sampling_method="lc", save_path="lc_performance.jpg")
```

As the above shows, we use `performance_recorder` to record the evaluation result of all iterations to the `performance_list` property. For each `iteration_performance`, we could get the scores by accessing properties. Finally, we can save the performance by calling `performance_recorder.save()` and plot the graph by calling `performance_recorder.plot()`. The `metric` specify the score we want to draw, the `sampling_metod` will show up on the graph legend, and the `save_path` will save the graph to the image.
