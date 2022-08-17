# Tutorial 3: Active Learner Setup

This tutorial shows how so setup Active Learner.

Below is the simple example to setup the `ActiveLearner`.

```python
from flair.embeddings import WordEmbeddings

from seqal.active_learner import ActiveLearner
from seqal.datasets import ColumnCorpus, ColumnDataset
from seqal.samplers import LeastConfidenceSampler
from seqal.utils import load_plain_text, add_tags

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

# 2. tagger params
tagger_params = {}
tagger_params["tag_type"] = "ner"
tagger_params["hidden_size"] = 256
embeddings = WordEmbeddings("glove")
tagger_params["embeddings"] = embeddings
tagger_params["use_rnn"] = False

# 3. trainer params
trainer_params = {}
trainer_params["max_epochs"] = 1
trainer_params["mini_batch_size"] = 32
trainer_params["learning_rate"] = 0.1
trainer_params["patience"] = 5

# 4. setup active learner
sampler = LeastConfidenceSampler()
learner = ActiveLearner(corpus, sampler, tagger_params, trainer_params)
```

To setup active learner, we have to provide `corpus`, `sampler`, `tagger_params`, and `trainer_params`. We introduce `sampler`, `tagger_params` and `trainer_params` below. Detail about `corpus` in [TUTORIAL_2_Prepare_Corpus](TUTORIAL_2_Prepare_Corpus.md)

## Sampler

The `seqal.sampler` provide below sampling method.

- Uncertainty based sampling method:
  - `LeastConfidenceSampler` (Least Confidence; LC)
  - `MaxNormLogProbSampler` (Maximum Normalized Log-Probability; MNLP)
- Diversity based sampling method:
  - `StringNGramSampler` (String N-Gram Similarity)
  - `DistributeSimilaritySampler` (Distribute Similarity; DS)
  - `ClusterSimilaritySampler` (Cluster Similarity; CS)
- Combine uncertainty and diversity sampling method:
  - `CombinedMultipleSampler`: LC+DS, LC+CS, MNLP+DS, MNLP+CS
- Other:
  - `RandomSampler` 

According to our [experiment](https://fintan.jp/page/4127/) (Japanese), there is some advice to choose a suitable sampling method.
- If you want to decrease training time, we recommend the uncertainty-based sampling methods
- If you want to increase the performance, we recommend the combined sampled methods.


## Tagger parameters

Because we use the flair model, you can find more detail about parameters in [flair.SequenceTagger](https://github.com/flairNLP/flair/blob/v0.10/flair/models/sequence_tagger_model.py#L89)

### GPU model (Bi-LSTM CRF)

```python
tagger_params = {}
tagger_params["tag_type"] = "ner" 
tagger_params["hidden_size"] = 256
embeddings = WordEmbeddings("glove")  
tagger_params["embeddings"] = embeddings
```

The `tagger_params` means model parameters. The default model is Bi-LSTM CRF.

- `tag_type`: what kind of tag we want to predict, like 'ner', 'pos', and so on.
- `hidden_szie`: number of hidden states in RNN
- `embeddings`: word embedding used in tagger. Make sure the dataset and embeddings are the same language.

### CPU model (CRF)

If we want to speed up the training cycle, we can just use the CRF model by add below setup.

```python
tagger_params["use_rnn"] = False
```

According to the [comparing result](performance.md) of the GPU model and CPU model, we highly recommend to use the CPU model. The performance of the GPU model is slightly better than the performance of the CPU model, but the CPU model's training speed is far faster than the GPU model's. And the price of a CPU machine is only about half the price of a GPU machine.


## Trainer parameters

```python
trainer_params = {}
trainer_params["max_epochs"] = 10
trainer_params["mini_batch_size"] = 32
trainer_params["learning_rate"] = 0.1
trainer_params["patience"] = 5
```

The `trainer_params` control the training process.

- `max_epochs`: the maximum number of epochs to train in each iteration. Usually, we set this value smaller than 20 to decrease the training time.
- `mini_batch_size`: minimum size of data samples in each batch.
- `learning_rate`: initial learning rate.
- `patience`: the number of epochs with no improvement the Trainer waits.

Because we use the flair model, you can find more detail about parameters in [flair.ModelTrainer.train](https://github.com/flairNLP/flair/blob/master/flair/trainers/trainer.py#L129)
