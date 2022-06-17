# Tutorial 3: Training Parameter

This tutorial shows how so set the training parameter.

Below is the simple example to initialize the `ActiveLearner`.

```python
from seqal.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings
from seqal.active_learner import ActiveLearner
from seqal.samplers import LeastConfidenceSampler

## 1 Load data
columns = {0: "text", 3: "ner"}
data_folder = "./datasets/conll"
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
tagger_params["use_rnn"] = False

trainer_params = {}
trainer_params["max_epochs"] = 10
trainer_params["mini_batch_size"] = 32
trainer_params["learning_rate"] = 0.1
trainer_params["patience"] = 5

sampler = LeastConfidenceSampler()
learner = ActiveLearner(corpus, sampler, tagger_params, trainer_params)
```

We can see that there are two kinds of parameters we have to prepare, `tagger_params` and `trainer_params`. 

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

- `tag_type`: what kind of tag we want to predict, like 'ner', 'pos' and so on.
- `hidden_szie`: number of hidden states in RNN
- `embeddings`: word embedding used in tagger. Make sure the dataset and emebddings are the same language.

### CPU model (CRF)

If we want to speed up the training cycle, we can just use the CRF model by add below setup.

```python
tagger_params["use_rnn"] = False
```

## Trainer parameters

```python
trainer_params = {}
trainer_params["max_epochs"] = 10
trainer_params["mini_batch_size"] = 32
trainer_params["learning_rate"] = 0.1
trainer_params["patience"] = 5
```

The `trainer_params` contorls training process. parameters.

- `max_epochs`: the maximum number of epochs to train in each iteration. Usually we set this value smaller than 20 to decrease the training time.
- `mini_batch_size`: minimum size of data samples in each batch.
- `learning_rate`: initial learning rate.
- number of hidden states in RNN
- `patience`: the number of epochs with no improvement the Trainer waits

Because we use the flair model, you can find more detail about parameters in [flair.ModelTrainer.train](https://github.com/flairNLP/flair/blob/master/flair/trainers/trainer.py#L129)

