import argparse

from flair.embeddings import WordEmbeddings

from seqal import samplers
from seqal.active_learner import ActiveLearner
from seqal.datasets import ColumnCorpus, ColumnDataset

parser = argparse.ArgumentParser()


# Dataset
parser.add_argument("--text_column", help="which column is text", type=int, default=0)
parser.add_argument("--tag_column", help="which column is tag", type=int, default=1)
parser.add_argument(
    "--data_folder", help="data folder", type=str, default="./data/sample_bio"
)
parser.add_argument(
    "--train_file", help="training data file", type=str, default="train_seed.txt"
)
parser.add_argument(
    "--dev_file", help="development data file", type=str, default="dev.txt"
)
parser.add_argument("--test_file", help="test data file", type=str, default="test.txt")
parser.add_argument(
    "--pool_file", help="data pool file", type=str, default="labeled_data_pool.txt"
)


# Tagger Params
parser.add_argument("--tag_type", help="tag type", type=str, default="ner")
parser.add_argument("--hidden_size", help="hidden size", type=int, default=256)
parser.add_argument("--embeddings", help="embedding method", type=str, default="glove")
parser.add_argument(
    "--use_rnn", help="use run or not, if false, only use CRF", type=bool, default=False
)

# Trainer Params
parser.add_argument("--max_epochs", help="max epochs", type=int, default=1)
parser.add_argument("--mini_batch_size", help="mini batch size", type=int, default=32)
parser.add_argument("--learning_rate", help="learning rate", type=float, default=0.1)
parser.add_argument("--patience", help="patience to stop learning", type=int, default=5)

# Sampler
parser.add_argument(
    "--sampler",
    help="Sampling method - RandomSampler, LeastConfidenceSampler, MaxNormLogProbSampler, etc.",
    type=str,
    default="MaxNormLogProbSampler",
)

# Query setup
parser.add_argument("--query_number", help="query number", type=int, default=2)
parser.add_argument("--token_based", help="token based", type=bool, default=False)
parser.add_argument("--iterations", help="iterations", type=int, default=5)
parser.add_argument(
    "--research_mode", help="research (simulation) mode", type=bool, default=True
)


args = parser.parse_args()


# 1. get the corpus
columns = {args.text_column: "text", args.tag_column: args.tag_type}
data_folder = args.data_folder
corpus = ColumnCorpus(
    data_folder,
    columns,
    train_file=args.train_file,
    dev_file=args.dev_file,
    test_file=args.test_file,
)

# 2. tagger params
tagger_params = {}
tagger_params["tag_type"] = args.tag_type
tagger_params["hidden_size"] = args.hidden_size
embeddings = WordEmbeddings(args.embeddings)
tagger_params["embeddings"] = embeddings
tagger_params["use_rnn"] = args.use_rnn

# 3. trainer params
trainer_params = {}
trainer_params["max_epochs"] = args.max_epochs
trainer_params["mini_batch_size"] = args.mini_batch_size
trainer_params["learning_rate"] = args.learning_rate
trainer_params["patience"] = args.patience

# 4. setup active learner
sampler = samplers.__dict__[args.sampler]()
learner = ActiveLearner(corpus, sampler, tagger_params, trainer_params)

# 5. initialize active learner
learner.initialize(dir_path="output/init_train")

# 6. prepare data pool
pool_file = data_folder + "/" + args.pool_file
data_pool = ColumnDataset(pool_file, columns)
unlabeled_sentences = data_pool.sentences

# 7. query setup
query_number = args.query_number
token_based = args.token_based
iterations = args.iterations
research_mode = args.research_mode

# 8. iteration
for i in range(iterations):
    # 9. query unlabeled sentences
    queried_samples, unlabeled_sentences = learner.query(
        unlabeled_sentences,
        query_number,
        token_based=token_based,
        research_mode=research_mode,
    )

    # 10. retrain model, the queried_samples will be added to corpus.train
    learner.teach(queried_samples, dir_path=f"output/retrain_{i}")
