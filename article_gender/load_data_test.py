import pandas as pd
import torch

from argparse import Namespace
from modules.vocabulary import Vocabulary
from modules.vectorizer import ArticleVectorizer
from modules.article_dataset import ArticleDataset
from modules.utils import generate_batches

args = Namespace(
    # Data and Path information
    article_csv="data/articles_with_splits.csv",
    vectorizer_file="vectorizer.json",
    model_state_file="model.pth",
    save_dir="model_storage",
    # Model hyper parameters
    hidden_dim=100,
    num_channels=256,
    # Training hyper parameters
    seed=1337,
    learning_rate=0.001,
    batch_size=128,
    num_epochs=100,
    early_stopping_criteria=5,
    dropout_p=0.1,
    # Runtime options
    cuda=False,
    reload_from_files=False,
    expand_filepaths_to_save_dir=True,
    catch_keyboard_interrupt=True)

dataset = ArticleDataset.load_dataset_and_make_vectorizer(args.article_csv)
dataset.save_vectorizer(args.vectorizer_file)

vectorizer = dataset.get_vectorizer()
dataset.set_split('train')
args.device = torch.device("cuda" if args.cuda else "cpu")
batch_generator = generate_batches(
    dataset, batch_size=args.batch_size, device=args.device)
for batch in batch_generator:
    print(batch)
