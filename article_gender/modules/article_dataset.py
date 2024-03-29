import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from modules.vectorizer import ArticleVectorizer


class ArticleDataset(Dataset):
    def __init__(self, article_df, vectorizer):
        """
        Args:
            name_df (pandas.DataFrame): the dataset
            vectorizer (SurnameVectorizer): vectorizer instatiated from dataset
        """
        self.article_df = article_df
        self._vectorizer = vectorizer
        self.train_df = self.article_df[self.article_df.split == 'train']
        self.train_size = len(self.train_df)

        self.val_df = self.article_df[self.article_df.split == 'val']
        self.validation_size = len(self.val_df)

        self.test_df = self.article_df[self.article_df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {
            'train': (self.train_df, self.train_size),
            'val': (self.val_df, self.validation_size),
            'test': (self.test_df, self.test_size)
        }

        self.set_split('train')

        # Class weights
        class_counts = article_df.fem.value_counts().to_dict()

        def sort_key(item):
            return self._vectorizer.article_vocab.lookup_token(item[0])

        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(
            frequencies, dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, article_csv):
        """Load dataset and make a new vectorizer from scratch

        Args:
            surname_csv (str): location of the dataset
        Returns:
            an instance of ArticleDataset
        """
        article_df = pd.read_csv(article_csv)

        train_article_df = article_df[article_df.split == 'train']
        return cls(article_df,
                   ArticleVectorizer.from_dataframe(train_article_df))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, article_csv,
                                         vectorizer_filepath):
        """Load dataset and the corresponding vectorizer.
        Used in the case in the vectorizer has been cached for re-use

        Args:
            surname_csv (str): location of the dataset
            vectorizer_filepath (str): location of the saved vectorizer
        Returns:
            an instance of SurnameDataset
        """
        article_df = pd.read_csv(article_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(article_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """a static method for loading the vectorizer from file

        Args:
            vectorizer_filepath (str):
            the location of the serialized vectorizer
        Returns:
            an instance of SurnameDataset
        """
        with open(vectorizer_filepath) as fp:
            return ArticleVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        """saves the vectorizer to disk using json

        Args:
            vectorizer_filepath (str): the location to save the vectorizer
        """
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer

    def set_split(self, split="train"):
        """ selects the splits in the dataset
        using a column in the dataframe """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets

        Args:
            index (int): the index to the data point
        Returns:
            a dictionary holding the data point's
             features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        article_matrix = \
            self._vectorizer.vectorize(row.article_content)

        fem_index = \
            row.fem

        return {
            'x_article': article_matrix,
            'y_fem': fem_index
        }

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset

        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size


def generate_batches(dataset,
                     batch_size,
                     shuffle=True,
                     drop_last=True,
                     device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict
