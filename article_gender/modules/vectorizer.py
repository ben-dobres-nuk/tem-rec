import numpy as np
import string
from modules.vocabulary import Vocabulary
from collections import Counter


class ArticleVectorizer(object):
    """ The Vectorizer which coordinates
     the Vocabularies and puts them to use"""

    def __init__(self, article_vocab):
        """
        Args:
            review_vocab (Vocabulary): maps words to integers
            rating_vocab (Vocabulary): maps class labels to integers
        """
        self.article_vocab = article_vocab

    def vectorize(self, article):
        """Create a collapsed one-hit vector for the article

        Args:
            article (str): the article
        Returns            one_hot (np.ndarray): th:
e collapsed one-hot encoding
        """
        one_hot = np.zeros(len(self.article_vocab), dtype=np.float32)

        for token in article.split(" "):
            if token not in string.punctuation:
                one_hot[self.article_vocab.lookup_token(token)] = 1

        return one_hot

    @classmethod
    def from_dataframe(cls, article_df, cutoff=25):
        """Instantiate the vectorizer from the dataset dataframe

        Args:
            review_df (pandas.DataFrame): the review dataset
            cutoff (int): the parameter for frequency-based filtering
        Returns:
            an instance of the ReviewVectorizer
        """
        article_vocab = Vocabulary(add_unk=True)

        # Add top words if count > provided count
        word_counts = Counter()
        for article in article_df.article_content:
            for word in article.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1

        for word, count in word_counts.items():
            if count > cutoff:
                article_vocab.add_token(word)

        return cls(article_vocab)

    @classmethod
    def from_serializable(cls, contents):
        """Instantiate a ReviewVectorizer from a serializable dictionary

        Args:
            contents (dict): the serializable dictionary
        Returns:
            an instance of the ReviewVectorizer class
        """
        review_vocab = Vocabulary.from_serializable(contents['review_vocab'])
        rating_vocab = Vocabulary.from_serializable(contents['rating_vocab'])

        return cls(review_vocab=review_vocab, rating_vocab=rating_vocab)

    def to_serializable(self):
        """Create the serializable dictionary for caching

        Returns:
            contents (dict): the serializable dictionary
        """
        return {
            'article_vocab': self.article_vocab.to_serializable()

        }
