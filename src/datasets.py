import logging
import os
from bangla_stemmer.stemmer.stemmer import BanglaStemmer
import nltk
from nltk.stem.porter import *
import re
import json
nltk.download('indian')
nltk.download('punkt')


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore
STOP_WORDS_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../stopwords/stopwords-bn.json')


class FbDataset(object):

    def __init__(self, posts):
        """

               Parameters
               ----------
        posts : list of tuples as input with [0] as ID and [1] as text
            list of posts[post-id, post-text]

        """
        self.posts = posts

    def get_post(self, index):
        return self.posts[index]

    def is_bengali(self, word):
        for ch in word:
            if not '\u0980' <= ch <= '\u09FF':
                return False
        return True

    def tokenize(self, text, stopwords):
        result = []
        text = re.sub('\s+', ' ', text)
        text = re.sub("\'", "", text)
        text = re.sub("[’,\.!?]+", "", text)

        for token in nltk.word_tokenize(text):

            if token not in stopwords and self.is_bengali(token):
                result.append(token)

        return result

    def preprocess(self):
        """
        preprocess the dataset
        :return: list of preprocessed documents. Each preprocessed document is a list of tokens
        """
        with open(STOP_WORDS_FILE) as json_file:
            stopwords = json.load(json_file)

        tokenized_posts = [ self.tokenize(post[1], stopwords) for post in self.posts]
        print('tokenized_posts', tokenized_posts)

        result = []
        for token in tokenized_posts:
            if token not in stopwords:
                result.append(BanglaStemmer().stem(token))

        return result
