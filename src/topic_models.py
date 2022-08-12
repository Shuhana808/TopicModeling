from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import logging
import pandas as pd

from datasets import FbDataset

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pickle
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


class GensimLDA:
    def fit(self, dataset, num_topics):
        """
        @dataset is our documents. List of string.
        @num_topics is the amount of topics to be specified in the document

        """
        preprocessed_data = dataset.preprocess()
        bigram = gensim.models.Phrases(preprocessed_data, min_count=5, threshold=100)  # higher threshold fewer phrases.
        # trigram = gensim.models.Phrases(bigram[preprocessed_data], threshold=100)

        bigram = gensim.models.Phrases(preprocessed_data, min_count=5, threshold=100)  # higher threshold fewer phrases.
        # trigram = gensim.models.Phrases(bigram[preprocessed_data], threshold=100)

        # Faster way to get a sentence clubbed as a trigram/bigram
        self.bigram_mod = gensim.models.phrases.Phraser(bigram)
        # trigram_mod = gensim.models.phrases.Phraser(trigram)

        data_words_bigrams = [self.bigram_mod[doc] for doc in preprocessed_data]

        self.id2word = corpora.Dictionary(data_words_bigrams)

        self.id2word.filter_extremes(no_below=10, no_above=0.1)

        corpus = [self.id2word.doc2bow(text) for text in data_words_bigrams]

        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                         id2word=self.id2word,
                                                         num_topics=num_topics,
                                                         random_state=100,
                                                         update_every=1,
                                                         chunksize=100,
                                                         passes=10,
                                                         alpha='auto',
                                                         per_word_topics=True)

        pprint(self.lda_model.print_topics())
        print('Perplexity: ', self.lda_model.log_perplexity(corpus))
        coherence_model_lda = CoherenceModel(model=self.lda_model, texts=data_words_bigrams, dictionary=self.id2word,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('Coherence Score: ', coherence_lda)

    def update_model(self, tokenized_documnent):
        """
        @tokenized_document = tokenized document to be included in the
        LDA model

        """

        bow_new_article = [self.id2word.doc2bow(doc) for doc in tokenized_documnent]
        self.lda_model.update(bow_new_article)
        print('Updated list of topic words :')
        print('\n')
        pprint(self.lda_model.print_topics())

    def predict_topics(self, tokenized_documents):
        """
        @tokenized_documents : list of tokenized documents to be predicted by the LDA model.

        """

        data_words_bigrams = [self.bigram_mod[doc] for doc in tokenized_documents]
        corpus = [self.id2word.doc2bow(doc) for doc in data_words_bigrams]

        result = []
        for i, row in enumerate(self.lda_model[corpus]):

            row = sorted(row[0], key=lambda x: x[1], reverse=True)
            # Get the Dominant topic for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    result.append(int(topic_num))
                else:
                    break

        return result

    def format_topics_sentences(self, tokenized_articles, actual_articles):

        topics_df = pd.DataFrame()
        data_words_bigrams = [self.bigram_mod[doc] for doc in tokenized_articles]
        corpus = [self.id2word.doc2bow(doc) for doc in data_words_bigrams]

        # Get main topic in each document
        for i, row in enumerate(self.lda_model[corpus]):

            row = sorted(row[0], key=lambda x: x[1], reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = self.lda_model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    topics_df = topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(actual_articles)
        topics_df = pd.concat([topics_df, contents], axis=1)
        return topics_df

    def save(self, filename):
        d = {'tf': self.id2word, 'classifier': self.lda_model, 'bigram': self.bigram_mod}
        with open(filename, 'wb') as f:
            pickle.dump(d, f)
            f.close

    def load_model(self, filename):
        with open(filename, "rb") as f:
            d = pickle.load(f)
            self.id2word = d['tf']
            self.lda_model = d['classifier']
            self.bigram_mod = d['bigram']


