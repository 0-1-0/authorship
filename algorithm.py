from __future__ import unicode_literals
import pickle
from utils import dependencies, load_train_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import numpy as np
import nltk
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import string
from spacy import parts_of_speech
from spacy.en import English
from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")
nlp = English()


def extract_pos(x):
    doc = nlp(unicode(x))
    tokens = [token.pos for token in doc]
    # print res
    return tokens

table = string.maketrans("", "")


def extract_pun(x):
    return [c for c in x if c in string.punctuation]


def named_entity(x):
    doc = nlp(unicode(x))
    return [ent.label_ for ent in doc.ents]


def word_per_sent(x):
    doc = nlp(unicode(x))
    return ([len([w for w in s]) for s in doc.sents])


def extract_stop_words(x):
    return [w for w in x.split() if x in stopwords.words("english")]


class Model(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.cls = None
        self.classes_ = None
        self.sel = None
        self.vectorizer_params = [
            {'analyzer': "word", 'ngram_range': (2, 3), 'binary': False, 'max_features': 2000}, # word frequencies
            {'analyzer': "char", 'ngram_range': (2, 3), 'binary': False, 'max_features': 2000, 'min_df': 0}, # character freqs.
            {'analyzer': extract_pos, 'ngram_range': (2, 4), 'binary': False, 'max_features': 2000, 'min_df': 0}, # POS freqs.
            # {'analyzer': extract_pun, 'ngram_range': (1, 1), 'binary': False, 'max_features': 2000, 'min_df': 0}, # Punct. freqs.
            {'analyzer': named_entity, 'ngram_range': (1, 1), 'binary': False, 'max_features': 200, 'min_df': 0}, # NE. freqs.
            {'analyzer': word_per_sent, 'ngram_range': (1, 1), 'binary': False, 'max_features': 200, 'min_df': 0}, # WPS. freqs.
            # {'analyzer': extract_stop_words, 'ngram_range': (1, 2), 'binary': False, 'max_features': 2000, 'min_df': 0} # stop words. freqs.
        ]

    def vectorize(self, X):
        # remove punctuation
        X1 = []
        for x in X:
            X1.append(x.translate(table, string.punctuation))

        vectorizers = [TfidfVectorizer() for _ in self.vectorizer_params]
        for (params, v) in zip(self.vectorizer_params, vectorizers):
            v.set_params(**params)

        vectorizers = [(str(i), v) for i, v in enumerate(vectorizers)]
        vectorizer = FeatureUnion(vectorizers)
        matrix = vectorizer.fit_transform(X1)
        XX = matrix.toarray()

        for i, v in enumerate(vectorizers):
            self.vectorizer_params[i]['vocabulary'] = v[1].vocabulary_

        return XX

    def fit(self, X, y):
        print 'training model..'

        XX = self.vectorize(X)
        self.sel = SelectKBest(chi2, k=4000)
        XX = self.sel.fit_transform(XX, y)

        self.cls = LogisticRegression()
        self.cls.fit(XX, y)
        self.classes_ = self.cls.classes_

    def predict(self, X):
        XX = self.vectorize(X)
        XX = self.sel.transform(XX)
        return self.cls.predict(XX)

    def predict_proba(self, X):
        XX = self.vectorize(X)
        XX = self.sel.transform(XX)
        return self.cls.predict_proba(XX)
