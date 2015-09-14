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


def extract_pos(x):
    tokens = nltk.word_tokenize(x)
    res = ['_'.join(p) for p in nltk.pos_tag(tokens)]
    # print res
    return res


class Model(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.cls = None
        self.classes_ = None
        self.vectorizer_params = [
            {'analyzer': "word", 'ngram_range': (2, 3), 'binary': False, 'max_features': 2000},
            {'analyzer': "char", 'ngram_range': (2, 3), 'binary': False, 'max_features': 2000, 'min_df': 0},
        ]

    def vectorize(self, X):
        vectorizers = [TfidfVectorizer() for _ in self.vectorizer_params]
        for (params, v) in zip(self.vectorizer_params, vectorizers):
            v.set_params(**params)

        vectorizers = [(str(i), v) for i, v in enumerate(vectorizers)]
        vectorizer = FeatureUnion(vectorizers)
        matrix = vectorizer.fit_transform(X)
        XX = matrix.toarray()

        for i, v in enumerate(vectorizers):
            self.vectorizer_params[i]['vocabulary'] = v[1].vocabulary_

        return XX

    def fit(self, X, y):
        print 'training model..'

        XX = self.vectorize(X)
        self.cls = LogisticRegression()
        self.cls.fit(XX, y)
        self.classes_ = self.cls.classes_

    def predict(self, X):
        XX = self.vectorize(X)
        return self.cls.predict(XX)

    def predict_proba(self, X):
        XX = self.vectorize(X)
        return self.cls.predict_proba(XX)
