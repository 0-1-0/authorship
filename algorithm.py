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
import nltk.data, nltk.tag

tagger = nltk.data.load(nltk.tag._POS_TAGGER)


def extract_pos(x):
    tokens = nltk.word_tokenize(x)
    res = [p[1] for p in tagger.tag(tokens)]
    # print res
    return res

table = string.maketrans("", "")


class Model(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.cls = None
        self.classes_ = None
        self.sel = None
        self.vectorizer_params = [
            {'analyzer': "word", 'ngram_range': (2, 3), 'binary': False, 'max_features': 2000},
            {'analyzer': "char", 'ngram_range': (2, 3), 'binary': False, 'max_features': 2000, 'min_df': 0},
            {'analyzer': extract_pos, 'ngram_range': (1, 3), 'binary': False, 'max_features': 2000, 'min_df': 0}
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
        return self.cls.predict_proba(XX)
