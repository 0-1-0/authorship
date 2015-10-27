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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import string
from sklearn.ensemble import RandomForestClassifier
from spacy import parts_of_speech
from spacy.en import English
from nltk.corpus import stopwords
from sklearn.feature_selection import RFECV, RFE
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold

# cachedStopWords = set(stopwords.words("english"))
cachedPunctuation = set(list(string.punctuation))
from spacy import language

nlp = English()

docCache = {}


# parsing is time-consuming operation, so we perform caching
def get_doc(x):
    if x not in docCache:
        content = x.decode('utf-8')
        docCache[x] = nlp(unicode(content))
    return docCache[x]


def extract_pos(x):
    doc = get_doc(x)
    tokens = [token.pos for token in doc]
    # print res
    return tokens


def extract_pun(x):
    doc = get_doc(x)
    return [t.lemma for t in doc if t.is_punct]


def named_entity(x):
    doc = get_doc(x)
    return [ent.label for ent in doc.ents]


def word_per_sent(x):
    doc = get_doc(x)
    return [len(s) for s in doc.sents]


def stop_word_per_sent(x):
    doc = get_doc(x)
    return [len([w for w in s if w.lemma_ in cachedStopWords]) for s in doc.sents]


def numbers_cnt(x):
    doc = get_doc(x)
    return [t.like_num for t in doc]


def capitals_cnt(x):
    doc = get_doc(x)
    return [t.is_title for t in doc]


def get_prefixes(x):
    doc = get_doc(x)
    return [w.prefix_ for w in doc]


def get_suffixes(x):
    doc = get_doc(x)
    return [w.suffix_ for w in doc]


def get_lemmas(x):
    doc = get_doc(x)
    return [w.lemma for w in doc]


def get_clusters(x):
    doc = get_doc(x)
    return [t.cluster for t in doc]


class Model(BaseEstimator, ClassifierMixin):
    def __init__(self, selection_method='chi2', classifier_type='logreg'):
        self.classes_ = None

        self.selection_method = selection_method
        self.classifier_type = classifier_type
        print self.selection_method, self.classifier_type

        self.vectorizer_params = [
            {'analyzer': "word", 'ngram_range': (1, 3), 'binary': False, 'max_features': 2000}, # word frequencies
            # {'analyzer': "char", 'ngram_range': (3, 3), 'binary': False, 'max_features': 2000, 'min_df': 0}, # character freqs.
            {'analyzer': extract_pos, 'ngram_range': (1, 4), 'binary': False, 'max_features': 1000, 'min_df': 0}, # POS freqs.
            {'analyzer': extract_pun, 'ngram_range': (1, 2), 'binary': False, 'max_features': 1000, 'min_df': 0}, # Punct. freqs.
            {'analyzer': named_entity, 'ngram_range': (1, 2), 'binary': False, 'max_features': 200, 'min_df': 0}, # NE. freqs.

            {'analyzer': numbers_cnt, 'ngram_range': (1, 1), 'binary': False, 'max_features': 100, 'min_df': 0}, # WPS. freqs.
            {'analyzer': capitals_cnt, 'ngram_range': (1, 1), 'binary': False, 'max_features': 100, 'min_df': 0}, # WPS. freqs.

            {'analyzer': get_prefixes, 'ngram_range': (1, 1), 'binary': False, 'max_features': 200, 'min_df': 0}, # WPS. freqs.
            {'analyzer': get_suffixes, 'ngram_range': (1, 1), 'binary': False, 'max_features': 200, 'min_df': 0}, # WPS. freqs.
            {'analyzer': get_lemmas, 'ngram_range': (1, 1), 'binary': False, 'max_features': 500, 'min_df': 0}, # WPS. freqs.
            {'analyzer': get_clusters, 'ngram_range': (1, 3), 'binary': False, 'max_features': 1000, 'min_df': 0}, # WPS. freqs.

            {'analyzer': word_per_sent, 'ngram_range': (1, 1), 'binary': False, 'max_features': 200, 'min_df': 0}, # WPS. freqs.
            # {'analyzer': stop_word_per_sent, 'ngram_range': (1, 2), 'binary': False, 'max_features': 200, 'min_df': 0} # stop words. freqs.
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

    def fit(self, X, y, selection_method='chi2', classifier='logreg'):

        if self.selection_method == 'chi2':
            self.sel = SelectKBest(chi2, k=1500)

        if self.selection_method == 'logreg':
            self.sel = LogisticRegression(penalty='l1', C=1)

        if self.classifier_type == 'svc':
            self.cls = SVC(kernel=str('linear'), C=1)

        if self.classifier_type == 'logreg':
            self.cls = LogisticRegression()

        print 'vectorizing model..'

        XX = self.vectorize(X)

        print 'selecting features..'
        print XX.shape
        XX = self.sel.fit_transform(XX, y)
        print XX.shape

        print 'training model..'
        self.cls.fit(XX, y)
        self.classes_ = self.cls.classes_

    def predict(self, X):
        XX = self.vectorize(X)
        XX = self.sel.transform(XX)
        # XX = self.sel2.transform(XX)
        return self.cls.predict(XX)

    def predict_proba(self, X):
        XX = self.vectorize(X)
        XX = self.sel.transform(XX)
        return self.cls.predict_proba(XX)
