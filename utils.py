from options import SPECIAL1, PARAGRAPH
import random


def clear_text(x):
    txt = x
    txt = txt.split('***********')[0]
    txt = txt.split('___________')[0]
    txt = txt.split('> >')[0]
    txt = txt.split('>>')[0]
    txt = txt.replace("\0", "")
    txt = txt.replace("^@", "")
    return txt


def load_train_data(path='example_train.txt'):
    data = open(path).read()
    data = data.split(PARAGRAPH)[:-1]
    random.shuffle(data)
    X, y = [], []
    for p in data:
        l, x = p.split(SPECIAL1)
        x = clear_text(x)
        X.append(x)
        y.append(l)

    return (X, y)


def save_data(X, y, path):
    output = open(path, 'w')
    for (x, y) in zip(X, y):
        output.write(y)
        output.write(SPECIAL1)
        output.write(x)
        output.write(PARAGRAPH)


def save_random_paragraphs():
    X, y = load_train_data()
    output = open('example_test.txt', 'wb')
    for x in X[:100]:
        output.write(x)
        output.write(PARAGRAPH)


def load_test_data(path):
    data = open(path).read().split(PARAGRAPH)[:-1]
    return map(clear_text, data)


def dependencies():
    from scipy.sparse.csgraph import _validation
    from sklearn.utils import lgamma
    import cymem
    import cymem
    import cymem.cymem
    import preshed.maps
    from spacy import strings
    import murmurhash.mrmr
    from preshed import counter
    from spacy import morphology
    from spacy import lemmatizer
    from spacy import lexeme
    import unidecode
    from spacy import cfile
    from spacy import tokens
    from spacy.tokens import doc
    from spacy.serialize import bits
    from spacy.serialize import huffman
    from spacy.serialize import packer
    from spacy.serialize.packer import util
    from spacy import vocab
    from spacy import util
    import thinc.cache
    from thinc import features
    from thinc import sparse
    from thinc import learner
    from thinc import search
    from thinc import api
    from spacy.syntax import stateclass
    from spacy import gold
    from spacy.syntax import transition_system
    from spacy.syntax import parser
    from spacy.syntax import _parse_features
    from spacy.syntax import util
    from sklearn.utils import weight_vector
    from sklearn.decomposition import PCA, FastICA
    from sklearn.pls import PLSRegression
    from matplotlib import numerix
    import matplotlib.numerix.random_array
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
    from sklearn.feature_selection import RFECV
    from sklearn.svm import SVC
    import spacy

from os import listdir
from os.path import isfile, join, isdir


from collections import defaultdict


# Enron email corpus
# Data: https://www.cs.cmu.edu/~./enron/
def load_enrone(input_dir='/Users/N/Desktop/maildir'):
    trainfiles = [f for f in listdir(input_dir) if isdir(join(input_dir, f))]
    trainset = defaultdict(set)

    for author in trainfiles:

        sent_items = join(input_dir, author, 'sent_items')
        if isdir(sent_items):
            print author, len(listdir(sent_items))

            for msg in listdir(sent_items):
                fname = join(sent_items, msg)
                if isfile(fname):
                    try: # dirty hacks
                        txt = open(fname).read()
                        txt = txt.split('X-FileName')[1]
                        txt = txt.split('.pst\r\n\r\n')[1]
                        txt = txt.split('---')[0]
                        txt = txt.split('From:')[0]
                        txt = txt.split('To:')[0]
                        txt = txt.split('cc:')[0]
                        txt = txt.split('<OMNI>')[0]
                        txt = txt.split('***********')[0]
                        txt = txt.split('___________')[0]
                        txt = txt.split('> >')[0]

                        if len(txt) > 100:
                            trainset[author].add(txt)
                    except:
                        pass
    return trainset


# Of course we couldn't save them from bancrupcy
# but at least we could extract corpus with users who have 100+ emails with 100+ characters each
def save_enrone():
    enrone = load_enrone()
    X, y = [], []

    for author in enrone.keys():
        if len(enrone[author]) > 100:
            for msg in enrone[author]:
                X.append(msg)
                y.append(author)

    print len(X), 'items by', len(set(y)), 'authors'

    save_data(X, y, 'enrone.txt')
