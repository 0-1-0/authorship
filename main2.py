from utils import load_data

X, y = load_data()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn import cross_validation
import numpy as np
import nltk


def extract_pos(x):
    tokens = nltk.word_tokenize(x)
    res = ['_'.join(p) for p in nltk.pos_tag(tokens)]
    # print res
    return res

# N-gram features
word_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(2, 2), binary = False, max_features = 2000)
char_vectorizer = TfidfVectorizer(ngram_range=(2, 3), analyzer="char", binary=False, min_df=0, max_features = 2000)
POS_vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer=extract_pos, binary=False, min_df=0, max_features = 2000)


# our vectors are the feature union of word/char ngrams
vectorizer = FeatureUnion([
    ("chars", char_vectorizer),
    ("words", word_vectorizer),
    #("pos", POS_vectorizer)
])
matrix = vectorizer.fit_transform(X)
X = matrix.toarray()


# Syntactic Features


print "num of training instances: ", len(y)
print "num of training classes: ", len(set(y))

print "num of features: ", len(vectorizer.get_feature_names())
print "training model"


cls = LinearSVC(loss='l1', dual=True)

scores = cross_validation.cross_val_score(estimator=cls, X=X, y=np.asarray(y), cv=10)
print scores.mean()
