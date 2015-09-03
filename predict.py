import pickle
from utils import load_test_data

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn import cross_validation
import numpy as np
import nltk

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-i", "--input", action="store", type="string", dest="input")
parser.add_option("-m", "--model", action="store", type="string", dest="model")
parser.add_option("-o", "--output", action="store", type="string", dest="output")

(options, args) = parser.parse_args()

if options.model:
    (cls, vectorizer) = pickle.load(open(options.model))
else:
    (cls, vectorizer) = pickle.load(open('cls.pkl'))

if options.input:
    X = load_test_data(options.input)
else:
    X = load_test_data()

print 'num of testing samples:', len(X), '\n'

# N-gram features
# word_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(2, 2), binary = False, max_features = 2000)
# char_vectorizer = TfidfVectorizer(ngram_range=(2, 3), analyzer="char", binary=False, min_df=0, max_features = 2000)

# our vectors are the feature union of word/char ngrams
# vectorizer = FeatureUnion([
#     ("chars", char_vectorizer),
#     ("words", word_vectorizer),
#     #("pos", POS_vectorizer)
# ])
matrix = vectorizer.transform(X)
X = matrix.toarray()

if options.output:
    f = open(options.output, 'wb')
    for i, _ in enumerate(cls.predict_proba(X)):
        f.write(str(i) + '\n')
        for (p, a) in reversed(sorted(zip(_, cls.classes_))):
            f.write(a + ' : ' + str(p) + '\n')
        f.write('--\n')
else:
    for i, _ in enumerate(cls.predict_proba(X)):
        print i
        for (p, a) in reversed(sorted(zip(_, cls.classes_))):
            print a, ':', p
        print '--'
