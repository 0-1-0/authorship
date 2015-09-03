from utils import load_train_data
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i", "--input", action="store", type="string", dest="input")
(options, args) = parser.parse_args()

print 'loading data..'
if options.input:
    X, y = load_train_data(options.input)
else:
    X, y = load_train_data()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
import nltk
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn import cross_validation


def extract_pos(x):
    tokens = nltk.word_tokenize(x)
    res = ['_'.join(p) for p in nltk.pos_tag(tokens)]
    # print res
    return res

# N-gram features
word_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(2, 4), binary = False, max_features = 3000)
char_vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer="char", binary=False, min_df=0, max_features = 3000)
POS_vectorizer = TfidfVectorizer(ngram_range=(2, 7), analyzer=extract_pos, binary=False, min_df=0, max_features = 3000)


# our vectors are the feature union of word/char ngrams
vectorizer = FeatureUnion([
    ("chars", char_vectorizer),
    ("words", word_vectorizer),
    ("pos", POS_vectorizer)
])
matrix = vectorizer.fit_transform(X)
X = matrix.toarray()


# Syntactic Features


print "num of training instances: ", len(y)
print "num of training classes: ", len(set(y))

print "num of features: ", len(vectorizer.get_feature_names())
print "performing cross-validation.."


# cls = LinearSVC(loss='l1', dual=True)
cls = LogisticRegression()
print 'done'

scores = cross_validation.cross_val_score(estimator=cls, X=matrix.toarray(), y=np.asarray(y), cv=3)

print "3-fold cross-validation results:", "mean score = ", scores.mean(), "std=", scores.std(), ", num folds =", len(scores)
