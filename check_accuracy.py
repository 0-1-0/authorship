from utils import load_train_data, dependencies
from optparse import OptionParser
from algorithm import Model
from sklearn import cross_validation
import numpy as np

parser = OptionParser()
parser.add_option("-i", "--input", action="store", type="string", dest="input", default="enrone.txt")
parser.add_option("-s", "--samples", action="store", type="int", dest="N", default=10**5)
(options, args) = parser.parse_args()

print 'loading data..'
X, y = load_train_data(options.input)
X, y = X[:options.N], y[:options.N]

m = Model()

print "num of training instances: ", len(y)
print "num of training classes: ", len(set(y))
print "performing cross-validation.."

scores = cross_validation.cross_val_score(estimator=m, X=X, y=np.asarray(y), cv=3)

print "3-fold cross-validation results:", "mean score = ", scores.mean(), "std=", scores.std(), ", num folds =", len(scores)
