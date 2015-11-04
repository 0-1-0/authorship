from utils import load_train_data, dependencies
from optparse import OptionParser
from algorithm import Model
import pickle

parser = OptionParser()
parser.add_option("-i", "--input", action="store", type="string", dest="input", default="example_train.txt")
parser.add_option("-o", "--output", action="store", type="string", dest="output", default="model.pkl")
parser.add_option("-s", "--selection", action="store", type="string", dest="selection_method", default="chi2")
parser.add_option("-c", "--cls", action="store", type="string", dest="classifier_type", default="logreg")
parser.add_option("-v", "--vectorizer", action="store", type="string", dest="vectorizer", default="bow")
(options, args) = parser.parse_args()

print 'loading data..'
X, y = load_train_data(options.input)

m = Model(classifier_type=options.classifier_type, selection_method=options.selection_method)
m.fit(X, y)

pickle.dump(m, open(options.output, 'wb'))
print "Now you can predict authors with \'python predict.py -m  " + options.output + "\'"
