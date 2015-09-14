from utils import load_train_data, dependencies
from optparse import OptionParser
from algorithm import Model
import pickle

parser = OptionParser()
parser.add_option("-i", "--input", action="store", type="string", dest="input", default="example_train.txt")
parser.add_option("-o", "--output", action="store", type="string", dest="output", default="model.pkl")
(options, args) = parser.parse_args()

print 'loading data..'
X, y = load_train_data(options.input)

m = Model()
m.fit(X, y)

pickle.dump(m, open(options.output, 'wb'))
print "Now you can predict authors with \'python predict.py -m  " + options.output + "\'"
