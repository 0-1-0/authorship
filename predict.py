import pickle
from utils import load_test_data, dependencies
from algorithm import Model
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-i", "--input", action="store", type="string", dest="input", default="example_test.txt")
parser.add_option("-m", "--model", action="store", type="string", dest="model", default="model.pkl")
parser.add_option("-o", "--output", action="store", type="string", dest="output")

(options, args) = parser.parse_args()

m = pickle.load(open(options.model))
X = load_test_data(options.input)
print 'num of testing samples:', len(X), '\n'


if options.output:
    f = open(options.output, 'wb')
    for i, _ in enumerate(m.predict_proba(X)):
        f.write(str(i) + '\n')
        for (p, a) in reversed(sorted(zip(_, m.classes_))):
            f.write(a + ' : ' + str(p) + '\n')
        f.write('--\n')
else:
    for i, _ in enumerate(m.predict_proba(X)):
        print i
        for (p, a) in reversed(sorted(zip(_, m.classes_))):
            print a, ':', p
        print '--'
