# How to use this:

1. Download training data 
  * Enron email corpus: https://www.cs.cmu.edu/~./enron/
  * Judgement attribution in Law corpus:  http://www.csse.monash.edu.au/research/umnl/data
2. Unpack both corpuses and edit paths to datasets on lines 116-117 of main.py
3. Install pyhon 2.7
4. Install python setup utils / easy_install (instructions: https://pypi.python.org/pypi/setuptools)
5. Install pip ('sudo easy_install pip')
6. Install scikit_learn, scipy, numpy, nltk ('pip install -U numpy scipy scikit-learn nltk')
7. Run code with 'python main.py'
8. Uncomment specific lines of code (lines 100-108) to use different models
9. Change training corpus by commenting / uncommenting lines 117 / 118


TODO: implement execution parameters as command line arguments
