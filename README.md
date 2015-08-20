# How to use this:

1. Download training data 
  * Enron email corpus: https://www.cs.cmu.edu/~./enron/
  * Judgement attribution in Law corpus:  http://www.csse.monash.edu.au/research/umnl/data
2. Unpack both corpuses and edit paths to datasets on lines 116-117 of main.py
3. Install pyhon 2.7
4. Install python setup utils / easy_install (instructions: https://pypi.python.org/pypi/setuptools)
5. Install pip ('sudo easy_install pip')
6. Install scikit_learn, scipy, numpy, nltk ('pip install -U numpy scipy scikit-learn nltk')
7. Train model with 'python train.py', command options are '-i' : input file with train data, '-o': output file for model
8. Predict probabilities of test data with 'python predict.py', command options: '-i': input test file, '-o': output file with results
