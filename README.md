# How to use this:

1. Install pyhon 2.7
2. Create virtual environment 'virtualenv venv --no-site-packages'
3. Activate it 'source venv/bin/activate'
4. Install python setup utils / easy_install (instructions: https://pypi.python.org/pypi/setuptools)
5. Install pip ('sudo easy_install pip')
6. Install scikit_learn, scipy, numpy, nltk, spacy ('pip install -U numpy scipy scikit-learn nltk spacy &&  python -m spacy.en.download all')
7. Train model with 'python train.py', command options are '-i' : input file with train data, '-o': output file for model. If no options provided, example_train.txt will be used by defauld, and model will be saved as cls.pkl
8. Predict probabilities of test data with 'python predict.py', command options: '-i': input test file, '-o': output file with results, '-m': path to model file. If no options probided, cls.pkl will be used as model, example_test.txt as a test data, and results will be shown to console.
9. Train and test data have to be in the same format, as example_train.txt and example_test.txt.
10. As well, you could check accuracy with cross-validation. Use python check_accuracy.py -i train.txt for that

# How to use binary distributions:

1. Unzip dist.zip
2. Run train and predict binaries with ./dist/train -i train.txt and ./dist/predict -i test.txt

# How to create binaries:

1. Install cxfreeze (http://cx-freeze.sourceforge.net)
2. Run cxfreeze <name of the script you want to compile>, for example cxfreeze train.py
3. By default binaries are saved in ./dist As well you could specify another directory (see cxfreeze documentation)

# TODO:

1. Try approach with cumulative TF-IDF weighted word2vec semantic vectors
2. Provide command options for 
a) feature generation algorithm (-f) [ngrams, word2vec]
b) feature selection method (-s) [chi2, regression]
c) classification algorithm (-c) [svc, linearsvc, regression, dt, boosting]
3. Try to use different feature selection algorithm (iterative logistic regression with l1 regularisation)

# Future:
1. Use full reddit corpus for training? https://www.reddit.com/r/datasets/comments/3mg812/full_reddit_submission_corpus_now_available_2006/ 42 Gb compressed