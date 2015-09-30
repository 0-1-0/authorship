# How to use this:

1. Install pyhon 2.7
2. Install python setup utils / easy_install (instructions: https://pypi.python.org/pypi/setuptools)
3. Install pip ('sudo easy_install pip')
4. Install scikit_learn, scipy, numpy, nltk ('pip install -U numpy scipy scikit-learn nltk')
5. Train model with 'python train.py', command options are '-i' : input file with train data, '-o': output file for model. If no options provided, example_train.txt will be used by defauld, and model will be saved as cls.pkl
6. Predict probabilities of test data with 'python predict.py', command options: '-i': input test file, '-o': output file with results, '-m': path to model file. If no options probided, cls.pkl will be used as model, example_test.txt as a test data, and results will be shown to console.
7. Train and test data have to be in the same format, as example_train.txt and example_test.txt.
8. As well, you could check accuracy with cross-validation. Use python check_accuracy.py -i train.txt for that

# How to use binary distributions:

1. Unzip dist.zip
2. Run train and predict binaries with ./dist/train -i train.txt and ./dist/predict -i test.txt

# How to create binaries:

1. Install cxfreeze (http://cx-freeze.sourceforge.net)
2. Run cxfreeze <name of the script you want to compile>, for example cxfreeze train.py
3. By default binaries are saved in ./dist As well you could specify another directory (see cxfreeze documentation)

# TODO:

1. Try approach with cumulative TF-IDF weighted word2vec semantic vectors
2. Provide command options for model size and kind of classification algorithm, describe them below
3. Try to use different feature selection algorithm (iterative logistic regression with l1 regularisation)

# Future:
1. Use full reddit corpus for training? https://www.reddit.com/r/datasets/comments/3mg812/full_reddit_submission_corpus_now_available_2006/ 42 Gb compressed