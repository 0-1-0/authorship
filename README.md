# How to setup python environment:

1. Install pyhon 2.7
2. Create virtual environment 'virtualenv venv --no-site-packages'
3. Activate it 'source venv/bin/activate'
4. Install python setup utils / easy_install (instructions: https://pypi.python.org/pypi/setuptools)
5. Install pip ('sudo easy_install pip')
6. Install scikit_learn, scipy, numpy, spacy ('pip install -U numpy scipy scikit-learn spacy &&  python -m spacy.en.download all')
7. Train model with 'python train.py', command options are '-i' : input file with train data, '-o': output file for model, -s: feature selection algorithm [logreg, chi2], -c: classifier type [logreg, svc]. If no options provided, example_train.txt will be used by defauld, and model will be saved as cls.pkl
8. Predict probabilities of test data with 'python predict.py', command options: '-i': input test file, '-o': output file with results, '-m': path to model file. If no options probided, cls.pkl will be used as model, example_test.txt as a test data, and results will be shown to console.
9. Train and test data have to be in the same format, as example_train.txt and example_test.txt.
10. As well, you could check accuracy with cross-validation. Use python check_accuracy.py -i train.txt for that

Alternatevly, you could just run ./setup script

# How to use binary distributions:

1. Unzip dist.zip
2. Unzip data.zip into dist folder
3. Run train and predict binaries with ./dist/train -i train.txt and ./dist/predict -i test.txt
4. Check accuracy with ./dist/check_accuracy -i enrone.txt

# CLI options for check_accuracy

1. -s, --selection [chi2, logreg] - feature selection method, either chi2-based statistical test, or most important features with L1 logisitic regression
2. -c, --cls [logreg, svc] - classifier type, either logistic regression with L2 regularization, or SVM classifier
3. -i, --input - location of file with train data
4. -n - number of samples to use

# How to create binaries:

1. Install cxfreeze (http://cx-freeze.sourceforge.net)
2. Run cxfreeze <name of the script you want to compile>, for example cxfreeze train.py
3. By default binaries are saved in ./dist As well you could specify another directory (see cxfreeze documentation)

# TODO:

1. Try approach with cumulative TF-IDF weighted word2vec semantic vectors
2. Provide command options for 
a) feature generation algorithm (-f) [ngrams, word2vec]
b) classification algorithm (-c) [svc, linearsvc, regression, dt, boosting]

# Future:
1. Use full reddit corpus for training? https://www.reddit.com/r/datasets/comments/3mg812/full_reddit_submission_corpus_now_available_2006/ 42 Gb compressed