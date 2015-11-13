# Setup

0. Most convinient way is just to run `./setup` script.
If you have any troubles with it, here is the full list of manual operations to complete:

* Install pyhon 2.7
* Create virtual environment `virtualenv venv --no-site-packages`
* Activate it `source venv/bin/activate`
* Install python setup utils / easy_install (instructions: https://pypi.python.org/pypi/setuptools)
* Install pip `sudo easy_install pip`
* Install scikit_learn, scipy, numpy, spacy `pip install -U numpy scipy scikit-learn spacy &&  python -m spacy.en.download all`

# Usage
Firstly, you want to activate virtual environment created on previous step (to use local python libraries, not global).
You could do this with `source venv/bin/activate`. It modifies your PATH in this purpose.
After end of work you could leave venv with `deactivate` command.

Secondly, there are 3 main utilities: train, predict, and check_accuracy
With train.py you are training model on your data, and then predict authorship probabilities with predict.py for anonymous texts.
check_accuracy.py performs cross-validation on training data, so you can test and compare performans on different approaches.

Here is full list of command-line options for them:

CLI options for check_accuracy:

1. -i, --input - path to file with train data. Data have to be in the same format as example_train.py

2. -s, --selection [chi2, logreg, svd1000] - feature selection method. 
    * chi2 - chi2-based statistical test
    * logreg -  most important features with L1 logisitic regression
    * svd1000 - Singular Value Decomposition. 1000 or other number specifies final dimensionality
    * pca500 - Principal Component Analysis. 500 or other number specifies dimensionality like with svd.

3. -c, --cls [logreg, svc, rf100] - classifier type:
    * logreg - logistic regression with L2 regularization
    * svc - SVM classifier. 
    * rf100 - Random Forest classifier with 100 estimators. You could also use rf500, rf1000, and so on.

4. -n - number of samples to use (for quick accuracy check you may want to use small number of samples)

5. -v, --vectorizer [bow, word2vec, word2vec2] - text vectorizer type
    * bow - Bag of Words - vector representation of texts based of frequencies of n-grams, POS tags, and some other additonal statistics
    * word2vec - straitforward sum of word2vec vectors for words in paragraph
    * word2vec2 - TF-IDF weighted  sum of word2vec vectors in text
    * combined - frequency features + sum of word vectors

CLI options for train:

1. -i (same as check_accuracy)
2. -s (same as check_accuracy)
3. -c (same as check_accuracy)
4. -o, --output - output file for model. Default is model.pkl

CLI options for predict:

1. -i, --input - location of file with test data, the file have to be in the same format as example_test.txt
2. -m, --model - location of model file
3. -o, --output - name of file where to store predictions

# Binary distributions:

As alternative way to launching python scripts, you can use binary distributions. How to do that:

* Unzip dist.zip
* Unzip data.zip into dist folder
* Run train and predict binaries with ./dist/train -i train.txt and ./dist/predict -i test.txt
* Check accuracy with ./dist/check_accuracy -i enrone.txt

How to create binaries:

1. Install cxfreeze (http://cx-freeze.sourceforge.net)
2. Run cxfreeze <name of the script you want to compile>, for example cxfreeze train.py
3. By default binaries are saved in ./dist As well you could specify another directory (see cxfreeze documentation)

# TODO:

1. Try approach with cumulative TF-IDF weighted word2vec semantic vectors
2. Provide command options for:
a) feature generation algorithm (-f) [ngrams, word2vec]
3. provide options for logreg, rf, svc parameters as a part of their name

# Future:
1. Use full reddit corpus for training? https://www.reddit.com/r/datasets/comments/3mg812/full_reddit_submission_corpus_now_available_2006/ 42 Gb compressed