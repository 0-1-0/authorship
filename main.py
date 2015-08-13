from os import listdir
from os.path import isfile, join, isdir


# Enron email corpus
# Data: https://www.cs.cmu.edu/~./enron/
def load_corpus(input_dir):
  import email

  trainfiles= [  f for f in listdir( input_dir ) if isdir(join(input_dir ,f)) ]

  trainset = []
  for author in trainfiles:

    sent_items = join(input_dir, author, 'sent_items')
    if isdir(sent_items) and len(listdir(sent_items)) > 500:
      print author, len(listdir(sent_items))

      for msg in listdir(sent_items):
        fname = join(sent_items, msg)
        if isfile(fname):
          e = email.message_from_file(open(fname))
          txt = e.get_payload().split('-----')[0]
          txt = ''.join(e for e in txt if e.isalnum() or e == ' ')

          trainset.append({'label':author,'text':txt})
  return trainset


# Judgement attribution in Law
# Data: www.csse.monash.edu.au/research/umnl/data
def load_judje(input_dir):
  trainfiles = [  f for f in listdir( input_dir ) if isdir(join(input_dir ,f)) ]
  trainset = []

  for author in trainfiles:
    for txt in listdir(join(input_dir, author)):
      fname = join(input_dir, author, txt)
      if isfile(fname):
        try:
          data = open(fname).read()
          data.encode('utf-8')
          trainset.append({'label': author, 'text': data})
        except:
          pass
  return trainset


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import NuSVC
import sys
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion
 
from sklearn import cross_validation
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

def train_model(trainset):

  # create 2 blocks of features, word and character ngrams, size of 2 (using TF-IDF method)
  # we can also append here multiple other features in general

  word_vector = TfidfVectorizer( analyzer="word" , ngram_range=(2,2), binary = False, max_features= 2000 )
  char_vector = TfidfVectorizer(ngram_range=(2, 3), analyzer="char", binary=False, min_df=0 , max_features=2000 )

  # our vectors are the feature union of word/char ngrams
  vectorizer = FeatureUnion([  ("chars", char_vector),("words", word_vector)  ] )

  corpus, classes = [], []
    

  for item in trainset:    
    corpus.append( item['text'] )
    classes.append( item['label'] )

  print "num of training instances: ", len(classes)    
  print "num of training classes: ", len(set(classes))

  #fit the model of tfidf vectors for the coprus
  matrix = vectorizer.fit_transform(corpus)
 
  print "num of features: " , len(vectorizer.get_feature_names())
  print "training model"
  X = matrix.toarray()
  y = np.asarray(classes)

  print X[0]

  # Here are results of several different models for Law corpus:

  # model  = SVC(kernel='sigmoid') # ->                       0.38
  # model  = KNeighborsClassifier(algorithm = 'kd_tree') # -> 0.41
  # model = AdaBoostClassifier() #->                            0.46
  # model  = RandomForestClassifier() # ->                    0.52
  # model  = LogisticRegression() # ->                        0.65 
  model  = LinearSVC( loss='l1', dual=True) # ->              0.70
  # Results of several different models for Enron corpus:
  # model  = LinearSVC( loss='l1', dual=True) # ->              0.6

  scores = cross_validation.cross_val_score(  estimator = model,
    X = matrix.toarray(), 
        y= np.asarray(classes), cv=10  )

  print "10-fold cross-validation results:", "mean score = ", scores.mean(), "std=", scores.std(), ", num folds =", len(scores)


data = load_judje('/Users/N/Desktop/original')
#data = load_corpus('/Users/N/Desktop/maildir')
train_model(data)

