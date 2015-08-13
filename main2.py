from os import listdir
from os.path import isfile, join, isdir
import numpy as np
import nltk
import glob
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.vq import whiten
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

def LexicalFeatures(chapters):
    """
    Compute feature vectors for word and punctuation features
    """
    num_chapters = len(chapters)
    fvs_lexical = np.zeros((len(chapters), 3), np.float64)
    fvs_punct = np.zeros((len(chapters), 3), np.float64)
    for e, ch_text in enumerate(chapters):
        # note: the nltk.word_tokenize includes punctuation
        tokens = nltk.word_tokenize(ch_text.lower())
        words = word_tokenizer.tokenize(ch_text.lower())
        sentences = sentence_tokenizer.tokenize(ch_text)
        vocab = set(words)
        words_per_sentence = np.array([len(word_tokenizer.tokenize(s))
                                       for s in sentences])

        # average number of words per sentence
        fvs_lexical[e, 0] = words_per_sentence.mean()
        # sentence length variation
        fvs_lexical[e, 1] = words_per_sentence.std()
        # Lexical diversity
        fvs_lexical[e, 2] = len(vocab) / float(len(words))

        # Commas per sentence
        fvs_punct[e, 0] = tokens.count(',') / float(len(sentences))
        # Semicolons per sentence
        fvs_punct[e, 1] = tokens.count(';') / float(len(sentences))
        # Colons per sentence
        fvs_punct[e, 2] = tokens.count(':') / float(len(sentences))

    # apply whitening to decorrelate the features
    fvs_lexical = whiten(fvs_lexical)
    fvs_punct = whiten(fvs_punct)

    return fvs_lexical, fvs_punct

def BagOfWords(chapters):
    """
    Compute the bag of words feature vectors, based on the most common words
     in the whole book
    """
    # get most common words in the whole book
    NUM_TOP_WORDS = 10
    all_tokens = nltk.word_tokenize(all_text)
    fdist = nltk.FreqDist(all_tokens)
    vocab = fdist.keys()[:NUM_TOP_WORDS]

    # use sklearn to create the bag for words feature vector for each chapter
    vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=nltk.word_tokenize)
    fvs_bow = vectorizer.fit_transform(chapters).toarray().astype(np.float64)

    # normalise by dividing each row by its Euclidean norm
    fvs_bow /= np.c_[np.apply_along_axis(np.linalg.norm, 1, fvs_bow)]

    return fvs_bow

def SyntacticFeatures(chapters):
    """
    Extract feature vector for part of speech frequencies
    """
    def token_to_pos(ch):
        tokens = nltk.word_tokenize(ch)
        return [p[1] for p in nltk.pos_tag(tokens)]

    chapters_pos = [token_to_pos(ch) for ch in chapters]
    pos_list = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS']
    fvs_syntax = np.array([[ch.count(pos) for pos in pos_list]
                           for ch in chapters_pos]).astype(np.float64)

    # normalise by dividing each row by number of tokens in the chapter
    fvs_syntax /= np.c_[np.array([len(ch) for ch in chapters_pos])]

    return fvs_syntax


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

trainset = load_judje('/Users/N/Desktop/original')
for x in trainset[:2]:
    print SyntacticFeatures(x['text'])
