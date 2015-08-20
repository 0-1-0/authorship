from options import SPECIAL1, PARAGRAPH
import random


def load_train_data(path='example_train.txt'):
    data = open(path).read().split(PARAGRAPH)[:-1]
    random.shuffle(data)
    X, y = [], []
    for p in data:
        l, x = p.split(SPECIAL1)
        X.append(x)
        y.append(l)

    return (X, y)


def save_random_paragraphs():
    X, y = load_train_data()
    output = open('example_test.txt', 'wb')
    for x in X[:100]:
        output.write(x)
        output.write(PARAGRAPH)


def load_test_data(path='example_test.txt'):
    data = open(path).read().split(PARAGRAPH)[:-1]
    return data
