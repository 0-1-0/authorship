from os import listdir
from os.path import isfile, join, isdir
from options import SPECIAL1, PARAGRAPH


def load_judje(input_dir):
    trainfiles = [f for f in listdir(input_dir) if isdir(join(input_dir, f))]
    X, y = [], []

    for author in trainfiles:
        for txt in listdir(join(input_dir, author)):
            fname = join(input_dir, author, txt)
            if isfile(fname):
                try:
                    data = open(fname).read()
                    data.encode('utf-8')
                    X.append(data)
                    y.append(author)
                except:
                    pass
    return (X, y)


def save_corpus(X, y, path):
    f = open(path, 'w')
    for x, l in zip(X, y):
        f.write(l)
        f.write(SPECIAL1)
        f.write(x)
        f.write(PARAGRAPH)

X, y = load_judje('/Users/N/Desktop/original')
print len(X)
save_corpus(X, y, 'example.txt')
