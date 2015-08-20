from options import SPECIAL1, PARAGRAPH


def load_data(path='example.txt'):
    data = open(path).read().split(PARAGRAPH)
    X, y = [], []
    for p in data:
        x, l = p.split(SPECIAL1)
        X.append(x)
        y.append(l)

    return (X, y)
