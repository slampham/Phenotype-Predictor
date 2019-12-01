import pickle

import pandas as pd

from P4 import fourSVM, getData


def reducedDimensionSVM():
    X, ys = getData()

    X_pca = pickle.load(open('models/P6-PCA.pkl', 'rb'))
    X_tsne = pickle.load(open('models/P6-tSNE.pkl', 'rb'))

    fourSVM(X, ys, '4')
    fourSVM(pd.DataFrame(X_pca), ys, '7-PCA')
    fourSVM(pd.DataFrame(X_tsne), ys, '7-tSNE')


if __name__ == '__main__':
    reducedDimensionSVM()
