import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def saveModels(pca, tSNE):
    with open('models/P6-PCA.pkl', 'wb') as f:
        pickle.dump(pca, f)

    with open('models/P6-tSNE.pkl', 'wb') as f:
        pickle.dump(tSNE, f)


def getData():
    df = pd.read_csv('data/ecs171.dataset.txt', delim_whitespace=True)
    df = df.dropna(axis=1, how='any')

    strain = df.iloc[:, 1]
    med = df.iloc[:, 2]
    env_pert = df.iloc[:, 3]
    gene_pert = df.iloc[:, 4]
    X = df.iloc[:, 6:]

    return X, [strain, med, env_pert, gene_pert]


def pltPCA(X, y):
    classes = np.unique(y)

    plt.figure()
    for c in classes:
        plt.scatter(X[y == c, 0], X[y == c, 1], label=c)

    plt.legend(loc='best')
    plt.title('PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    y_name = pd.DataFrame(y).columns[0]
    plt.savefig('plots/P6-' + 'PCA-' + y_name + '.pdf')


def pltTSNE(X, y):
    classes = np.unique(y)

    plt.figure()
    for c in classes:
        plt.scatter(X[y == c, 0], X[y == c, 1], label=c)

    plt.legend(loc='best')
    plt.title('t-SNE')

    y_name = pd.DataFrame(y).columns[0]
    plt.savefig('plots/P6-' + 'tSNE-' + y_name + '.pdf')


def reduceDimension():
    X, ys = getData()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    tsne = TSNE()
    X_tsne = tsne.fit_transform(X)

    for y in ys:
        pltPCA(X_pca, y)
        pltTSNE(X_tsne, y)


if __name__ == '__main__':
    reduceDimension()
