import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


def getData():
    df = pd.read_csv('data/ecs171.dataset.txt', delim_whitespace=True)
    df = df.dropna(axis=1, how='any')
    X = df.iloc[:, 6:]
    return X


def reduceDimension():
    X = getData()
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)


if __name__ == '__main__':
    reduceDimension()
