import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.svm import SVC
from tensorflow_core.python.keras.utils import np_utils


def getData():
    df = pd.read_csv('data/ecs171.dataset.txt', delim_whitespace=True)
    df = df.dropna(axis=1, how='any')

    med = df.iloc[:, 2]
    env_pert = df.iloc[:, 3]
    X = df.iloc[:, 6:]

    # Use feature selection from P1
    reg = pickle.load(open('P1.sav', 'rb'))
    X = X[X.columns[reg.coef_ != 0]]
    print("Num features: ", X.shape[1])

    return X, med, env_pert


def plotROC():
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Multiclass')
    plt.legend(loc="lower right")
    plt.savefig("plots/P5-ROC.pdf")


def plotPR():
    plt.figure(2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="lower right")
    plt.savefig("plots/P5-PR.pdf")


# Referenced implementation: "https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html"
def compositeSVM():
    X, med, env_pert = getData()
    y = np.concatenate([med, env_pert])

    X = np.concatenate([X, X])
    y = label_binarize(y, np.unique(y))
    y_scores = []; y_tests = []
    kf = KFold(n_splits=10)

    for train, test in kf.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        classifier = OneVsRestClassifier(SVC(kernel='linear'))
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)

        y_tests.append(y_test)
        y_scores.append(y_score)

    y_tests = np.concatenate(y_tests).ravel()
    y_scores = np.concatenate(y_scores).ravel()

    fpr, tpr, _ = roc_curve(y_tests, y_scores)
    precision, recall, _ = precision_recall_curve(y_tests, y_scores)

    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    plt.figure(1)
    plt.plot(fpr, tpr, label=f"ROC (area = {roc_auc: .2f})")
    plt.figure(2)
    plt.plot(recall, precision, label=f"PR (area = {pr_auc: .2f})")


if __name__ == '__main__':
    compositeSVM()