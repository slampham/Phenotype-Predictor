import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import concat
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC


def getData():
    df = pd.read_csv('data/ecs171.dataset.txt', delim_whitespace=True)
    df = df.dropna(axis=1, how='any')

    med = df.iloc[:, 2]
    env_pert = df.iloc[:, 3]
    X = df.iloc[:, 6:]

    # Use feature selection from P1
    reg = pickle.load(open('models/P1.pkl', 'rb'))
    X = X[X.columns[reg.coef_ != 0]]
    print("Num features: ", X.shape[1])

    y = pd.concat([med, env_pert], axis=1)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y.to_numpy())

    return X, y


def dummyClassifier(X_train, X_test, y_train, y_test):
    for strategy in ['stratified', 'most_frequent', 'prior', 'uniform']:
        dummy = DummyClassifier(strategy=strategy, random_state=0)
        dummy.fit(X_train, y_train)
        print(f"{strategy}: {dummy.score(X_test, y_test) * 100: .3g}%")


def plotROC(fpr, tpr, roc_auc):
    plt.figure(1)
    plt.plot(fpr, tpr, label=f"ROC (area = {roc_auc: .2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('plots/P5-ROC.pdf')
    plt.legend(loc="lower right")
    plt.savefig("plots/P5-ROC.pdf")


def plotPR(recall, precision, pr_auc):
    plt.figure(2)
    plt.plot(recall, precision, label=f"PR (area = {pr_auc: .2f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('plots/P5-PR.pdf')
    plt.legend(loc="lower right")
    plt.savefig("plots/P5-PR.pdf")


# Referenced implementation: "https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html"
def compositeSVM():
    X, y = getData()
    y_scores = []; y_tests = []
    kf = KFold(n_splits=10)
    classifier = OneVsRestClassifier(SVC(kernel='linear'))

    for train, test in kf.split(X):
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y[train], y[test]

        y_tests.append(y_test)
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
        y_scores.append(y_score)

    y_tests = np.concatenate(y_tests).ravel()
    y_scores = np.concatenate(y_scores).ravel()

    fpr, tpr, _ = roc_curve(y_tests, y_scores)
    precision, recall, _ = precision_recall_curve(y_tests, y_scores)

    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    dummyClassifier(x_train, x_test, y_train, y_test)

    plotROC(fpr, tpr, roc_auc)
    plotPR(recall, precision, pr_auc)


if __name__ == '__main__':
    compositeSVM()
