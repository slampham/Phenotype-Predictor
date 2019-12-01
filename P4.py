import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from tensorflow_core.python.keras.utils import np_utils

warnings.filterwarnings("ignore", category=UserWarning)


def getData():
    df = pd.read_csv('data/ecs171.dataset.txt', delim_whitespace=True)
    df = df.dropna(axis=1, how='any')

    strain = df.iloc[:, 1]
    med = df.iloc[:, 2]
    env_pert = df.iloc[:, 3]
    gene_pert = df.iloc[:, 4]
    X = df.iloc[:, 6:]

    # Use feature selection from P1
    reg = pickle.load(open('models/P1.sav', 'rb'))
    X = X[X.columns[reg.coef_ != 0]]
    print("Num features: ", X.shape[1])

    return X, [strain, med, env_pert, gene_pert]


def plotROC(problem):
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("plots/" + problem + "-ROC.pdf")
    plt.legend(loc="lower right")
    plt.savefig("plots/" + problem + "-ROC.pdf")
    plt.close(fig=1)


def plotPR(problem):
    plt.figure(2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("plots/" + problem + "-PR.pdf")
    plt.legend(loc="lower right")
    plt.savefig("plots/" + problem + "-PR.pdf")
    plt.close(fig=2)


# Referenced implementation: "https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html"
#                            "https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html"
def fourSVM(X, ys, problem):
    y_names = ["strain", "med", "env_pert", "gene_pert"]

    for i, y in enumerate(ys):
        encoder = LabelEncoder()
        encoded_Y = encoder.fit_transform(y)
        encoded_Y = np_utils.to_categorical(encoded_Y)

        y_scores = []; y_tests = []

        kf = KFold(n_splits=5)
        for train, test in kf.split(X):
            X_train, X_test = X.iloc[train], X.iloc[test]
            y_train, y_test = encoded_Y[train], encoded_Y[test]

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
        plt.plot(fpr, tpr, label="{0} (area = {1:.2f})".format(y_names[i], roc_auc))
        plt.figure(2)
        plt.plot(recall, precision, label="{0} (area = {1:.2f})".format(y_names[i], pr_auc))

    plotROC(problem)
    plotPR(problem)


if __name__ == '__main__':
    X, ys = getData()
    fourSVM(X, ys, problem='P4')
