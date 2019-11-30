import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LassoCV


def getData():
    df = pd.read_csv('data/ecs171.dataset.txt', delim_whitespace=True)
    df = df.dropna(axis=1, how='any')

    growth_rate = df.iloc[:, 5]
    gene_expr = df.iloc[:, 6:]

    return gene_expr, growth_rate


def lassoPredictor(X, y):
    reg = LassoCV(cv=5, max_iter=1000, tol=0.0075, n_jobs=-1, precompute='auto', verbose=1)
    reg.fit(X, y)
    return reg


if __name__ == '__main__':
    X, y = getData()
    # reg = lassoPredictor(X, y)      # If you want to run model again
    reg = pickle.load(open('P1.sav', 'rb'))

    print(f"Optimal parameter val: {reg.alpha_ : .3g}")
    print(f"Num non-zero coeff: {np.count_nonzero(reg.coef_)}")
    print(f"Gen Error: {reg.mse_path_.mean() * 100 : .3g}%")
