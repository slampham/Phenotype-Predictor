import pickle

import numpy as np
import pandas as pd
from sklearn.utils import resample

from P1 import getData
from P2 import confidenceInterval


def bootStrap(num_times=1000, n_samples=100):
    mses = []
    X, y = getData()
    X = X.mean()
    y = y.mean()
    data = pd.concat([X, y], axis=1)

    for boot_strap in range(num_times):
        boot_data = resample(data, n_samples=n_samples, replace=True)
        Xboot = boot_data.iloc[:, :-1]
        yboot = boot_data.iloc[:, -1]
        reg = pickle.load(open('models/P1.sav', 'rb'))
        MSE = np.mean((reg.predict(Xboot) - yboot)**2)
        mses.append(MSE)

    return mses


if __name__ == '__main__':
    mses = bootStrap(num_times=10)
    print(confidenceInterval(mses))
