import math
import pickle
from statistics import mean

import pandas as pd
import scipy
import numpy as np
from scipy.stats import sem
from sklearn.utils import resample

from P1 import getData

# Referenced implementation : "https://machinelearningmastery.com/prediction-intervals-for-machine-learning/"

def bootStrap(num_times=1000, n_samples=100):
    mses = []
    X, y = getData()
    data = pd.concat([X, y], axis=1)

    for boot_strap in range(num_times):
        boot_data = resample(data, n_samples=n_samples, replace=True)
        Xboot = boot_data.iloc[:, :-1]
        yboot = boot_data.iloc[:, -1]
        reg = pickle.load(open('P1.sav', 'rb'))
        MSE = np.mean((reg.predict(Xboot) - yboot)**2)
        mses.append(MSE)

    return mses


def confidenceInterval(data):
    n = len(data)
    sigma = scipy.std(data)
    z_crit = scipy.stats.norm.ppf(q=0.975)
    std_err = sigma / math.sqrt(n)
    margin_of_err = z_crit * std_err
    interval = (mean(data) - margin_of_err, mean(data) + margin_of_err)
    return interval


if __name__ == '__main__':
    mses = bootStrap(num_times=10)
    print(confidenceInterval(mses))
