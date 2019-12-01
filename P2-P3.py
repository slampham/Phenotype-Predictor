import math
import pickle

import scipy
import numpy as np
from sklearn.utils import resample

from P1 import getData


# Referenced implementation : "https://machinelearningmastery.com/prediction-intervals-for-machine-learning/"
def bootStrap(num_times=1000):
    X, _ = getData()
    reg = pickle.load(open('models/P1.pkl', 'rb'))

    y_predicts = []
    for boot in range(num_times):
        X_boot = resample(X, n_samples=X.shape[0], replace=True)
        X_mean = np.array(X_boot.mean()).reshape(1, -1)
        y_predicts.append(reg.predict(X_mean))

    return np.concatenate(y_predicts)


def confidenceInterval(data):
    mean = np.mean(data)
    n = len(data)
    std = scipy.std(data)
    z_crit = 1.96
    margin_err = z_crit * std / math.sqrt(n)
    confi_intv = (mean - margin_err, mean + margin_err)
    return confi_intv


if __name__ == '__main__':
    y_predicts = bootStrap(num_times=1000)
    print(f"Predicted value for mean X (resampled n times) {np.around(y_predicts, 3)}")
    print(f"Confidence interval: {np.around(confidenceInterval(y_predicts), 6)}")
