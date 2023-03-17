import numpy as np
from sklearn import metrics
from scipy import stats


def roc_auc(y_true, y_pred):
    ind = np.any(y_true > 0, axis=0)
    roc = metrics.roc_auc_score(y_true[:, ind], y_pred[:, ind], average=None)

    return np.mean(roc)


def average_precision(y_true, y_pred):
    ind = np.any(y_true > 0, axis=0)
    ap = metrics.average_precision_score(y_true[:, ind], y_pred[:, ind], average=None)

    return np.mean(ap)


def d_prime(auc):
    return np.sqrt(2) * stats.norm.ppf(auc)

