
import numpy as np
from sklearn import metrics

import torch


def batch_precision_recall_f1(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.0):
    yp = y_pred.reshape(-1, y_pred.size(-1)).detach().cpu().numpy()
    yt = y_true.reshape(-1, y_true.size(-1)).detach().cpu().numpy().astype(np.int32)

    yp = (yp >= threshold).astype(np.int32)
    
    ind = np.any(yt, axis=0)  

    if np.any(ind):
        pre, rec, fscore, support = metrics.precision_recall_fscore_support(yt[:, ind], yp[:, ind], average=None, zero_division=0)
        return np.mean(pre), np.mean(rec), np.mean(fscore)
    else:
        return 0.0, 0.0, 0.0


def roc_auc(y_pred: torch.Tensor, y_true: torch.Tensor):
    yp = y_pred.reshape(-1, y_pred.size(-1)).detach().cpu().numpy()
    yt = y_true.reshape(-1, y_true.size(-1)).detach().cpu().numpy().astype(np.int32)

    ind = np.any(yt > 0, axis=0)
    roc = metrics.roc_auc_score(yt[:, ind], yp[:, ind], average=None)

    return np.mean(roc)


def average_precision(y_pred: torch.Tensor, y_true: torch.Tensor):
    yp = y_pred.reshape(-1, y_pred.size(-1)).detach().cpu().numpy()
    yt = y_true.reshape(-1, y_true.size(-1)).detach().cpu().numpy().astype(np.int32)

    ind = np.any(yt > 0, axis=0)
    ps = metrics.average_precision_score(yt[:, ind], yp[:, ind], average=None)

    return np.mean(ps)

