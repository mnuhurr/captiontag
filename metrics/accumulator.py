import torch

from typing import Dict, Optional


class Accumulator:
    def __init__(self):
        self.tp = None
        self.tn = None
        self.fp = None
        self.fn = None

    def reset(self):
        self.tp = None
        self.tn = None
        self.fp = None
        self.fn = None

    def add_batch(self, y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.0):
        """
        assume classes are in the last dimension, e.g. y_pred = (batch, ..., n_classes)

        :param y_pred:
        :param y_true:
        :param threshold:
        :return:
        """
        y_pred = y_pred.detach().cpu()
        y_true = y_true.detach().cpu()

        # binarize prediction scores
        y_pred = y_pred if threshold is None else (y_pred > threshold)

        y_pred = y_pred.to(torch.bool)
        y_true = y_true.to(torch.bool)

        dims = list(range(y_pred.dim() - 1))

        tp = torch.sum(y_pred & y_true, dim=dims)
        fp = torch.sum(y_pred & ~y_true, dim=dims)
        tn = torch.sum(~y_pred & ~y_true, dim=dims)
        fn = torch.sum(~y_pred & y_true, dim=dims)

        # print(torch.prod(torch.as_tensor(y_pred.shape[:-1])))
        if self.tp is None:
            self.tp = tp
            self.fp = fp
            self.tn = tn
            self.fn = fn
        else:
            self.tp = self.tp + tp
            self.fp = self.fp + fp
            self.tn = self.tn + tn
            self.fn = self.fn + fn

    @property
    def micro(self) -> Dict[str, torch.Tensor]:
        # counts:
        positives = self.tp + self.fn
        negatives = self.fp + self.tn
        counts = positives + negatives

        n_cls = counts.size(-1)
        precision = torch.zeros(n_cls)
        recall = torch.zeros(n_cls)
        fscore = torch.zeros(n_cls)
        pre_ind = self.tp + self.fp > 0
        rec_ind = self.tp + self.fn > 0
        ind = self.tp + self.fp + self.fn > 0
        precision[pre_ind] = self.tp[pre_ind] / (self.tp[pre_ind] + self.fp[pre_ind])
        recall[rec_ind] = self.tp[rec_ind] / (self.tp[rec_ind] + self.fn[rec_ind])
        fscore[ind] = 2 * precision[ind] * recall[ind] / (precision[ind] + recall[ind])

        return {
            'fscore': fscore,
            'precision': precision,
            'recall': recall,
        }

    @property
    def macro(self) -> Dict[str, float]:
        tp = torch.sum(self.tp)
        fp = torch.sum(self.fp)
        tn = torch.sum(self.tn)
        fn = torch.sum(self.fn)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore = 2 * precision * recall / (precision + recall)

        return {
            'fscore': fscore.item(),
            'precision': precision.item(),
            'recall': recall.item(),
        }
