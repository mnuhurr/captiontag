import math
import torch
import torch.nn.functional as F

from typing import Optional


def masked_loss(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """function to compute token prediction loss with mask"""
    loss = F.cross_entropy(y_pred.permute(0, 2, 1), y_true, reduction='none')
    mask = (mask == 0).to(loss.dtype)
    return torch.sum(loss * mask) / torch.sum(mask).to(loss.dtype)


def step_lr(step: int, warmup_steps: int = 4000):
    """
    :param step:
    :param warmup_steps:
    :return: learning rate scaling factor
    """
    arg1 = torch.tensor(1 / math.sqrt(step)) if step > 0 else torch.tensor(float('inf'))
    arg2 = torch.tensor(step * warmup_steps**-1.5)
    return math.sqrt(warmup_steps) * torch.minimum(arg1, arg2)



@torch.no_grad()
def batch_weights(y_true: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    batch_size = y_true.size(0)

    if padding_mask is not None:
        n_timesteps = torch.sum(padding_mask == 0)
    else:
        n_timesteps = y_true.numel()

    label_counts = torch.sum(y_true, dim=(0, 1))
    weights = torch.ones_like(label_counts, dtype=torch.float32)
    weights[label_counts > 0] = n_timesteps / (2 * label_counts[label_counts > 0])

    weights = weights.unsqueeze(0).unsqueeze(1)

    pos_weight = weights.broadcast_to(y_true.shape).to(y_true.device)
    batch_weight = 1 / (2 - 1 / pos_weight)
    batch_weight[y_true == 1] *= pos_weight[y_true == 1]

    return weights

