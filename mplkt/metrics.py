import math

import numpy as np
import torch
import torch.nn.functional as F


def masked_bce(pred, target, mask):
    pred = pred.clamp(1e-6, 1.0 - 1e-6)
    target = target.float()
    mask = mask.bool()
    if mask.sum() == 0:
        return pred.sum() * 0.0
    loss = F.binary_cross_entropy(pred[mask], target[mask], reduction="mean")
    return loss


def collect_masked(pred, target, mask):
    mask = mask.bool()
    if mask.sum() == 0:
        return np.asarray([]), np.asarray([])
    return (
        pred.detach().cpu()[mask.detach().cpu()].numpy(),
        target.detach().cpu()[mask.detach().cpu()].numpy(),
    )


def auc_acc(pred_values, target_values):
    if len(target_values) == 0:
        return math.nan, math.nan
    acc = float(((pred_values >= 0.5).astype(np.int64) == target_values.astype(np.int64)).mean())
    if len(np.unique(target_values)) < 2:
        auc = math.nan
    else:
        auc = float(_binary_auc(pred_values, target_values))
    return auc, acc


def _binary_auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    n = len(sorted_scores)
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[start:end] = avg_rank
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    pos_rank_sum = original_ranks[labels == 1].sum()
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
