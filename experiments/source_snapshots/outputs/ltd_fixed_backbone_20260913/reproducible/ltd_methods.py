"""Fixed-action adaptations: frozen predictions; only the router is trained."""
import torch
from torch.nn import functional as F


def output_count(method):
    return 4 if method == 'four_state' else 2 if method in ('css', 'ova') else 1


def loss_values(raw, states, method, cost):
    correct = torch.stack((states[:, 0] + states[:, 2], states[:, 0] + states[:, 1]), 1)
    if method == 'four_state':
        return -(states * F.log_softmax(raw, dim=1)).sum(1)
    if method == 'css':
        # Cost-sensitive softmax over two fixed actions, not K trainable classes.
        return -(correct * F.log_softmax(raw, dim=1)).sum(1)
    if method == 'ova':
        # Sum the independent binary losses as in the original OvA loss.
        return F.binary_cross_entropy_with_logits(raw, correct, reduction='none').sum(1)
    if method == 'posthoc':
        errors = 1 - correct
        return errors[:, 0] * F.softplus(-raw[:, 0]) + (errors[:, 1] + cost) * F.softplus(raw[:, 0])
    raise ValueError(method)


def scores(raw, method):
    raw = torch.as_tensor(raw)
    if method == 'four_state':
        p = raw.softmax(1)
        return (p[:, 1] - p[:, 2]).numpy()
    if method == 'css':
        p = raw.softmax(1)
        return (p[:, 1] - p[:, 0]).numpy()
    if method == 'ova':
        p = raw.sigmoid()
        return (p[:, 1] - p[:, 0]).numpy()
    if method == 'posthoc':
        return raw[:, 0].numpy()
    raise ValueError(method)
