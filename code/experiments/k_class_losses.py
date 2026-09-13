"""K real task classes plus expert correctness; original zero-cost CSS/OvA losses."""
import torch
from torch.nn import functional as F


def labels_from_records(records, task, endpoint):
    if task == 'cub':
        k = 200
        labels = [int(row['ground_truth']['category_id']) - 1 for row in records]
    elif task == 'nlvr2':
        k = 2
        classes = {'False': 0, 'True': 1}
        labels = [classes[row['ground_truth']] for row in records]
    else:
        raise ValueError(f'{task} is not a single-label classification task; no invented K-label mapping')
    y = torch.tensor(labels, dtype=torch.long)
    if not bool(((y >= 0) & (y < k)).all()):
        raise ValueError('Task class label out of range')
    expert_correct = torch.tensor([row['large_labels'][endpoint]['correct'] for row in records], dtype=torch.float32)
    return k, y, expert_correct


def k_class_loss(logits, labels, expert_correct, method):
    k = logits.shape[1] - 1
    if method == 'ova':
        target = torch.cat((F.one_hot(labels, k).to(logits.dtype), expert_correct[:, None]), 1)
        return F.binary_cross_entropy_with_logits(logits, target, reduction='none').sum(1)
    if method == 'css':
        logp = F.log_softmax(logits, 1)
        return -logp.gather(1, labels[:, None])[:, 0] - expert_correct * logp[:, k]
    raise ValueError(method)


def native_scores(logits, method):
    raw = torch.as_tensor(logits)
    p = raw.sigmoid() if method == 'ova' else raw.softmax(1)
    return (p[:, -1] - p[:, :-1].max(1).values).numpy()


def native_records(records, logits, task):
    predicted = torch.as_tensor(logits[:, :-1]).argmax(1).tolist()
    result = []
    for row, index in zip(records, predicted):
        prediction = index + 1 if task == 'cub' else ['False', 'True'][index]
        label = dict(row['small_label'])
        label.update(prediction=prediction,correct=prediction==label['target'],valid_output=True,error_reason=None)
        updated = dict(row)
        updated['small_label'] = label
        result.append(updated)
    return result
