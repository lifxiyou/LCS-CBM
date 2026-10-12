import torch
import numpy as np
import sklearn.metrics


def compute_concept_metric(c_prob, c_true, mean=True):

    if isinstance(c_prob, torch.Tensor):
        c_prob = c_prob.cpu().detach().numpy()
    c_pred = (c_prob >= 0.5).astype(np.int32)

    if isinstance(c_true, torch.Tensor):
        c_true = c_true.cpu().detach().numpy()
    # Doing the following transformation for when labels are not fully certain
    c_true = (c_true >= 0.5).astype(np.int32)

    c_acc, c_f1 = [], []
    for i in range(c_true.shape[-1]):
        true_vars = c_true[:, i]
        if (true_vars == 0).all():
            continue
        pred_vars = c_pred[:, i]
        c_acc.append(sklearn.metrics.accuracy_score(true_vars, pred_vars))
        c_f1.append(sklearn.metrics.f1_score(true_vars, pred_vars, average="macro"))

    c_acc, c_f1 = torch.tensor(c_acc), torch.tensor(c_f1)
    if mean:
        return c_acc.mean(), c_f1.mean()
    else:
        return c_acc, c_f1


def compute_task_metric(y_logit, y_true):
    if isinstance(y_logit, torch.Tensor):
        y_pred = y_logit.argmax(dim=-1).cpu().detach().numpy()
    else:
        y_pred = np.array(y_logit).argmax(axis=-1)
    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.cpu().detach().numpy()
    else:
        y_true_np = np.asarray(y_true)

    # sklearn 需要 numpy，且 macro-F1 在缺失类别时用 zero_division=0 避免未定义
    y_f1 = sklearn.metrics.f1_score(
        y_true_np, y_pred, average="macro", zero_division=0
    )
    y_acc = sklearn.metrics.accuracy_score(y_true_np, y_pred)

    return float(y_acc), float(y_f1)


def compute_metric(c_prob, y_logit, c_true, y_true):
    c_acc, c_f1 = compute_concept_metric(c_prob, c_true)
    y_acc, y_f1 = compute_task_metric(y_logit, y_true)
    return (c_acc, c_f1), (y_acc, y_f1)