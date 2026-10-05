import torch
from torchmetrics.regression import MeanAbsoluteError
from sklearn.metrics import roc_auc_score, r2_score


def compute_metric(outputs: torch.Tensor, labels: torch.Tensor, metric: str):
    if metric == "mae":
        score = - MeanAbsoluteError()(outputs.flatten(), labels).item()
    elif metric == "auc":
        outputs = outputs[:, 1] - outputs[:, 0]
        score = roc_auc_score(labels.cpu().numpy(), outputs.cpu().numpy())
    elif metric == "r2":
        score = r2_score(labels.cpu().numpy(), outputs.cpu().numpy())
    else:
        raise NotImplementedError(metric)
    return score
