import torch
import torch.nn.functional as F
from torchmetrics.regression import (
    MeanSquaredError,
    MeanAbsoluteError,
)
from sklearn.metrics import roc_auc_score, r2_score

def compute_metric(outputs: torch.Tensor, labels: torch.Tensor, metric: str):
    if metric == "rmse":
        score = - MeanSquaredError(squared=False)(outputs.flatten(), labels).item()
    elif metric == "mae":
        score = - MeanAbsoluteError()(outputs.flatten(), labels).item()
    elif metric == "hr@1":
        score = torch.mean((outputs.argmax(dim=1) == labels).to(torch.float)).item()
    elif metric == "retrieval_logloss":
        score = - F.cross_entropy(outputs, labels)  # nn.()(outputs, labels).item()
    # Metrics for comparison with relfm
    elif metric == "auc":
        outputs = outputs[:, 1] - outputs[:, 0]
        score = roc_auc_score(labels.cpu().numpy(), outputs.cpu().numpy())
    elif metric == "r2":
        score = r2_score(labels.cpu().numpy(), outputs.cpu().numpy())
    else:
        raise NotImplementedError(metric)
    return score