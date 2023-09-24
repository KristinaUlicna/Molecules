import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def normalise_pIC50_value(
    y: float | torch.Tensor, 
    y_min: float | torch.Tensor, 
    y_max: float | torch.Tensor,
) -> float | torch.Tensor:
    return ( y - y_min ) / ( y_max - y_min )


def denormalise_pIC50_value(
    y_norm: float | torch.Tensor, 
    y_min: float | torch.Tensor,
    y_max: float | torch.Tensor,
) -> float | torch.Tensor:
    return y_norm * ( y_max - y_min ) + y_min
    

def rmse_metric(
    y_true: torch.Tensor, 
    y_pred: torch.Tensor, 
    y_min: torch.Tensor = None,
    y_max: torch.Tensor = None,
) -> torch.Tensor:
    """Calculate the RMSE metric between true & predicted value."""
    if y_min is not None and y_max is not None:
        y_t = denormalise_pIC50_value(y_true, y_min, y_max)
        y_p = denormalise_pIC50_value(y_pred, y_min, y_max)
        rmse = torch.sqrt(torch.mean((y_p-y_t)**2))
    else:
        rmse = torch.sqrt(torch.mean((y_pred-y_true)**2))
    return rmse


def accuracy_metric(
    y_true: torch.Tensor, 
    y_pred: torch.Tensor, 
) -> torch.Tensor:
    """Calculate the accuracy between true & predicted value."""
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    return torch.Tensor([acc])


def f1_score_metric(
    y_true: torch.Tensor, 
    y_pred: torch.Tensor, 
) -> torch.Tensor:
    """Calculate the F1-score between true & predicted value."""
    f1 = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred)
    return torch.Tensor([f1[2]])
