from __future__ import annotations
import numpy as np

Array = np.ndarray


def mse(y_true: Array, y_pred: Array) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: Array, y_pred: Array) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true: Array, y_pred: Array, eps: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error in percentage.
    Handles zero targets by adding a tiny epsilon denominator.
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)