from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict

Array = np.ndarray


class StandardScaler:
    """Minimal standard scaler for features."""
    def __init__(self):
        self.mean_: Array | None = None
        self.std_: Array | None = None

    def fit(self, X: Array) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X: Array) -> Array:
        return (X - self.mean_) / self.std_

    def inverse_transform(self, Xs: Array) -> Array:
        return Xs * self.std_ + self.mean_


class ScalarStandardizer:
    """Standardize a 1D target; useful for stable optimization."""
    def __init__(self):
        self.mean_: float | None = None
        self.std_: float | None = None

    def fit(self, y: Array) -> "ScalarStandardizer":
        y = y.reshape(-1)
        self.mean_ = float(y.mean())
        std = float(y.std())
        self.std_ = 1.0 if std == 0.0 else std
        return self

    def transform(self, y: Array) -> Array:
        return (y.reshape(-1) - self.mean_) / self.std_

    def inverse_transform(self, ys: Array) -> Array:
        return ys.reshape(-1) * self.std_ + self.mean_


def load_age_regression_csv(path: str) -> Tuple[Array, Array]:
    """
    The CSV has ResNet features in all columns except the last ('gt') which is the age [0,100].
    Headers example: feat1, feat2, ..., gt
    """
    df = pd.read_csv(path)
    if "gt" in df.columns:
        y = df["gt"].to_numpy(dtype=float)
        X = df.drop(columns=["gt"]).to_numpy(dtype=float)
    else:
        # Fallback: assume last column is gt
        y = df.iloc[:, -1].to_numpy(dtype=float)
        X = df.iloc[:, :-1].to_numpy(dtype=float)
    return X, y


def train_test_split(n: int, test_size: float, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns train_indices, test_indices with stratification not required (regression).
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(np.round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def kfold_indices(n: int, k: int, shuffle: bool = True, seed: int = 0):
    """
    Yields (train_idx, val_idx) for each fold. Balanced partitions by index slicing.
    """
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, val_idx