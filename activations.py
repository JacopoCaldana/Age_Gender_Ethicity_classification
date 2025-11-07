
# Simple, differentiable activation functions and their derivatives.

from __future__ import annotations
import numpy as np
from typing import Callable, Tuple

Array = np.ndarray


class Activation:
    """
    Encapsulates an activation function and its derivative with respect to the pre-activation z.
    Each function is vectorized over inputs.
    """
    def __init__(self, name: str):
        name = name.lower()
        if name == "tanh":
            self.name = "tanh"
            self.func = np.tanh
            self.deriv = lambda z: 1.0 - np.tanh(z) ** 2
        elif name == "sigmoid":
            self.name = "sigmoid"
            self.func = lambda z: 1.0 / (1.0 + np.exp(-z))
            self.deriv = lambda z: self.func(z) * (1.0 - self.func(z))
        elif name == "relu":
            self.name = "relu"
            self.func = lambda z: np.maximum(0.0, z)
            self.deriv = lambda z: (z > 0.0).astype(z.dtype)
        elif name == "elu":
            self.name = "elu"
            alpha = 1.0
            self.func = lambda z: np.where(z > 0.0, z, alpha * (np.exp(z) - 1.0))
            self.deriv = lambda z: np.where(z > 0.0, 1.0, np.exp(z))
        else:
            raise ValueError(f"Unsupported activation '{name}'. Choose among: tanh, sigmoid, relu, elu")

    def __call__(self, z: Array) -> Array:
        return self.func(z)

    def grad(self, z: Array) -> Array:
        return self.deriv(z)


def xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """Xavier/Glorot uniform initialization (good for tanh/sigmoid)."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """He normal initialization (good for ReLU)."""
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=(fan_in, fan_out))


def init_weights(fan_in: int, fan_out: int, act_name: str, rng: np.random.Generator) -> np.ndarray:
    """Select initialization based on activation."""
    if act_name.lower() in ("relu", "elu"):
        return he_init(fan_in, fan_out, rng)
    return xavier_init(fan_in, fan_out, rng)