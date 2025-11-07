from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from scipy.optimize import minimize

from activations import Activation, init_weights
from metrics import mse, mape

Array = np.ndarray


class MLPRegressor:
    """
    Fully-connected MLP for regression with:
    - L hidden layers (>=1) + 1 linear output unit
    - Differentiable activations (tanh/sigmoid/relu/elu)
    - L2 regularization on weights only (not biases)
    - Full-batch optimization via scipy.optimize (L-BFGS-B by default)

    Parameters are packed into a 1D vector for SciPy. Backprop is implemented manually.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        activation: str = "tanh",
        l2: float = 0.0,
        seed: int = 0,
    ):
        assert len(hidden_layers) >= 1, "At least one hidden layer is required."
        self.input_dim = int(input_dim)
        self.hidden_layers = list(map(int, hidden_layers))
        self.output_dim = 1
        self.activation = Activation(activation)
        self.act_name = activation.lower()
        self.l2 = float(l2)
        self.rng = np.random.default_rng(seed)

        # Parameters (weights and biases) are stored after fit() as theta_ plus shapes for (W,b)
        self.theta_: Optional[Array] = None
        self.shapes_: List[Tuple[Tuple[int, int], Tuple[int]]] = []
        self.sizes_: List[int] = []
        self._init_param_layout()

        # History captured via SciPy callback
        self.history_: Dict[str, list] = {"obj": [], "grad_norm": []}

    # ----- Parameter packing / unpacking -----
    def _init_param_layout(self):
        layer_dims = [self.input_dim] + self.hidden_layers + [self.output_dim]
        shapes = []
        for din, dout in zip(layer_dims[:-1], layer_dims[1:]):
            shapes.append(((din, dout), (dout,)))  # (W_shape, b_shape)
        self.shapes_ = shapes
        self.sizes_ = [ws[0][0] * ws[0][1] + ws[1][0] for ws in shapes]

    def _pack(self, params: List[Tuple[Array, Array]]) -> Array:
        return np.concatenate([W.ravel() for (W, b) in params] + [b.ravel() for (W, b) in params])

    def _unpack(self, theta: Array) -> List[Tuple[Array, Array]]:
        params: List[Tuple[Array, Array]] = []
        ptr = 0
        # First read all W then all b to match _pack
        Ws: List[Array] = []
        bs: List[Array] = []
        for (W_shape, b_shape) in self.shapes_:
            size_W = W_shape[0] * W_shape[1]
            Ws.append(theta[ptr: ptr + size_W].reshape(W_shape))
            ptr += size_W
        for (W_shape, b_shape) in self.shapes_:
            size_b = b_shape[0]
            bs.append(theta[ptr: ptr + size_b].reshape(b_shape))
            ptr += size_b
        for W, b in zip(Ws, bs):
            params.append((W, b))
        return params

    def _init_theta(self) -> Array:
        params: List[Tuple[Array, Array]] = []
        layer_dims = [self.input_dim] + self.hidden_layers + [self.output_dim]
        for din, dout in zip(layer_dims[:-1], layer_dims[1:]):
            W = init_weights(din, dout, self.act_name, self.rng)
            b = np.zeros(dout, dtype=float)
            params.append((W, b))
        return self._pack(params)

    # ----- Forward / backward -----
    def _forward(self, X: Array, params: List[Tuple[Array, Array]]):
        """
        Returns:
          - zs: list of pre-activations per layer (hidden layers + output linear)
          - activs: list of activations including input as activs[0]
        """
        activs = [X]
        zs = []
        L = len(params)
        for l, (W, b) in enumerate(params):
            z = activs[-1] @ W + b  # (N, dout)
            zs.append(z)
            if l < L - 1:
                a = self.activation(z)
            else:
                a = z  # linear output
            activs.append(a)
        return zs, activs

    def _loss_and_grad(self, theta: Array, X: Array, y: Array) -> Tuple[float, Array]:
        """
        Computes MSE + L2 and its gradient wrt theta.
        """
        params = self._unpack(theta)
        N = X.shape[0]
        zs, activs = self._forward(X, params)
        y_pred = activs[-1].reshape(-1)

        # Loss
        err = y_pred - y
        data_loss = float(np.mean(err**2))
        reg_loss = self.l2 * sum((W**2).sum() for (W, b) in params)
        loss = data_loss + reg_loss

        # Backprop
        grads_W: List[Array] = []
        grads_b: List[Array] = []

        # dL/dyhat = 2/N * (yhat - y)
        delta = (2.0 / N) * err.reshape(-1, 1)  # shape (N, 1) for last linear layer

        L = len(params)
        for l in reversed(range(L)):
            W, b = params[l]
            a_prev = activs[l]  # (N, din)

            # Gradients for current layer
            gW = a_prev.T @ delta + 2.0 * self.l2 * W  # L2 only on weights
            gb = delta.sum(axis=0)

            grads_W.append(gW)
            grads_b.append(gb)

            if l > 0:
                # Propagate to previous layer: delta_prev = (delta @ W^T) * phi'(z_prev)
                delta = (delta @ W.T) * self.activation.grad(zs[l - 1])

        grads_W.reverse()
        grads_b.reverse()

        # Pack gradient same layout as theta: all W then all b
        grad_vec = np.concatenate([gW.ravel() for gW in grads_W] + [gb.ravel() for gb in grads_b])
        return loss, grad_vec

    # ----- Public API -----
    def fit(
        self,
        X: Array,
        y: Array,
        max_iter: int = 500,
        tol: float = 1e-6,
        method: str = "L-BFGS-B",
        verbose: bool = False,
    ):
        """
        Optimize parameters using SciPy's minimize with analytic gradient.
        This version avoids recomputing loss/grad in the callback (saves ~2x time).
        """
        theta0 = self._init_theta()
        self.history_ = {"obj": [], "grad_norm": []}

        # Cache last (f, g) computed by fun so callback can log without recomputing
        self._last_f = None
        self._last_g = None

        def fun_and_jac(th):
            f, g = self._loss_and_grad(th, X, y)
            self._last_f, self._last_g = f, g
            return f, g

        def cb(th):
            # Use cached values from the last fun_and_jac call
            self.history_["obj"].append(float(self._last_f))
            self.history_["grad_norm"].append(float(np.linalg.norm(self._last_g)))

        res = minimize(
            fun=fun_and_jac,
            x0=theta0,
            method=method,
            jac=True,
            callback=cb,
            options={"maxiter": max_iter, "gtol": tol, "disp": verbose},
        )
        self.theta_ = res.x
        self.opt_result_ = res
        return self

    def predict(self, X: Array) -> Array:
        assert self.theta_ is not None, "Call fit() first."
        params = self._unpack(self.theta_)
        _, activs = self._forward(X, params)
        return activs[-1].reshape(-1)

    def evaluate(self, X: Array, y: Array) -> Dict[str, float]:
        y_pred = self.predict(X)
        return {"mse": mse(y, y_pred), "mape": mape(y, y_pred)}