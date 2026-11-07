"""Random Kitchen Sinks and one-hidden-layer MLP regression for Homework 2."""
from typing import List, Tuple
import numpy as np
import torch
from torch import nn


def _inputs(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("X must be a finite two-dimensional array.")
    return X


def _targets(y, n):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(y) != n or n == 0 or not np.isfinite(y).all():
        raise ValueError("y must contain one finite target per input row.")
    return y


class RandomKitchenSinks:
    """Paired sinusoidal features with ridge regression.

    Minimize ||y - Phi beta||^2 + ridge_lambda * ||beta_without_intercept||^2.
    Features use the unnormalized sine/cosine convention in the assignment.
    """
    def __init__(self, D: int = 1000, sigma: float = 1.0,
                 ridge_lambda: float = 1e-3, random_state: int = 42):
        if not isinstance(D, (int, np.integer)) or D <= 0 or D % 2:
            raise ValueError("D must be a positive even integer (= 2h).")
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be positive and finite.")
        if not np.isfinite(ridge_lambda) or ridge_lambda < 0:
            raise ValueError("ridge_lambda must be nonnegative and finite.")
        self.D, self.sigma = D, sigma
        self.ridge_lambda, self.random_state = ridge_lambda, random_state
        # Input dimension is only known at fit time.
        self.weights_ = self.bias_ = self.coef_ = None

    def _features(self, X: np.ndarray) -> np.ndarray:
        X = _inputs(X)
        if self.weights_ is None:
            raise RuntimeError("Call fit before computing features or predictions.")
        if X.shape[1] != self.weights_.shape[0]:
            raise ValueError("Input dimension differs from the training data.")
        projection = X @ self.weights_ + self.bias_
        phi = np.empty((len(X), self.D + 1))
        phi[:, 0:self.D:2] = np.sin(projection)
        phi[:, 1:self.D:2] = np.cos(projection)
        phi[:, -1] = 1.0  # Unpenalized intercept.
        return phi

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = _inputs(X)
        y = _targets(y, len(X))
        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.normal(0, 1 / self.sigma, size=(self.D // 2, X.shape[1])).T
        self.bias_ = rng.uniform(0, 2 * np.pi, size=self.D // 2)
        phi = self._features(X)
        if self.ridge_lambda == 0:
            self.coef_ = np.linalg.lstsq(phi, y, rcond=None)[0]
        else:
            gram = phi.T @ phi
            gram[np.arange(self.D), np.arange(self.D)] += self.ridge_lambda
            self.coef_ = np.linalg.solve(gram, phi.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Call fit before predict.")
        return self._features(X) @ self.coef_


def sweep_rks_model(sweep_D: List[int], X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray, **model_kwargs) -> Tuple:
    """Return (best_model, best_validation_MSE, best_D).

    Both losses for every candidate are stored in best_model.sweep_results_.
    Other hyperparameters are held fixed via model_kwargs.
    """
    X_val = _inputs(X_val)
    y_val = _targets(y_val, len(X_val))
    best_model, best_val_mse, best_D = None, float("inf"), None
    results = []
    for D in sweep_D:
        model = RandomKitchenSinks(D=D, **model_kwargs).fit(X_train, y_train)
        train_mse = float(np.mean((model.predict(X_train) - np.asarray(y_train).reshape(-1)) ** 2))
        val_mse = float(np.mean((model.predict(X_val) - y_val) ** 2))
        results.append({"D": int(D), "train_mse": train_mse, "val_mse": val_mse})
        if val_mse < best_val_mse:
            best_model, best_val_mse, best_D = model, val_mse, D
    if best_model is None:
        raise ValueError("sweep_D must contain at least one candidate.")
    best_model.sweep_results_ = results
    return best_model, best_val_mse, best_D


class MLP(nn.Module):
    """One-hidden-layer regression network with minibatch Adam and MSE."""
    def __init__(self, d_in: int, d_hidden: int = 128, d_out: int = 1,
                 lr: float = 1e-3, batch_size: int = 128, activation: str = "relu",
                 device: str = "cpu", num_epochs: int = 1000, random_state: int = 42):
        super().__init__()
        if min(d_in, d_hidden, batch_size, num_epochs) <= 0 or lr <= 0:
            raise ValueError("Dimensions, batch size, epochs and learning rate must be positive.")
        if d_out != 1:
            raise ValueError("This regression implementation expects d_out=1.")
        activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid,
                       "gelu": nn.GELU}
        if activation.lower() not in activations:
            raise ValueError(f"activation must be one of {tuple(activations)}.")
        self.device = torch.device(device)
        self.lr, self.batch_size = lr, batch_size
        self.num_epochs, self.random_state = num_epochs, random_state
        self.d_in = d_in
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(random_state)
            self.net = nn.Sequential(nn.Linear(d_in, d_hidden), activations[activation.lower()](),
                                     nn.Linear(d_hidden, d_out))
        self.to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = _inputs(X)
        y = _targets(y, len(X))
        if X.shape[1] != self.d_in:
            raise ValueError("Input dimension differs from d_in.")
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=self.device).reshape(-1, 1)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        generator = torch.Generator().manual_seed(self.random_state)
        self.train()
        for _ in range(self.num_epochs):
            order = torch.randperm(len(X), generator=generator).to(self.device)
            for batch in order.split(self.batch_size):
                optimizer.zero_grad()
                loss = loss_fn(self(X_tensor[batch]), y_tensor[batch])
                loss.backward()
                optimizer.step()
        self.eval()
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _inputs(X)
        self.eval()
        with torch.no_grad():
            inputs = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            # Flat predictions prevent accidental (n, n) broadcasting in MSE.
            return self(inputs).cpu().numpy().reshape(-1)


def mlp_sweep(sweep_values: List[float], X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              parameter: str = "d_hidden", **model_kwargs) -> Tuple:
    """Return (best_model, best_val_MSE, best_value), sweeping one parameter.

    Supported parameters: d_hidden (default), lr, batch_size.
    """
    if parameter not in {"d_hidden", "lr", "batch_size"}:
        raise ValueError("parameter must be d_hidden, lr, or batch_size.")
    X_train, X_val = _inputs(X_train), _inputs(X_val)
    y_val = _targets(y_val, len(X_val))
    best_model, best_val_mse, best_value = None, float("inf"), None
    results = []
    for value in sweep_values:
        kwargs = dict(model_kwargs)
        kwargs[parameter] = value if parameter == "lr" else int(value)
        model = MLP(d_in=X_train.shape[1], **kwargs).fit(X_train, y_train)
        train_mse = float(np.mean((model.predict(X_train) - np.asarray(y_train).reshape(-1)) ** 2))
        val_mse = float(np.mean((model.predict(X_val) - y_val) ** 2))
        results.append({parameter: value, "train_mse": train_mse, "val_mse": val_mse})
        if val_mse < best_val_mse:
            best_model, best_val_mse, best_value = model, val_mse, value
    if best_model is None:
        raise ValueError("sweep_values must contain at least one candidate.")
    best_model.sweep_results_ = results
    return best_model, best_val_mse, best_value
