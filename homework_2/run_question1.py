"""Question 1: Gaussian taxi-duration regression and probability evaluation.

Run: python run_question1.py
"""
from scipy.special import ndtr
from sklearn.linear_model import LinearRegression
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from training_utils import SEED, splits, train_network, predict_tensor

ROOT = Path(__file__).resolve().parent


class GaussianMLP(nn.Module):
    """Part 1(c): predict mean and positive standard deviation using the 1(a) links."""
    def __init__(self, d_in, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 2))

    def forward(self, X):
        raw = self.net(X)
        return torch.stack([raw[:, 0], F.softplus(raw[:, 1]) + 1e-3], dim=1)


def gaussian_loss(output, target):
    """Part 1(b): Gaussian negative log-likelihood used to fit the network."""
    mu, sigma = output[:, 0], output[:, 1]
    return (torch.log(sigma) + 0.5 * ((target - mu) / sigma) ** 2 + 0.5 * np.log(2*np.pi)).mean()


def taxi_metrics(y, mu, sigma):
    """Parts 1(d/e): normal-CDF probabilities and held-out scores."""
    y, mu, sigma = [np.asarray(a, dtype=np.float64) for a in (y, mu, sigma)]
    mu, sigma = np.broadcast_arrays(mu, sigma, y)[:2]
    p = ndtr((45 - mu) / sigma)
    z = (y < 45).astype(float)
    return {'gaussian_nll': float(np.mean(np.log(sigma) + 0.5*((y-mu)/sigma)**2 + 0.5*np.log(2*np.pi))),
            'rmse_minutes': float(np.sqrt(np.mean((y-mu)**2))),
            'brier': float(np.mean((p-z)**2)),
            'accuracy': float(np.mean((p >= 0.5) == z))}


def run_taxi(output, epochs, data_dir=ROOT):
    """Run the data split, model fitting, baselines, and evaluation for Question 1."""
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(data_dir / 'taxi_trips.parquet')
    features = [c for c in data if c != 'trip_duration']
    X, y = data[features].to_numpy(dtype=np.float32), data.trip_duration.to_numpy(dtype=np.float64)
    assert np.isfinite(X).all() and np.isfinite(y).all()
    train, val, test = splits(len(data))
    center, scale = float(y[train].mean()), float(y[train].std())
    tensors = [torch.from_numpy(X[i]) for i in (train, val, test)]
    targets = [torch.tensor((y[i]-center)/scale, dtype=torch.float32) for i in (train, val)]
    print('Question 1: fitting Gaussian MLP...', flush=True)
    torch.manual_seed(SEED)
    model = GaussianMLP(X.shape[1])
    history, best_epoch = train_network(model, [tensors[0]], targets[0], [tensors[1]], targets[1],
                                        gaussian_loss, epochs=epochs)
    # Models and early-stopping epoch are now fixed; evaluate the test split once.
    pred = predict_tensor(model, [tensors[2]]).numpy().astype(np.float64)
    mu, sigma = center + scale * pred[:, 0], scale * pred[:, 1]
    linear = LinearRegression().fit(X[train], y[train])
    residual_scale = float(np.sqrt(np.mean((linear.predict(X[train]) - y[train])**2)))
    results = {'constant_gaussian': taxi_metrics(y[test], center, scale),
               'linear_gaussian': taxi_metrics(y[test], linear.predict(X[test]), residual_scale),
               'gaussian_mlp': taxi_metrics(y[test], mu, sigma)}
    summary = {'rows': len(data), 'features': features, 'split_sizes': [len(train), len(val), len(test)],
               'seed': SEED, 'best_epoch': best_epoch, 'epochs_run': len(history),
               'train_target_mean': center, 'train_target_std': scale,
               'results': results}
    (output/'metrics.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--data-dir', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results_q1')
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error('--epochs must be positive')
    torch.set_num_threads(1)
    run_taxi(args.output_dir, args.epochs, data_dir=args.data_dir)


if __name__ == '__main__':
    main()
