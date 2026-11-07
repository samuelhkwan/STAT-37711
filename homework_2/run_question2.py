"""Question 2: retail target encoding, embeddings, and their combination.

Run: python run_question2.py
"""
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
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


def target_encode(train_cat, train_y, others, smoothing=20.0):
    """Part 2(b): smoothed raw-quantity means with 5-fold cross-fitting.

    Unknown categories use the relevant training-fold global target mean.
    Validation/test maps use the entire training split, never their labels.
    """
    def encode(fit_X, fit_y, apply_X):
        global_mean = float(np.mean(fit_y))
        encoded = np.empty(apply_X.shape, dtype=np.float64)
        for j in range(fit_X.shape[1]):
            table = pd.DataFrame({'category': fit_X[:, j], 'target': fit_y}).groupby('category').target.agg(['sum', 'count'])
            means = (table['sum'] + smoothing*global_mean)/(table['count']+smoothing)
            encoded[:, j] = pd.Series(apply_X[:, j]).map(means).fillna(global_mean).to_numpy()
        return encoded
    oof = np.empty(train_cat.shape, dtype=np.float64)
    for fit_idx, hold_idx in KFold(5, shuffle=True, random_state=SEED).split(train_cat):
        oof[hold_idx] = encode(train_cat[fit_idx], train_y[fit_idx], train_cat[hold_idx])
    return [oof] + [encode(train_cat, train_y, other) for other in others]


class RetailMLP(nn.Module):
    """Parts 2(c/d): categorical embeddings, optionally combined with target encoding."""
    def __init__(self, cardinalities, use_target_encoding=False, embedding_dim=16):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n+1, embedding_dim, padding_idx=0) for n in cardinalities])
        self.use_target_encoding = use_target_encoding
        self.net = nn.Sequential(nn.Linear(2*embedding_dim + (2 if use_target_encoding else 0), 64),
                                 nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, categories, target_features):
        inputs = [embedding(categories[:, j]) for j, embedding in enumerate(self.embeddings)]
        if self.use_target_encoding:
            inputs.append(target_features)
        return self.net(torch.cat(inputs, dim=1)).squeeze(1)


def retail_metrics(y, log_pred):
    log_pred = np.maximum(log_pred, 0)
    pred = np.expm1(log_pred)
    return {'msle': float(np.mean((np.log1p(y)-log_pred)**2)),
            'mae': float(np.mean(np.abs(y-pred))),
            'rmse': float(np.sqrt(np.mean((y-pred)**2)))}


def run_retail(output, epochs, data_dir=ROOT):
    """Run the shared split, all retail models, and comparisons for Question 2."""
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(data_dir/'retail.csv', dtype={'StockCode': str, 'CustomerID': str})
    categories = data[['StockCode','CustomerID']].to_numpy()
    y = data.Quantity.to_numpy(dtype=np.float64)
    assert (y > 0).all() and not data.isna().any().any()
    train, val, test = splits(len(data))
    cat_splits = [categories[i] for i in (train, val, test)]
    print('Question 2: generating cross-fitted target encodings...', flush=True)
    encoded = target_encode(cat_splits[0], y[train], cat_splits[1:])
    # log-compress the heavy-tailed raw-quantity means; fit scale on training only.
    scaler = StandardScaler().fit(np.log1p(encoded[0]))
    encoded = [scaler.transform(np.log1p(a)).astype(np.float32) for a in encoded]
    ids = [np.zeros(a.shape, dtype=np.int64) for a in cat_splits]
    cardinalities = []
    for j in range(2):
        mapping = {key: i+1 for i, key in enumerate(np.unique(cat_splits[0][:,j]))}
        cardinalities.append(len(mapping))
        for numeric, values in zip(ids, cat_splits):
            numeric[:,j] = pd.Series(values[:,j]).map(mapping).fillna(0).to_numpy(dtype=np.int64)
    # Fixed boosting capacity; no test-informed hyperparameter selection.
    target_model = HistGradientBoostingRegressor(loss='squared_error', learning_rate=.05,
        max_iter=200, max_leaf_nodes=15, l2_regularization=10., early_stopping=False, random_state=SEED)
    target_model.fit(encoded[0], np.log1p(y[train]))
    models, selected_epochs = {}, {}
    inputs = [(torch.from_numpy(i), torch.from_numpy(e)) for i,e in zip(ids,encoded)]
    targets = [torch.tensor(np.log1p(y[i]), dtype=torch.float32) for i in (train, val)]
    for name, combined in [('embedding',False), ('combined',True)]:
        print(f'Question 2: fitting {name} network...', flush=True)
        torch.manual_seed(SEED)
        model = RetailMLP(cardinalities, use_target_encoding=combined)
        _, best_epoch = train_network(model, inputs[0], targets[0], inputs[1], targets[1],
                                            F.mse_loss, epochs=epochs, batch_size=2048, patience=6)
        models[name], selected_epochs[name] = model, best_epoch
    # Final test evaluation, after both networks' early-stopping decisions.
    preds = {'target_encoding': target_model.predict(encoded[2])}
    for name, model in models.items():
        preds[name] = predict_tensor(model, inputs[2]).numpy().astype(float)
    results = {name: retail_metrics(y[test], pred) for name, pred in preds.items()}
    summary = {'rows':len(data), 'split_sizes':[len(train),len(val),len(test)], 'seed':SEED,
        'cardinalities_full':data[['StockCode','CustomerID']].nunique().to_dict(),
        'target_smoothing':20, 'cross_fit_folds':5, 'embedding_dimension':16,
        'best_epochs':selected_epochs, 'test_results':results}
    (output/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--data-dir', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results_q2')
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error('--epochs must be positive')
    torch.set_num_threads(1)
    run_retail(args.output_dir, args.epochs, data_dir=args.data_dir)


if __name__ == '__main__':
    main()
