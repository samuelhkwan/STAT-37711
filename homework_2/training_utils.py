"""Shared splitting, training, and prediction helpers for Questions 1 and 2."""
import copy
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

SEED = 42


def splits(n):
    train, remaining = train_test_split(np.arange(n), test_size=0.3, random_state=SEED)
    val, test = train_test_split(remaining, test_size=0.5, random_state=SEED)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    return train, val, test


def train_network(model, train_inputs, train_y, val_inputs, val_y, loss_fn,
                  epochs=60, batch_size=2048, lr=1e-3, patience=8, weight_decay=1e-4):
    """Minibatch fitting with validation early stopping and best-state restoration."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = torch.Generator().manual_seed(SEED)
    best, best_state, stale = float('inf'), None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(train_y), generator=rng).split(batch_size):
            optimizer.zero_grad()
            loss = loss_fn(model(*(x[idx] for x in train_inputs)), train_y[idx])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            validation_loss = loss_fn(predict_tensor(model, val_inputs), val_y).item()
        history.append({'epoch': epoch + 1, 'train_loss': total / len(train_y), 'val_loss': validation_loss})
        if validation_loss < best - 1e-6:
            best, best_state, stale = validation_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if (epoch + 1) % 10 == 0:
            print(f'  epoch {epoch + 1}: train={total/len(train_y):.5f}, val={validation_loss:.5f}', flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return history, min(history, key=lambda row: row['val_loss'])['epoch']


def predict_tensor(model, inputs, batch_size=8192):
    with torch.no_grad():
        return torch.cat([model(*(x[start:start+batch_size] for x in inputs))
                          for start in range(0, len(inputs[0]), batch_size)])
