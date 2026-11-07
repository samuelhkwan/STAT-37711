"""Run Question 3 from the supplied CSV files; no test labels are used.

Usage: python run_question3.py
All paths default to this file's folder, so it can be run from any directory.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataloader import HW2Dataset
from models import RandomKitchenSinks, MLP, sweep_rks_model, mlp_sweep
from plotting import plot_surface_2d


def save_sweep(rows, parameter, prefix, output):
    with (output / f"{prefix}_sweep.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[parameter, "train_mse", "val_mse"])
        writer.writeheader()
        writer.writerows(rows)
    for metric, label in [("train_mse", "Training"), ("val_mse", "Validation")]:
        fig, ax = plt.subplots()
        ax.plot([row[parameter] for row in rows], [row[metric] for row in rows], "o-")
        ax.set(xlabel=parameter, ylabel="MSE", title=f"{prefix.upper()}: {label} loss")
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / f"{prefix}_{metric}.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    folder = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=folder)
    parser.add_argument("--output-dir", type=Path, default=folder / "results_q3")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--device", default="cpu", help="cpu, mps, or cuda")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)  # Small networks run faster without thread overhead.
    ds = HW2Dataset.from_csv(*(str(args.data_dir / name) for name in
                              ("train.csv", "val.csv", "test.csv")))
    fig, ax = plt.subplots()
    points = ax.scatter(ds.train.X[:, 0], ds.train.X[:, 1], c=ds.train.y, s=16)
    ax.set(xlabel="x1", ylabel="x2", xlim=(-3, 3), ylim=(-3, 3), title="Training observations")
    fig.colorbar(points, ax=ax, label="y")
    fig.tight_layout()
    fig.savefig(output / "training_data.png", dpi=200)
    plt.close(fig)

    rks_kwargs = dict(sigma=1.0, ridge_lambda=1e-3, random_state=42)
    mlp_kwargs = dict(lr=1e-3, batch_size=128, activation="tanh", device=args.device,
                      num_epochs=args.epochs, random_state=42)
    print("Sweeping RKS feature counts...", flush=True)
    rks, _, best_D = sweep_rks_model([16, 32, 64, 128, 256, 512, 1024],
                                    ds.train.X, ds.train.y, ds.val.X, ds.val.y, **rks_kwargs)
    save_sweep(rks.sweep_results_, "D", "rks", output)
    print("RKS sweep:", rks.sweep_results_, flush=True)
    print("Sweeping MLP hidden widths...", flush=True)
    mlp, _, best_width = mlp_sweep([16, 32, 64, 128, 256],
                                  ds.train.X, ds.train.y, ds.val.X, ds.val.y, **mlp_kwargs)
    save_sweep(mlp.sweep_results_, "d_hidden", "mlp", output)
    print("MLP sweep:", mlp.sweep_results_, flush=True)

    # Refit the chosen configurations on training data only. Validation data
    # remain separate so their reported MSE is not a training-set measurement.
    rks = RandomKitchenSinks(D=best_D, **rks_kwargs).fit(ds.train.X, ds.train.y)
    mlp = MLP(d_in=ds.train.X.shape[1], d_hidden=best_width, **mlp_kwargs).fit(ds.train.X, ds.train.y)
    summary = {"seed": 42, "training_rows": len(ds.train.X),
               "validation_rows": len(ds.val.X), "test_rows": len(ds.test.X),
               "rks_settings": {"D": best_D, **rks_kwargs},
               "mlp_settings": {"d_hidden": best_width, **mlp_kwargs}}
    for name, model in [("rks", rks), ("mlp", mlp)]:
        summary[name] = {
            "train_mse": float(np.mean((model.predict(ds.train.X) - ds.train.y) ** 2)),
            "val_mse": float(np.mean((model.predict(ds.val.X) - ds.val.y) ** 2)),
        }
        prediction = model.predict(ds.test.X)
        if prediction.shape != (len(ds.test.X),) or not np.isfinite(prediction).all():
            raise RuntimeError(f"Invalid {name} test predictions.")
        # Headerless: exactly one prediction per row, in original test order.
        np.savetxt(output / f"{name}_test_predictions.csv", prediction, delimiter=",")
        plot_surface_2d(model.predict, (-3, 3), (-3, 3), title=f"{name.upper()} fitted surface",
                        filename=str(output / f"{name}_surface.png"))
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved results to {output}", flush=True)


if __name__ == "__main__":
    main()
