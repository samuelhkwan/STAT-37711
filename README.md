# STAT 37711 — Machine Learning & AI Part I

Course homework code is organized as `homework_1/`, `homework_2/`, etc.
The supplied datasets are included so the submitted code can run after cloning.

## Setup

Use Python 3.12. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.

## Homework 2

Run from the repository root:

```sh
python homework_2/run_question1.py
python homework_2/run_question2.py
python homework_2/run_question3.py
```

- Question 1 fits a Gaussian taxi-duration MLP and two regression baselines,
  then reports Gaussian NLL, RMSE, Brier score and classification accuracy.
- Question 2 compares target encoding, embeddings and their combination on
  retail quantities, reporting test MSLE, MAE and RMSE.
- Question 3 runs RKS and one-hidden-layer MLP sweeps, refits the selected
  models, and generates the required plots and test predictions.

Results are saved under `homework_2/results_q1/`, `results_q2/`, and
`results_q3/`. The required prediction files are:

```text
homework_2/results_q3/rks_test_predictions.csv
homework_2/results_q3/mlp_test_predictions.csv
```

Each contains 200 headerless predictions in the original test order.
All scripts use seed 42 and default to CPU; a GPU is not required.
The scripts locate their supplied datasets relative to themselves.
Use `--help` for optional training settings and input/output directories.
A saved writeup PDF is included for reference; its LaTeX source is not included.

## Homework 1

The first homework's code-upload requirement was optional. The existing notebook
and its dataset are included in `homework_1/`. To execute the notebook from the
repository root:

```sh
jupyter nbconvert --to notebook --execute --inplace homework_1/HW1.ipynb
```
