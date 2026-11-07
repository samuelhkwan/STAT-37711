# Homework 2

Install dependencies from the repository root using `requirements.txt`, then run:

```sh
python homework_2/run_question1.py
python homework_2/run_question2.py
python homework_2/run_question3.py
```

The `run_question*.py` files contain the question-specific experiments.
`training_utils.py` supplies the shared training loop for Questions 1 and 2.
`models.py`, `dataloader.py`, and `plotting.py` implement and support Question 3.

Keep all five supplied data files in this directory. Each script saves results
in its own `results_q*/` folder. Question 3 also saves all required graphs,
sweep tables, and the two headerless test-prediction CSVs.
