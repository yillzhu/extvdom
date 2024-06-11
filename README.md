#  Extended Validity Domain for Constraint Learning

Software and scripts to accompany the paper "An Extended Validity Domain for Constraint Learning", Yilin Zhu and Samuel Burer, June 2024.


This repository contains the code for Section 4: Numerical Results, Section 5: Two Stylized Optimization Models and Section 6: A Case Study.

## Numerical Results

In the [src](https://research-git.uiowa.edu/yilin-and-sam/extvdom/-/tree/main/src) folder, you can find the code for the numerical experiments conducted in Section 4.

All the benchmark functions tested are defined in the *functions* python file (**src/function.py**), you can add new benchmark functions by defining them in this file and then add basic properties in the function `generate_function`.

In the *other* python file (**src/other.py**), you can adjust sample distributions and sample noises in `generate_samples` and specify machine learning models used to learn from the data in `train_model`. If a new machine learning model is not supported by [Gurobi Machine Learning](https://gurobi-machinelearning.readthedocs.io/en/stable/index.html) to embed into an optimization model, then a customized helper function should be added into the *gurobi_helper* python file(**src/gurobi_helper.py**).

In the [scripts](https://research-git.uiowa.edu/yilin-and-sam/extvdom/-/tree/main/scripts) folder, you can find the scripts to run the numerical tests in a high performance computer (**scripts/hpc_many_fixed_seed.sh**, **scripts/hpc_many_seeds.sh**). In the [results](https://research-git.uiowa.edu/yilin-and-sam/extvdom/-/tree/main/results) folder, we store all the numerical test results.

## Two Stylized Optimization Models

These two notebooks implemented the two stylized optimization models in Section 5.

 - [A simple nonlinear optimization](https://research-git.uiowa.edu/yilin-and-sam/extvdom/-/blob/main/scripts/section_5_nonlinear_optimization.ipynb)
 - [A price optimization](https://research-git.uiowa.edu/yilin-and-sam/extvdom/-/blob/main/scripts/section_5_price_optimization.ipynb)
 
## A Case Study
The [avocado price model](https://research-git.uiowa.edu/yilin-and-sam/extvdom/-/blob/main/scripts/section_6_avocado.py) from Section 6 uses `gurobi_ml` and `gurobi_pandas` to generate optimization model with `GradientBoostingRegressor` from `Scikit Learn` embedded. This model takes long to solve in Gurobi v11, so setting a time limit for solving when setting up is recommended.

```python
m.Params.TimeLimit = 600
```
 

# Code Instructions

  

```python

mkdir results/csv

mkdir results/joblib

mkdir results/figures

mkdir results/tables

  

python run_one.py 1 0 normal_at_min 1000 2023 net

# python run_many.py

  

# Run notebook section_3_illustration.ipynb

  

python section_4_tables_and_figures.py

Rscript section_4_tables_and_figures.R

  

# Run notebook section_5_nonlinear_optimization.ipynb

# Run notebook section_5_price_optimization.ipynb

  

python section_6_avocado.py

```
