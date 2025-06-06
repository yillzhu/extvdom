#  Extended Validity Domain for Constraint Learning

Software and scripts to accompany the paper "An Extended Validity Domain for Constraint Learning", Yilin Zhu and Samuel Burer, June 2024, revised April 2025. In addition, this repository contains the code required to reproduce the tables and figures in Sections 3-6 of the paper and the online supplement.

## Section 3: Illustration

The following notebook shows the illustration of our extended convex hull validity domain: [Illustration](scripts/section_3_illustration.ipynb)

## Section 4: Numerical Results

In the [src](src) folder, you can find the code for the numerical experiments conducted in Section 4.

All the benchmark functions tested are defined in the *functions* python file (**src/function.py**). You can add new benchmark functions by defining them in this file and then add basic properties in the function `generate_function`.

In the *other* python file (**src/other.py**), you can adjust sample distributions and sample noises in `generate_samples` and specify machine learning models used to learn from the data in `train_model`. If a new machine learning model is not supported by [Gurobi Machine Learning](https://gurobi-machinelearning.readthedocs.io/en/stable/index.html), then a customized helper function should be added in the *gurobi_helper* python file(**src/gurobi_helper.py**).

In the [scripts](scripts) folder, you can find the scripts to run the numerical tests in a high performance computer (**scripts/hpc_many_fixed_seed.sh**, **scripts/hpc_many_seeds.sh**). If you want to run the tests on your local machine, you can use the Python files (**srcipts/run_one.py**, **scripts/run_many.py**).

In the [results](results) folder, we store all the numerical test results in [output.csv](results/output.csv). We also store all the figures and tables that can be found in this paper.

## Section 5: Two Stylized Optimization Models

These two notebooks implemented the two stylized optimization models in Section 5.

 - [A simple nonlinear optimization](scripts/section_5_nonlinear_optimization.ipynb)
 - [A price optimization](scripts/section_5_price_optimization.ipynb)
 
## Section 6: A Case Study
The [avocado price model](scripts/section_6_avocado.py) from Section 6 uses `gurobi_ml` and `gurobi_pandas` to generate optimization model with `GradientBoostingRegressor` from `Scikit Learn` embedded. This model takes long to solve in Gurobi v11, so setting a time limit for solving when setting up is recommended. Better performance can be acheived with Gurobi v12, in which case solves can be down in a few minutes.


```python
m.Params.TimeLimit = 600
```

## Online Supplement
The online supplement contains a more granular analysis of the [mean errors](scripts/section_s2_mean_errors.ipynb) and [sensitivities](scripts/section_s3_sensitivity.ipynb) with repect to different parameters. The figures and tables generated for the online supplement can also be found in [figures](figures) and [tables](tables) folders.
 

# Code Instructions

Here are sample commands to run from the root directory of the repository, which will run the code and create some of the tables and figures for the paper.  

```python
mkdir results/csv
mkdir results/joblib
mkdir results/figures
mkdir results/tables
python run_one.py 1 0 normal_at_min 1000 2023 net
# python run_many.py # Run all the experiments. May take awhile to run!
python section_4_tables_and_figures.py
Rscript section_4_tables_and_figures.R
python section_6_avocado.py
```
