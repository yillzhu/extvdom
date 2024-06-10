An Extended Validity Domain for Constraint Learning
===================================================

Software and scripts to accompany the paper "An Extended Validity Domain
for Constraint Learning", Yilin Zhu and Samuel Burer, June 2024.

# Code Instructions

```
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
