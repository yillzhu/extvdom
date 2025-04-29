#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(1, '../src')

from main import *

# Make sure to set do_plots = True in options.py

run_one(1, 0, "uniform", 1000, 2023, "net", ["box"])
os.system('mv plot.png ../results/plots/beale_uniform_net_plots.png')

run_one(2, 0, "uniform", 1000, 2023, "net", ["box"])
os.system('mv plot.png ../results/plots/peaks_uniform_net_plots.png')

run_one(7, 0, "uniform", 1000, 2023, "net", ["box"])
os.system('mv plot.png ../results/plots/rastrigin2d_uniform_net_plots.png')


run_one(1, 0, "normal_at_min", 1000, 2023, "net", ["box"])
os.system('mv plot.png ../results/plots/beale_normal_at_min_net_plots.png')

run_one(2, 0, "normal_at_min", 1000, 2023, "net", ["box"])
os.system('mv plot.png ../results/plots/peaks_normal_at_min_net_plots.png')

run_one(7, 0, "normal_at_min", 1000, 2023, "net", ["box"])
os.system('mv plot.png ../results/plots/rastrigin2d_normal_at_min_net_plots.png')
