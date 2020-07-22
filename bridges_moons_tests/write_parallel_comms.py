#!/usr/bin/env python
'''
writes run_genfit.py commands to file to be used by GNU parallel
'''

import os
import sys
import itertools

n_fits = 4 # 0
n_datasets = 4 # 0

angle_vals = [0.0, 0.6, 1.7]
sig_frac_vals = [0.001, 0.00316, 0.01, 0.0316, 0.1, 0.316, 0.5]
# sig_frac_vals = [0.0001, 0.000316, 0.001, 0.00316, 0.01, 0.0316]
noise_vals = [0.2, 0.3, 0.4]

n_datasets = 5
n_fits = 5

n_train = 20000
n_weight = 100000
# n_weight = 1000000

f = open('parallel_comms.txt', 'w')

for i, x in enumerate(itertools.product(angle_vals, sig_frac_vals, noise_vals)):
    print(i, x)
    cmd = './run_genfit.py'
    cmd += ' -a' + str(x[0])
    cmd += ' -s' + str(x[1])
    cmd += ' -n' + str(x[2])
    cmd += ' -f' + str(n_fits)
    cmd += ' -d' + str(n_datasets)
    cmd += ' -t' + str(n_train)
    cmd += ' -w' + str(n_weight) + '\n'
    print(cmd)
    f.write(cmd)

f.close()
