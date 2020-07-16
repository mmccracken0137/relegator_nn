#!/usr/bin/env python
'''
runs gen moons data and fits with supplied parameters
'''

import os
import json
import numpy as np
import sys
import string
import random

# # ts = ['regress', 'binary_softmax', 'relegator'] # , 'relegator_factor', 'relegator_diff']
# ts = ['relelgator']
# #, 'mod_binary']
# # ts = ['relegator_factor'] #, 'mod_binary']
# n_sigs = 1 # 8
# pow_range = (-3,-1)
# sig_pows = np.linspace(pow_range[0], pow_range[1], n_sigs + 1)

n_fits = 5 # 0
n_datasets = 5 # 0

n_train_events = 20000
n_weighted_events = 100000

run = True

sig_frac, noise, angle = 0, 0, 0
# parse command line
for c in sys.argv:
    if '-s' in c:
        sig_frac = float(c[2:])
    elif '-n' in c:
        noise = float(c[2:])
    elif '-a' in c:
        angle = float(c[2:])

print('generating data with \nnoise = ' + str(noise))
print('sig_frac = ' + str(sig_frac))
print('noise = ' + str(noise))

for dno in range(n_datasets):
    choose = string.ascii_lowercase + '0123456789'
    ds_tag = ''.join(random.choice(choose) for i in range(8))

    train_file_name = './datasets/train_ds_' + ds_tag
    weight_file_name = './datasets/weighted_ds_' + ds_tag

    # print(sig_frac)
    json_str = "{ \n \
    \"data\": { \n \
    \"n_events\": " + str(n_train_events) + ", \n \
    \"noise\": " + str(noise) + ", \n \
    \"angle\": " + str(angle) + ", \n \
    \"sig_fraction\": "
    json_str += str(sig_frac)
    json_str += ", \n \
    \"test_fraction\": 0.25, \n \
    \"min_mass\": 0.0, \n \
    \"max_mass\": 1.0, \n \
    \"mean_mass\": 0.5, \n \
    \"width_mass\": 0.2, \n \
    \"n_sigmas\": 2.5, \n \
    \"bkgd_beta\": 1.0, \n \
    \"ttsplit_random_state\": 42, \n \
    \"weighted_n_events\": " + str(n_weighted_events) + ", \n \
    \"train_data_file\": \"" + train_file_name + "\", \n \
    \"weighted_data_file\": \"" + weight_file_name + "\" \n \
    } \n \
    }"
    with open("config_run.json", "w") as f:
        f.write(json_str)
    cmd = "python make_datasets_2.py config_run.json"
    print(sig_frac)
    print(cmd)
    if run:
        os.system(cmd)

    for j in range(n_fits):
        ## python test_clf.py write_results tag:noise=0.3 tag:angle=1.4 tag:foo=bar noplot
        # run relegator
        cmd = 'python test_relegator_clf.py write_results noplot tag:noise=' + str(noise)
        cmd += ' tag:angle=' + str(angle)
        cmd += ' tag:nomsigfrac=' + str(sig_frac)
        cmd += ' tag:dataset=' + ds_tag
        print(cmd)
        if run:
            os.system(cmd)

        # run regressor
        cmd = 'python test_regressor_clf.py write_results noplot tag:noise=' + str(noise)
        cmd += ' tag:angle=' + str(angle)
        cmd += ' tag:nomsigfrac=' + str(sig_frac)
        cmd += ' tag:dataset=' + ds_tag
        print(cmd)
        if run:
            os.system(cmd)

    cmd = 'rm -f ' + train_file_name + '.csv ' + weight_file_name + '.csv'
    print(cmd)
    if run:
        os.system(cmd)
