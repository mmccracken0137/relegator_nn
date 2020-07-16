#!/usr/bin/env python
'''
runs master_moons_2.py with loops over parameters
'''

import os
import json
import numpy as np
import sys
import string
import random

# ts = ['regress', 'binary_softmax', 'relegator'] # , 'relegator_factor', 'relegator_diff']
ts = ['relelgator']
#, 'mod_binary']
# ts = ['relegator_factor'] #, 'mod_binary']
n_sigs = 1 # 8
pow_range = (-3,-1)
sig_pows = np.linspace(pow_range[0], pow_range[1], n_sigs + 1)

n_fits = 5 # 0
n_datasets = 10 # 0

n_train_events = 20000
n_weighted_events = 100000

noise = 0.3
angle = 1.4

for sp in sig_pows:
    sig_frac = np.round(10**sp, 4)
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
        \"width_mass\": 0.03, \n \
        \"n_sigmas\": 2.5, \n \
        \"bkgd_beta\": 0.6, \n \
        \"ttsplit_random_state\": 42, \n \
        \"weighted_n_events\": " + str(n_weighted_events) + ", \n \
        \"train_data_file\": \"" + train_file_name + "\", \n \
        \"weighted_data_file\": \"" + weight_file_name + "\" \n \
        } \n \
        }"
        with open("config_run.json", "w") as f:
            f.write(json_str)
        cmd = "python make_datasets_2.py config_run.json"
        print(ts[0], sig_frac)
        print(cmd)
        os.system(cmd)
        for j in range(n_fits):
            ## python test_clf.py write_results tag:noise=0.3 tag:angle=1.4 tag:foo=bar noplot
            cmd = 'python test_relegator_clf.py write_results noplot tag:noise=' + str(noise)
            cmd += ' tag:angle=' + str(angle)
            cmd += ' tag:nomsigfrac=' + str(sig_frac)
            cmd += ' tag:dataset=' + ds_tag
            print(cmd)
            os.system(cmd)
        cmd = 'rm -f ' + train_file_name + '.csv ' + weight_file_name + '.csv'
        print(cmd)
        os.system(cmd)
