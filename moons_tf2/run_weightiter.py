#!/usr/bon/env python
'''
runs master_moons_2.py with loops over parameters
'''

import os
import json
import numpy as np
import sys

ts = ['regress', 'binary_softmax', 'relegator'] #, 'mod_binary']
n_sigs = 6
pow_range = (-3,-1)
sig_pows = np.linspace(pow_range[0], pow_range[1], n_sigs + 1)

n_trials = 5
n_train_events = 20000
n_weighted_events = 100000
n_weight_iters = 25

for sp in sig_pows:
    sig_frac = np.round(10**sp, 4)
    train_file_name = './datasets/train_ds_' + str(n_train_events)
    train_file_name += '_' + str(np.round(sig_frac,4)) + '.pkl'
    weight_file_name = './datasets/weighted_ds_' + str(n_weighted_events)
    weight_file_name += '_' + str(np.round(sig_frac,4)) + '.pkl'
    for t in ts:
        # print(sig_frac)
        json_str = "{ \n \
        \"data\": { \n \
        \"n_events\": " + str(n_train_events) + ", \n \
        \"noise\": 0.2, \n \
        \"angle\": 1.4, \n \
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
        \"weighted_data_file\": \"" + weight_file_name + "\", \n \
        \"n_weighted_datasets\": \"" + str(n_weight_iters) + "\" \n \
        }, \n \
        \"model\": { \n \
        \"model_type\": \""
        json_str += t
        json_str += "\", \n \
        \"input_dropout\": 0.05, \n \
        \"learning_rate\": 0.0005, \n \
        \"hidden_nodes\": [40, 40, 20], \n \
        \"bias\": true, \n \
        \"signif_type\": \"proba\" \n \
        }, \n \
        \"run\": { \n \
        \"n_epochs\": 1000, \n \
        \"ot_cutoff\": true, \n \
        \"ot_cutoff_depth\": 20 \n \
        } \n \
        }"
        # print(json_str)
        with open("config_run.json", "w") as f:
            f.write(json_str)
        for j in range(n_trials):
            cmd = "python multi_weighted_moons_2.py config_run.json noplot write_results gen_data"
            print(cmd)
            os.system(cmd)
