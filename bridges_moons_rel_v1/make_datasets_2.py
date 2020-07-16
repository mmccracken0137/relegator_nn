#!/usr/bon/env python
'''
Created train, test, and weighted moons+mass data.  Command line is

python make_datasets_2.py <config_file>.json
'''

from colorama import Fore, Back, Style
import pandas as pd
import numpy as np
import sklearn.datasets
import sys
import pickle
import json

from sklearn.model_selection import train_test_split

from moons_tools_2 import *

pd.options.mode.chained_assignment = None  # default='warn'

# # # # # # # # # # # # # # # # # # # #
# read in parameters from json config file
with open(sys.argv[1], 'r') as f:
    config_pars = json.loads(f.read())

n_evts = config_pars['data']['n_events'] # 2*int(sys.argv[2])
noise = config_pars['data']['noise'] # float(sys.argv[3])
angle = config_pars['data']['angle'] # 0.0
test_fraction = config_pars['data']['test_fraction'] # 0.25
sig_frac = config_pars['data']['sig_fraction'] # 0.5

bkgd_beta = config_pars['data']['bkgd_beta'] # 0.6
ttsplit_random_state = config_pars['data']['bkgd_beta'] # 42

# parameters for 'mass' distribution
min_mass = config_pars['data']['min_mass'] # 0.0
max_mass = config_pars['data']['max_mass'] # 1.0
mean_mass = config_pars['data']['mean_mass'] # 0.5
width_mass = config_pars['data']['width_mass'] # 0.03
n_sigmas = config_pars['data']['n_sigmas'] # 2.5

weighted_n_evts = config_pars['data']['weighted_n_events'] # 50000

# # # # # # # # # # # # # # # # # # # #
# generate train/test datasets...
print('generating training dataset...')
print('n events: ', n_evts)
print('noise:', noise)
print('angle: ', angle)
print('background beta: ',bkgd_beta)
train_df = make_moons_mass(n_evts, min_mass, max_mass,
                           mean=mean_mass, sigma=width_mass,
                           noise=noise, angle=angle, beta=bkgd_beta)


# train_df = train_df.sample(frac=1)
# herp = derp

# # # # # # # # # # # # # # # # # #
# generating weighted dataset...
print('\n\ngenerating weighted dataset...')
print('n events: ', weighted_n_evts)
print('signal fraction: ', sig_frac)
print('noise:', noise)
print('angle: ', angle)
print('background beta: ',bkgd_beta)
weighted_df = make_moons_mass(weighted_n_evts, min_mass, max_mass, mean=mean_mass, sigma=width_mass,
                              noise=noise, angle=angle, beta=bkgd_beta, sig_fraction=sig_frac)

# # # # # # # # # # # # # # # # # #
# pickle the datasets...
train_df.to_csv(config_pars['data']['train_data_file'] + '.csv', index=False)
# t_file = config_pars['data']['train_data_file']
# with open(t_file, 'wb') as f:
#     pickle.dump(train_df, f)
#     f.close()

weighted_df.to_csv(config_pars['data']['weighted_data_file'] + '.csv', index=False)
# w_file = config_pars['data']['weighted_data_file']
# with open(w_file, 'wb') as f:
#     pickle.dump(weighted_df, f)
#     f.close()
