#!/usr/bon/env python
'''
Code to generate 2-feature moons dataset with additional peaking/exponential feature.
'''

import pandas as pd
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import sys
import pickle

from moons_tools import *

# to use latex with matplotlib
from matplotlib import rc
rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

fig = plt.figure(figsize=(11,3))

df_vals, pwrs = [], []
sfs = []
for i in range(len(sys.argv) - 1):
    print('reading ' + sys.argv[i+1])
    with open(sys.argv[i+1], 'rb') as f:
        df_vals.append(pickle.load(f))
        pwrs.append(pickle.load(f))
        farr = sys.argv[i+1].split(':')
        for s in farr:
            if 'sig_frac' in s:
                sfs.append(float(s.split('=')[-1]))


ax = plt.subplot(1,2,1)
for i in range(len(df_vals)):
    ax.plot(df_vals[i], pwrs[i], label=str(sfs[i]))
    plt.xlabel('decision function value')
    plt.ylabel('analysis power')
    plt.yscale('log')

plt.tight_layout()
plt.show()
# output_fname = 'reg_nn_pwrs/train_evts=' + str(n_evts)
# output_fname += ':noise=' + str(noise)
# output_fname += ':angle=' + str(angle)
# output_fname += ':epochs=' + str(n_epochs)
# output_fname += ':sig_frac=' + str(sig_frac)
#
#
#
# hist_ms(raw_df, min, max, nbins, ax)
#
# ax = plt.subplot(n_rows,n_cols, n_rows * n_cols)
# test_dict = {'x1':X_test['x1'], 'x2':X_test['x2'], 'm':masses_test, 'y':y_test, 'pred':y_pred_keras}
# test_df = pd.DataFrame(test_dict)
# min_df, max_df = np.amin(y_pred_keras), np.amax(y_pred_keras)
# nslices = 100
# dvals = np.linspace(min_df, max_df, num=nslices + 1)
# n_sig, n_bkgd, pwr = [], [], []
# for d in dvals:
#     n_sig.append(len(test_df['m'][test_df.y == 1][test_df.pred >= d]))
#     n_bkgd.append(len(test_df['m'][test_df.y == 0][test_df.pred >= d]))
#     pwr.append(n_sig[-1]*sig_frac / np.sqrt(n_sig[-1]*sig_frac + n_bkgd[-1]*(1-sig_frac)))
# opt_pwr = np.max(pwr)
# opt_idx = pwr.index(opt_pwr)
# opt_df  = dvals[opt_idx]
# ax.plot(dvals, pwr, label='sig fraction = ' + str(sig_frac))
# plt.xlabel('decision function value')
# plt.ylabel(r'$S / \sqrt{S+B}$')
# plt.axvline(x=opt_df, color='lightgray', dashes=(1,1))
# plt.legend(loc='lower right')
#
# output_fname += ':opt_df=' + str(np.round(opt_df, 6))
# with open(output_fname + ".pkl", 'wb') as f:
#     pickle.dump(dvals, f)
#     pickle.dump(pwr, f)
#
# plt.tight_layout()
#
# if 'noplot' not in sys.argv:
#     plt.show()
