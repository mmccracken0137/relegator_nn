#!/usr/bin/env python
'''
script that runs over results files and checks for missing run values...
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

import sys, os

pd.options.mode.chained_assignment = None  # default='warn'

# to use latex with matplotlib
#rc('font', **{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

tag = sys.argv[1]
files = sys.argv[2:]

print(tag)
print(files)

t_vals = []
for f in files:
    df = pd.read_csv(f)
    ts = df[tag].to_numpy()
    for v in ts:
        t_vals.append(float(v))

plt.hist(t_vals, bins=50)
plt.xlabel(tag)
plt.show()

# sfs = df.nominal_sig_frac.unique()
#
# df['final_signif'] = df['n_pass_sig'] / np.sqrt(df['n_pass_sig'] +
#                                                 df['n_pass_bkgd'])
#
# fig = plt.figure(figsize=(8,7))
# for i, s in enumerate(sfs):
#     ax = plt.subplot(3,2,i+1)
#     sub_df = df[df['nominal_sig_frac'] == s]
#     plt.scatter(sub_df[sub_df['model_type'] == 'regressor']['n_pass_bkgd'],
#                 sub_df[sub_df['model_type'] == 'regressor']['n_pass_sig'],
#                 marker='.', facecolors='none', edgecolors='b', label='regressor')
#     plt.scatter(sub_df[sub_df['model_type'] == 'relegator']['n_pass_bkgd'],
#                 sub_df[sub_df['model_type'] == 'relegator']['n_pass_sig'],
#                 marker='.', facecolors='none', edgecolors='orange', label='relegator')
#     plt.scatter(sub_df[sub_df['model_type'] == 'relegator_v2']['n_pass_bkgd'],
#                 sub_df[sub_df['model_type'] == 'relegator_v2']['n_pass_sig'],
#                 marker='.', facecolors='none', edgecolors='r', label='relegator_v2')
#
#     # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
#     # ax.text(0.05, 0.95, 'raw sig frac = ' + str(s), transform=ax.transAxes, fontsize=8,
#     #         verticalalignment='top', bbox=props)
#     ax.legend(loc='lower right', title='raw sig frac = ' + str(s))
#
# plt.tight_layout()
#
# fig = plt.figure(figsize=(8,7))
# for i, s in enumerate(sfs):
#     ax = plt.subplot(3,2,i+1)
#     sub_df = df[df['nominal_sig_frac'] == s]
#     plt.scatter(sub_df[sub_df['model_type'] == 'regressor']['n_pass_sig'],
#                 sub_df[sub_df['model_type'] == 'regressor']['final_signif'],
#                 marker='.', facecolors='none', edgecolors='b', label='regressor')
#     plt.scatter(sub_df[sub_df['model_type'] == 'relegator']['n_pass_sig'],
#                 sub_df[sub_df['model_type'] == 'relegator']['final_signif'],
#                 marker='.', facecolors='none', edgecolors='orange', label='relegator')
#     plt.scatter(sub_df[sub_df['model_type'] == 'relegator_v2']['n_pass_sig'],
#                 sub_df[sub_df['model_type'] == 'relegator_v2']['final_signif'],
#                 marker='.', facecolors='none', edgecolors='r', label='relegator_v2')
#
#     # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
#     # ax.text(0.05, 0.95, 'raw sig frac = ' + str(s), transform=ax.transAxes, fontsize=8,
#     #         verticalalignment='top', bbox=props)
#     ax.legend(loc='lower right', title='raw sig frac = ' + str(s))
#
# plt.tight_layout()
#
# plt.show()
