#!/usr/bin/env python
'''
checking the moons dataset...
'''

import pandas as pd
import numpy as np

import sys, os

pd.options.mode.chained_assignment = None  # default='warn'

res_files = sys.argv[1:]

df0, df1 = None, None
for i, f in enumerate(res_files):
    print(i, '\t', f)
    if i == 0:
        df0 = pd.read_csv(f)
    else:
        df1 = pd.read_csv(f)
        df0 = pd.concat([df0, df1], sort=False, ignore_index=True)

df0.to_csv('master_fit_results.csv')

# print(df0['opt_reg_thresh'])



# train_df = pd.read_csv(sys.argv[1]) # 'datasets/train_ds.csv')
