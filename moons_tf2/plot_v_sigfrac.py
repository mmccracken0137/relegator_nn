#!/usr/bon/env python
'''
Plotting script for generating plots vs signal fraction
'''

import pandas as pd
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import sys
import pickle
import pandas as pd

plt.style.use('ggplot')

# to use latex with matplotlib
from matplotlib import rc
rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

df = pd.read_csv(sys.argv[1])

fig = plt.figure(figsize=(7,5))

regress = df[df['model_type'] == 'regress']
binary_softmax = df[df['model_type'] == 'binary_softmax']
relegator = df[df['model_type'] == 'relegator']
# relegator_factor = df[df['model_type'] == 'relegator_factor']
# relegator_diff = df[df['model_type'] == 'relegator_diff']

# print(regress.head(10))
ax = plt.subplot(1,1,1)
ms = 8
plt.plot(regress['sig_frac']*0.92,
         regress['pass_signif'], '^', label='logistic regression', mfc='none', markersize=ms)
plt.plot(binary_softmax['sig_frac'],
         binary_softmax['pass_signif'], 'v', label='binary softmax', mfc='none', markersize=ms)
plt.plot(relegator['sig_frac']*1.08,
         relegator['pass_signif'], 'p', label='relegation classifier', mfc='none', markersize=ms)
# plt.plot(relegator_factor['sig_frac']*1.16,
#          relegator_factor['pass_signif'], 's', label='rel.~factor', mfc='none', markersize=ms)
# plt.plot(relegator_factor['sig_frac']*1.24,
#          relegator_factor['pass_signif'], '*', label='rel.~diff.', mfc='none', markersize=ms)
plt.legend(loc='lower right')

ax.set_xlabel('signal fraction')
ax.set_ylabel(r'pass analysis significance, $S/\sqrt{S+B}$')

ax.set_xscale('log')
ax.set_yscale('log')

plt.tight_layout()
plt.show()
