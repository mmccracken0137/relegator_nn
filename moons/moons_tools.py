#!/usr/bon/env python
'''
Code to generate 2-feature moons dataset with additional peaking/exponential feature.
'''

import pandas as pd
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import sys

def make_moons_mass(nevts, min, max, mean, sigma, noise=0.0, angle=0.0, beta=1.0):

    X, y = sklearn.datasets.make_moons(n_samples=nevts, shuffle=True, noise=noise)
    df = pd.DataFrame(dict(x1=X[:,0], x2=X[:,1], label=y))

    # if shift != 0.0:
    #     df['x1'][df.label==1] = df['x1'][df.label==1] + shift

    # add mass feature to X
    ms = []
    x1_rot, x2_rot = [], []
    for idx, row in df.iterrows():
        if row['label'] == 1:
            ms.append(np.random.normal(mean, sigma))
        elif row['label'] == 0:
            rand = max + 1
            while rand > max or rand < min:
                rand = np.random.exponential(beta)
            ms.append(rand)

        if row['label'] == 0:
            x1_rot.append(row['x1']*np.cos(angle) + row['x2']*np.sin(angle))
            x2_rot.append(row['x2']*np.cos(angle) - row['x1']*np.sin(angle))
        else:
            x1_rot.append(row['x1'])
            x2_rot.append(row['x2'])

    df = df.assign(m = ms)

    # replace x1 with shifted x1s
    if angle != 0.0:
        dic = {'x1_rot': x1_rot, 'x2_rot': x2_rot}
        dfr = pd.DataFrame(dic)
        df['x1'] = dfr['x1_rot']
        df['x2'] = dfr['x2_rot']

    return df

def plot_xs(df, ax):
    npts = len(df['x1'])
    msize = 10 / np.log10(npts)
    ax.plot(df['x1'][df.label==0], df['x2'][df.label==0], '.', markersize=msize, label='type 0')
    ax.plot(df['x1'][df.label==1], df['x2'][df.label==1], '.', markersize=msize, label='type 1')
    plt.ylabel(r'$x_{2}$')
    plt.xlabel(r'$x_{1}$')
    # ax.legend(loc='upper left')
    return

def hist_xs(df, tag, nbins, ax):
    ax.hist(df[tag][df.label==0], bins=nbins, histtype=u'step', label='type 0')
    ax.hist(df[tag][df.label==1], bins=nbins, histtype=u'step', label='type 1')
    plt.xlabel(tag)
    ax.legend(loc='upper right')
    return

def hist_ms(df, min, max, nbins, ax):
    ax.hist(df['m'][df.label==0], range=(min, max), bins=nbins, histtype=u'step', label='type 0')
    ax.hist(df['m'][df.label==1], range=(min, max), bins=nbins, histtype=u'step', label='type 1')
    ax.hist(df['m'], range=(min, max), bins=nbins, histtype=u'step', label='all events')
    plt.xlabel(r'$m$')
    ax.legend(loc='upper right')
    return 0
