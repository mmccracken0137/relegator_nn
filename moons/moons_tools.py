#!/usr/bon/env python
'''
Tools for moons classifiers...
'''

import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
import sklearn.datasets
import matplotlib.pyplot as plt
import sys

def make_moons_mass(nevts, min, max, mean, sigma, noise=0.0, angle=0.0, beta=1.0, sig_fraction=0.5):
    # signal is type 1...
    X, y, df = [], [], None
    df_sig, df_bkgd = None, None
    if sig_fraction == 0.5:
        X, y = sklearn.datasets.make_moons(n_samples=nevts, shuffle=True, noise=noise)
        df = pd.DataFrame(dict(x1=X[:,0], x2=X[:,1], label=y))
    else:
        # background events
        n_bkgd = int(nevts*(1-sig_fraction))
        print('making moons dataset...')
        print("number of background events:\t", n_bkgd)
        X, y = sklearn.datasets.make_moons(n_samples=2*n_bkgd, shuffle=True, noise=noise)
        df_bkgd = pd.DataFrame(dict(x1=X[:,0], x2=X[:,1], label=y))
        # drop signal events
        df_bkgd = df_bkgd[df_bkgd.label == 0]

        # signal events
        n_sig = nevts - n_bkgd
        print("number of signal events:\t", n_sig)
        X, y = sklearn.datasets.make_moons(n_samples=2*n_sig, shuffle=True, noise=noise)
        df_sig = pd.DataFrame(dict(x1=X[:,0], x2=X[:,1], label=y))
        # drop signal events
        df_sig = df_sig[df_sig.label == 1]

        df = pd.concat([df_sig, df_bkgd], ignore_index=True)

    ms = []
    t0_1hot, t1_1hot = [], []
    x1_rot, x2_rot = [], []
    for idx, row in df.iterrows():
        if row['label'] == 1:
            ms.append(np.random.normal(mean, sigma))
            t0_1hot.append(0)
            t1_1hot.append(1)
        elif row['label'] == 0:
            rand = max + 1
            while rand > max or rand < min:
                rand = np.random.exponential(beta)
            ms.append(rand)
            t0_1hot.append(1)
            t1_1hot.append(0)

        if row['label'] == 0:
            x1_rot.append(row['x1']*np.cos(angle) + row['x2']*np.sin(angle))
            x2_rot.append(row['x2']*np.cos(angle) - row['x1']*np.sin(angle))
        else:
            x1_rot.append(row['x1'])
            x2_rot.append(row['x2'])

    df = df.assign(m = ms)
    df = df.assign(label_0 = t0_1hot)
    df = df.assign(label_1 = t1_1hot)

    # replace x1 with shifted x1s
    if angle != 0.0:
        dic = {'x1_rot': x1_rot, 'x2_rot': x2_rot}
        dfr = pd.DataFrame(dic)
        df['x1'] = dfr['x1_rot']
        df['x2'] = dfr['x2_rot']

    return df

def plot_xs(df, ax):
    npts = len(df['x1'])
    msize = 13 / np.log10(npts)
    ax.plot(df['x1'][df.label==0], df['x2'][df.label==0], '.', markersize=msize, label='type 0', alpha=0.4)
    ax.plot(df['x1'][df.label==1], df['x2'][df.label==1], '.', markersize=msize, label='type 1', alpha=0.4)
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

def hist_weighted_ms(df, sig_weight, min, max, nbins, ax):
    # signal
    ax.hist(df['m'][df.y==0],
            range=(min, max), bins=nbins, histtype=u'step', label='type 0')
    ax.hist(df['m'][df.y==1],
            weights=sig_weight*np.ones(len(df['m'][df.y==1])),
            range=(min, max), bins=nbins, histtype=u'step', label='type 1')
    ax.hist(df['m'], range=(min, max),
            bins=nbins, histtype=u'step', label='all')
    plt.xlabel(r'$m$')
    ax.legend(loc='upper right')
    return 0

def hist_cut_ms(df, opt_df, min, max, nbins, ax):
    # signal
    ax.hist(df['m'][df.pred>=opt_df][df.label==0], #alpha=0.3, fill=True,
            range=(min, max), bins=nbins, histtype=u'step', label='type 0, post-cut')
    ax.hist(df['m'][df.pred>=opt_df][df.label==1], #alpha=0.3, fill=True,
            range=(min, max), bins=nbins, histtype=u'step', label='type 1, post-cut')
    ax.hist(df['m'][df.pred>=opt_df], #alpha=0.3, fill=True,
            range=(min, max), bins=nbins, histtype=u'step',
            label='all, post-cut')
    plt.xlabel(r'$m$')
    ax.legend(loc='upper right')
    return 0

def hist_softmax_cut_ms(df, min, max, nbins, ax):
    # signal
    ax.hist(df['m'][df.prob_1>=0.5][df.label==0], #alpha=0.3, #fill=True,
            range=(min, max), bins=nbins, histtype=u'step', label='type 0, post-cut')
    ax.hist(df['m'][df.prob_1>=0.5][df.label==1], #alpha=0.3, #fill=True,
            range=(min, max), bins=nbins, histtype=u'step', label='type 1, post-cut')
    ax.hist(df['m'][df.prob_1>=0.5], #alpha=0.3, #fill=True,
            range=(min, max), bins=nbins, histtype=u'step',
            label='all, post-cut')
    plt.xlabel(r'$m$')
    ax.legend(loc='upper right')
    return 0

def compute_signif_regress(df, opt_df, m_cent, m_wid, n_sig):
    n_raw_bkgd  = len(df['m'][df.label==0][np.abs(df.m - m_cent) < n_sig*m_wid])
    n_raw_sig   = len(df['m'][df.label==1][np.abs(df.m - m_cent) < n_sig*m_wid])
    n_pass_bkgd = len(df['m'][df.pred>=opt_df][df.label==0][np.abs(df.m - m_cent) < n_sig*m_wid])
    n_pass_sig  = len(df['m'][df.pred>=opt_df][df.label==1][np.abs(df.m - m_cent) < n_sig*m_wid])
    raw_signif  = n_raw_sig / np.sqrt(n_raw_sig + n_raw_bkgd)
    pass_signif = n_pass_sig / np.sqrt(n_pass_sig + n_pass_bkgd)
    print_pass_stats(n_raw_sig, n_pass_sig, n_raw_bkgd, n_pass_bkgd)
    return raw_signif, pass_signif, n_raw_bkgd, n_raw_sig, n_pass_bkgd, n_pass_sig

def compute_signif_binary(df, m_cent, m_wid, n_sig):
    n_raw_bkgd  = len(df['m'][df.label==0][np.abs(df.m - m_cent) < n_sig*m_wid])
    n_raw_sig   = len(df['m'][df.label==1][np.abs(df.m - m_cent) < n_sig*m_wid])
    n_pass_bkgd = len(df['m'][df.prob_1>=0.5][df.label==0][np.abs(df.m - m_cent) < n_sig*m_wid])
    n_pass_sig  = len(df['m'][df.prob_1>=0.5][df.label==1][np.abs(df.m - m_cent) < n_sig*m_wid])
    raw_signif  = n_raw_sig / np.sqrt(n_raw_sig + n_raw_bkgd)
    pass_signif = n_pass_sig / np.sqrt(n_pass_sig + n_pass_bkgd)
    print_pass_stats(n_raw_sig, n_pass_sig, n_raw_bkgd, n_pass_bkgd)
    return raw_signif, pass_signif, n_raw_bkgd, n_raw_sig, n_pass_bkgd, n_pass_sig

def print_pass_stats(n_raw_sig, n_pass_sig, n_raw_bkgd, n_pass_bkgd):
    print('\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    print('-events in signal region')
    print('-\t\t\t\t raw\t\t pass')
    print("-number of background events:\t", n_raw_bkgd, '\t\t', n_pass_bkgd)
    print("-number of signal events:\t", n_raw_sig, '\t\t', n_pass_sig)
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    return

def plot_confusion_matrix(y_true, y_pred, classes, ax,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)

    print("\nRaw confusion matrix")
    print(cm)

    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\nNormalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    #fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', vmin=0.0, vmax=1.0, cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='true label',
           xlabel='predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return ax
