#!/usr/bin/env python
'''
Tools for moons classifiers...
'''

from colorama import Fore, Back, Style
import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
import sklearn.datasets
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
from scipy.optimize import curve_fit
import tensorflow.keras.backend as K
import tensorflow as tf

def make_moons_mass(nevts, min, max, mean, sigma, noise=0.0, angle=0.0, beta=1.0, sig_fraction=0.5):
    # signal is signal...
    X, y, df = [], [], None
    df_sig, df_bkgd = None, None

    # if sig_fraction == 0.5:
    #     X, y = sklearn.datasets.make_moons(n_samples=nevts, shuffle=True, noise=noise)
    #     df = pd.DataFrame(dict(truth_class=y, x1=X[:,0], x2=X[:,1]))
    # else:

    # background events
    n_bkgd = int(nevts*(1-sig_fraction))
    print('making moons dataset...')
    print("number of background events:\t", n_bkgd)
    X, y = sklearn.datasets.make_moons(n_samples=2*n_bkgd, shuffle=True, noise=noise)
    df_bkgd = pd.DataFrame(dict(truth_class=y, x1=X[:,0], x2=X[:,1]))
    # drop signal events
    df_bkgd = df_bkgd[df_bkgd.truth_class == 0]

    # print(df_bkgd.head(10))

    # signal events
    n_sig = nevts - n_bkgd
    print("number of signal events:\t", n_sig)
    X, y = sklearn.datasets.make_moons(n_samples=2*n_sig, shuffle=True, noise=noise)
    df_sig = pd.DataFrame(dict(truth_class=y, x1=X[:,0], x2=X[:,1]))
    # drop signal events
    df_sig = df_sig[df_sig.truth_class == 1]

    # print(df_sig.head(10))

    df = pd.concat([df_sig, df_bkgd], ignore_index=True)

    ms = []
    # t0_1hot, t1_1hot = [], []
    x1_rot, x2_rot = [], []
    for idx, row in df.iterrows():
        if row['truth_class'] == 1:
            rand = max + 1
            while rand > max or rand < min:
                rand = np.random.normal(mean, sigma)
            ms.append(rand)
            # t0_1hot.append(0)
            # t1_1hot.append(1)
        elif row['truth_class'] == 0:
            rand = max + 1
            while rand > max or rand < min:
                rand = np.random.exponential(beta)
            ms.append(rand)
            # t0_1hot.append(1)
            # t1_1hot.append(0)

        if row['truth_class'] == 0:
            x1_rot.append(row['x1']*np.cos(angle) + row['x2']*np.sin(angle))
            x2_rot.append(row['x2']*np.cos(angle) - row['x1']*np.sin(angle))
        else:
            x1_rot.append(row['x1'])
            x2_rot.append(row['x2'])

    df = df.assign(mass = ms)
    #df = df.assign(truth_class_0 = t0_1hot)
    #df = df.assign(truth_class_1 = t1_1hot)

    # replace x1 with shifted x1s
    if angle != 0.0:
        dic = {'x1_rot': x1_rot, 'x2_rot': x2_rot}
        dfr = pd.DataFrame(dic)
        df['x1'] = dfr['x1_rot']
        df['x2'] = dfr['x2_rot']

    # shuffle the data
    df = df.sample(frac=1)
    # print(df.head(20))

    return df

def plot_xs(xs, labels, ax):
    npts = len(xs)
    msize = 13 / np.log10(npts)
    sig_idxs = np.where(labels == 1)[0]
    bkgd_idxs = np.where(labels == 0)[0]
    sig_xs = xs[sig_idxs]
    bkgd_xs = xs[bkgd_idxs]
    ax.plot(bkgd_xs[:,0], bkgd_xs[:,1], '.', markersize=msize, label='background', alpha=0.4)
    ax.plot(sig_xs[:,0], sig_xs[:,1], '.', markersize=msize, label='signal', alpha=0.4)
    plt.ylabel(r'$x_{1}$')
    plt.xlabel(r'$x_{0}$')
    # ax.legend(loc='upper left')
    return

def hist_xs(df, tag, nbins, ax):
    ax.hist(df[df.truth_class==0][tag], bins=nbins, histtype=u'step', label='background')
    ax.hist(df[df.truth_class==1][tag], bins=nbins, histtype=u'step', label='signal')
    plt.xlabel(tag)
    ax.legend(loc='upper right')
    return

def hist_fom(df, fom_name, min, max, nbins, ax, sig_limits=None):
    ax.hist(df[df.truth_class==0][fom_name], range=(min, max),
            bins=nbins, histtype=u'step', label='background')
    ax.hist(df[df.truth_class==1][fom_name], range=(min, max),
            bins=nbins, histtype=u'step', label='signal')
    occs, edges, _ = ax.hist(df[fom_name], range=(min, max),
                             bins=nbins, histtype=u'step', label='all events')
    cents = edges[:-1] + np.diff(edges) / 2
    pars = fit_mass_hist(cents, occs)
    ax.plot(edges, f_expbkgd(edges, pars[3], pars[4]), label='bkgd fit',
            alpha=0.8, linestyle=':', color='cornflowerblue')

    plt.xlabel(r'$m$')
    ax.legend(loc='upper right')

    if sig_limits != None:
        vert_sig_line(x=sig_limits[0])
        vert_sig_line(x=sig_limits[1])

    return cents, occs, f_expbkgd(cents, pars[3], pars[4])

def f_gauss_expbkgd(x, a, mu, sig, b, lam):
    f = f_gaussian(x, a, mu, sig) + f_expbkgd(x, b, lam)
    return f

def f_gaussian(x, a, mu, sig):
    f = a*np.exp(-(x - mu)**2/2/sig**2)
    return f

def f_expbkgd(x, b, lam):
    f = b*np.exp(-lam*x)
    return f

def fit_mass_hist(x, y):
    p_vals=[10, 0.5, 0.15, y[0], 1.0]
    popt, pcov = curve_fit(f_gauss_expbkgd, x, y, p0=p_vals)
    return popt

def hist_cut_ms(df, fom_name, opt_df, min, max, nbins, ax, sig_limits=None):
    # signal
    ax.hist(df[fom_name][df.pred>=opt_df][df.truth_class==0], #alpha=0.3, fill=True,
            range=(min, max), bins=nbins, histtype=u'step', label='background')
    ax.hist(df[fom_name][df.pred>=opt_df][df.truth_class==1], #alpha=0.3, fill=True,
            range=(min, max), bins=nbins, histtype=u'step', label='signal')
    occs, edges, _ = ax.hist(df[fom_name][df.pred>=opt_df], #alpha=0.3, fill=True,
                          range=(min, max), bins=nbins, histtype=u'step',
                          label='all')
    cents = edges[:-1] + np.diff(edges) / 2
    pars = fit_mass_hist(cents, occs)
    ax.plot(edges, f_expbkgd(edges, pars[3], pars[4]), label='bkgd fit',
            alpha=0.8, linestyle=':', color='cornflowerblue')
    plt.xlabel('fom')
    ax.legend(loc='upper right')

    if sig_limits != None:
        vert_sig_line(x=sig_limits[0])
        vert_sig_line(x=sig_limits[1])

    return cents, occs, f_expbkgd(cents, pars[3], pars[4])

def hist_softmax_cut_ms(df, fom_name, min, max, nbins, ax, sig_limits=None):
    # signal
    ax.hist(df[fom_name][df.prob_1>=0.5][df.truth_class==0], #alpha=0.3, #fill=True,
            range=(min, max), bins=nbins, histtype=u'step', label='background, post-cut')
    ax.hist(df[fom_name][df.prob_1>=0.5][df.truth_class==1], #alpha=0.3, #fill=True,
            range=(min, max), bins=nbins, histtype=u'step', label='signal, post-cut')
    occs, edges, _ = ax.hist(df[fom_name][df.prob_1>=0.5], #alpha=0.3, #fill=True,
            range=(min, max), bins=nbins, histtype=u'step',
            label='all, post-cut')
    cents = edges[:-1] + np.diff(edges) / 2
    pars = fit_mass_hist(cents, occs)
    ax.plot(edges, f_expbkgd(edges, pars[3], pars[4]), label='bkgd fit',
            alpha=0.8, linestyle=':', color='cornflowerblue')
    plt.xlabel(r'$m$')
    ax.legend(loc='upper right')

    if sig_limits != None:
        vert_sig_line(x=sig_limits[0])
        vert_sig_line(x=sig_limits[1])

    return cents, occs, f_expbkgd(cents, pars[3], pars[4])

def signif_function(n_s, n_b):
    if tf.is_tensor(n_s): # for relegator loss
        sig = tf.math.divide(n_s, K.sqrt(n_s + n_b))
    else:
        sig = n_s / np.sqrt(n_s + n_b)
    return sig

def signif_error(n_s, n_b):
    err = n_s**2 / 4 / (n_s + n_b)**2
    err += (2*n_b + n_s)**2 / 4 / (n_s + n_b)**2
    err = np.sqrt(err)
    return err

def hist_diff_signif(x, y_tot, y_bkgd):
    idxs = np.array(np.nonzero(y_tot))
    # remove points with zero n_tot
    x = x[idxs].flatten()
    y_tot = y_tot[idxs].flatten()
    y_bkgd = y_bkgd[idxs].flatten()

    diff = np.subtract(y_tot, y_bkgd)
    signif = signif_function(diff, y_bkgd)
    errs = signif_error(diff, y_bkgd) #np.sqrt(np.abs(diff)/y_bkgd) # approximate, TKTKTK
    plt.errorbar(x, signif, yerr=errs, fmt='.k')
    plt.ylabel(r'$s / \sqrt{s+b}$')
    plt.xlabel(r'$m$')
    return 0

def hist_residuals(x, y_tot, y_bkgd, sig_limits=None):
    # idxs = np.array(np.nonzero(y_tot))
    # remove points with zero n_tot
    # x = x[idxs].flatten()
    y_tot = y_tot.flatten()
    y_tot_err = np.sqrt(y_tot)
    y_bkgd = y_bkgd.flatten()

    diff = np.subtract(y_tot, y_bkgd)
    plt.errorbar(x, diff, yerr=y_tot_err, fmt='.k')
    plt.ylabel(r'residuals')
    plt.xlabel(r'$m$')

    if sig_limits != None:
        vert_sig_line(x=sig_limits[0])
        vert_sig_line(x=sig_limits[1])
    return 0

def compute_signif_regress(df, fom_name, opt_df, m_cent, m_wid, n_sig):
    n_raw_bkgd  = len(df[fom_name][df.truth_class==0][np.abs(df[fom_name] - m_cent) < n_sig*m_wid])
    n_raw_sig   = len(df[fom_name][df.truth_class==1][np.abs(df[fom_name] - m_cent) < n_sig*m_wid])
    n_pass_bkgd = len(df[fom_name][df.pred>=opt_df][df.truth_class==0][np.abs(df[fom_name] - m_cent) < n_sig*m_wid])
    n_pass_sig  = len(df[fom_name][df.pred>=opt_df][df.truth_class==1][np.abs(df[fom_name] - m_cent) < n_sig*m_wid])
    raw_signif  = signif_function(n_raw_sig, n_raw_bkgd)
    pass_signif = signif_function(n_pass_sig, n_pass_bkgd)
    print_pass_stats(n_raw_sig, n_pass_sig, n_raw_bkgd, n_pass_bkgd)
    return raw_signif, pass_signif, n_raw_bkgd, n_raw_sig, n_pass_bkgd, n_pass_sig

def compute_signif_binary(df, fom_name, m_cent, m_wid, n_sig):
    n_raw_bkgd  = len(df[fom_name][df.truth_class==0][np.abs(df[fom_name] - m_cent) < n_sig*m_wid])
    n_raw_sig   = len(df[fom_name][df.truth_class==1][np.abs(df[fom_name] - m_cent) < n_sig*m_wid])
    n_pass_bkgd = len(df[fom_name][df.prob_1>=0.5][df.truth_class==0][np.abs(df[fom_name] - m_cent) < n_sig*m_wid])
    n_pass_sig  = len(df[fom_name][df.prob_1>=0.5][df.truth_class==1][np.abs(df[fom_name] - m_cent) < n_sig*m_wid])
    raw_signif  = signif_function(n_raw_sig, n_raw_bkgd)
    pass_signif = signif_function(n_pass_sig, n_pass_bkgd)
    print_pass_stats(n_raw_sig, n_pass_sig, n_raw_bkgd, n_pass_bkgd)
    return raw_signif, pass_signif, n_raw_bkgd, n_raw_sig, n_pass_bkgd, n_pass_sig

def print_pass_stats(n_raw_sig, n_pass_sig, n_raw_bkgd, n_pass_bkgd):
    print(Fore.BLUE)
    print('\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    print('-events in signal region')
    print('-\t\t\t\t raw\t\t pass')
    print("-number of background events:\t", n_raw_bkgd, '\t\t', n_pass_bkgd)
    print("-number of signal events:\t", n_raw_sig, '\t\t', n_pass_sig)
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    print(Style.RESET_ALL)
    return

def gen_confusion_matrix(y_true, y_pred, normalize=False):
    """
    This function returns a 2-d confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return cm

def cm_to_str(cm):
    cm = np.array(cm).flatten().astype('str')
    return ':'.join(cm)

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
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = gen_confusion_matrix(y_true, y_pred, normalize=False)
    raw_cm = cm.copy()

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
           ylabel='truth class',
           xlabel='pred. class')

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
    return ax, raw_cm

def np_to_tfds(X, y, batch_size=1024):
    feats_arr = ['x1', 'x2']
    n_evts = len(X)
    tf_ds = tf.data.Dataset.from_tensor_slices((tf.cast(X[feats_arr].values, tf.float32),
                                                tf.cast(y.values, tf.int32)))
    tf_ds = tf_ds.shuffle(n_evts).batch(batch_size)
    return tf_ds

def pred_1hot_to_class(X_in, model, n_classes):
    # converts 1hot model output to class label
    pred_1hot = model.predict(X_in)
    pred_class = [np.where(p == np.max(p)) for p in pred_1hot]
    pred_class = np.reshape(pred_class, (np.shape(X_in)[0], 1))
    return pred_class

def predict_bound_class(model, df, n_outputs, opt_thr=0):
    x1_min, x1_max = df['x1'].min() - 0.25, df['x1'].max() + 0.25
    x2_min, x2_max = df['x2'].min() - 0.25, df['x2'].max() + 0.25
    x1_range = x1_max - x1_min
    x2_range = x2_max - x2_min
    x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, x1_range/100),
                                   np.arange(x2_min, x2_max, x2_range/100))

    mesh_xs = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]
    class_mesh = []
    if n_outputs == 1:
        class_mesh = model.predict(mesh_xs)
        if opt_thr > 0.0:
            class_mesh[class_mesh > opt_thr]  = 1
            class_mesh[class_mesh <= opt_thr] = 0
    else:
        class_mesh = pred_1hot_to_class(mesh_xs, model, n_outputs)
    class_mesh = class_mesh.reshape(x1_mesh.shape)
    return x1_mesh, x2_mesh, class_mesh

def vert_sig_line(x=0.0):
    plt.axvline(x, color='gray', linestyle='--', alpha=0.4)

def relegator_cmap():
    # purple, orange, blue
    # cdict = {'red': ((0.0, 61/255, 61/255),
    #                  (0.5, 1.0, 1.0),
    #                  (1.0, 0, 0)),
    #
    #          'green': ((0.0, 30/255, 30/255),
    #                    (0.5, 135/255, 135/255),
    #                    (1.0, 223/255, 223/255)),
    #
    #          'blue': ((0.0,  1.0, 1.0),
    #                   (0.5,  33/255, 33/255),
    #                   (1.0,  1.0, 1.0))}

    # blue, orange, grey
    cdict = {'red': ((0.0, 0/255, 0/255),
                     (0.5, 1.0, 1.0),
                     (1.0, 0.4, 0.4)),

             'green': ((0.0, 0.25, 0.25),
                       (0.5, 135/255, 135/255),
                       (1.0, 0.4, 0.4)),

             'blue': ((0.0,  1.0, 1.0),
                      (0.5,  33/255, 33/255),
                      (1.0,  0.4, 0.4))}
    return LinearSegmentedColormap('RelegatorCMap', cdict)
