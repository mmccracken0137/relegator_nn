#!/usr/bon/env python
'''
Fits arbitrary model to moons+mass data.  Command line is

python master_moons.py <model> <n_train_events> <noise> [<angle> <max_epochs> <sig_fraction>]

model can take the values: regress, nn_binary, relegator

Last three arguments may be omitted to run with default values.
'''

from colorama import Fore, Back, Style
import pandas as pd
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import sys
import pickle

from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.utils.multiclass import unique_labels
import tensorflow as tf  ## this code runs with tf2.0-cpu!!!
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from moons_tools import *
from moons_models import *

# to use latex with matplotlib
from matplotlib import rc
rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

model_type = sys.argv[1]
allowed_models = ['regress', 'signif_regress',
                  'nn_binary', 'signif_nn_binary',
                  'relegator', 'signif_relegator']
loss_types = {'regress':'bin. CE', 'signif_regress':'mod CE',
              'nn_binary':'cat. CE', 'signif_nn_binary':'mode cat. CE',
              'relegator':'cat. CE', 'signif_relegator':'mod cat. CE'}
loss_type = loss_types[model_type]
if model_type not in allowed_models:
    print('error: model type \"' + model_type + '\" undefined')
    exit

n_evts = 2*int(sys.argv[2])
noise = float(sys.argv[3])
angle = 0.0
n_epochs = 100
sig_frac = 0.5
if len(sys.argv) >= 5:
    angle = float(sys.argv[4])
if len(sys.argv) >= 6:
    n_epochs = int(sys.argv[5])
if len(sys.argv) >= 7:
    sig_frac = float(sys.argv[6])

ot_cutoff_depth = 5

output_fname = 'reg_nn_pwrs/train_evts=' + str(n_evts)
output_fname += ':noise=' + str(noise)
output_fname += ':angle=' + str(angle)
output_fname += ':epochs=' + str(n_epochs)
output_fname += ':sig_frac=' + str(sig_frac)

# parameters for 'mass' distribution
min_mass, max_mass = 0.0, 1.0
mean_mass, width_mass, n_sigmas = 0.5, 0.03, 2.5

# make the data and labels
raw_df = make_moons_mass(n_evts, min_mass, max_mass, mean=mean_mass,
                         sigma=width_mass, noise=noise, angle=angle, beta=0.60)
df = raw_df.copy()

y = df['label']
y_1hot = pd.concat([df['label_0'], df['label_1']], axis=1, sort=False)
if 'relegator' in model_type:
    y_1hot['label_rel'] = 0
if 'regress' not in model_type:
    y = y_1hot.copy()

df.drop(['label', 'label_0', 'label_1'], axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=42)
masses_train = X_train['m']
masses_test = X_test['m']
X_train.drop('m', axis=1, inplace=True)
X_test.drop('m', axis=1, inplace=True)

dropout_frac = 0.05
learning_rate = 1e-3
hidden_nodes = [40, 40, 20]

# initialize model...
clf = None
rel_loss = None
n_outs = 0

if model_type == 'regress':
    clf = regressor_model(len(X_train.columns), hidden_nodes,
                          input_dropout=dropout_frac, learning_rate=learning_rate)
    n_outs = 1
elif model_type == 'signif_regress':
    mod_loss = regress_signif_loss(sig_frac)
    clf = signif_regressor_model(len(X_train.columns), hidden_nodes, mod_loss,
                          input_dropout=dropout_frac, learning_rate=learning_rate)
    n_outs = 1
elif model_type == 'nn_binary':
    clf = binary_softmax_model(len(X_train.columns), hidden_nodes,
                               input_dropout=dropout_frac, learning_rate=learning_rate)
    n_outs = 2
elif model_type == 'relegator':
    rel_loss = relegator_loss(sig_frac,
                              reg_min=mean_mass - width_mass, reg_max=mean_mass + width_mass)
    clf = relegator_model(len(X_train.columns), hidden_nodes, rel_loss,
                          input_dropout=dropout_frac, learning_rate=learning_rate)
    n_outs = 3
else:
    print('error: model type \"' + model_type + '\" not quite ready...')
    exit

print(clf.summary())

# train model...
train_results_df = []
train_results_df = train_model(clf, X_train, y_train, X_test, y_test, n_epochs,
                               batch_size=1024, verbose=1, ot_shutoff=True,
                               ot_shutoff_depth=ot_cutoff_depth)

# if 'relegator' in model_type:
#     train_results_df = train_model(clf, X_train, y_train, X_test, y_test, n_epochs,
#                                        batch_size=1024, verbose=1, ot_shutoff=True,
#                                        ot_shutoff_depth=ot_cutoff_depth)
# else:
#     train_results_df = train_model(clf, X_train, y_train, X_test, y_test, n_epochs,
#                                    batch_size=1024, verbose=1, ot_shutoff=True,
#                                    ot_shutoff_depth=ot_cutoff_depth)

print('\n... NN trained, plotting...\n')

y_pred_train, y_pred_test = [], []

fig = plt.figure(figsize=(9,6))
nbins = int(np.sqrt(n_evts)/2)

n_rows, n_cols = 2, 2

ax = plt.subplot(n_rows,n_cols, 1)
plt.plot(train_results_df['eps'], train_results_df['eval_accs'], label='train, dropout=' + str(dropout_frac))
plt.plot(train_results_df['eps'], train_results_df['train_accs'], label='train')
plt.plot(train_results_df['eps'], train_results_df['test_accs'], label='test')
plt.plot(train_results_df['eps'], train_results_df['test_acc_sma'], label='test, sma' + str(ot_cutoff_depth))
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.legend(loc='lower right')
plt.ylabel('accuracy')
plt.xlabel('epoch')

ax = plt.subplot(n_rows, n_cols, 2)
plt.plot(train_results_df['eps'], train_results_df['eval_loss'], label='train, dropout=' + str(dropout_frac))
plt.plot(train_results_df['eps'], train_results_df['train_loss'], label='train')
plt.plot(train_results_df['eps'], train_results_df['test_loss'], label='test')
plt.plot(train_results_df['eps'], train_results_df['test_loss_sma'], label='test sma' + str(ot_cutoff_depth))
#plt.plot(history.history['val_acc'])
plt.title('loss (' + loss_type + ')')
plt.legend(loc='upper right')
plt.ylabel('loss')
plt.xlabel('epoch')

ax = plt.subplot(n_rows, n_cols, 3)
if 'regress' in model_type:
    y_pred_train = clf.predict(X_train).ravel()
    y_pred_test = clf.predict(X_test).ravel()

    # ROC metrics
    fpr_reg_clf, tpr_reg_clf, thresholds_reg_clf = metrics.roc_curve(y_test, y_pred_test)
    auc_keras = metrics.auc(fpr_reg_clf, tpr_reg_clf)
    print('\nauc score on test: %0.4f' % auc_keras, '\n')

    # plot ROC curve...
    roc_auc = metrics.auc(fpr_reg_clf, tpr_reg_clf)
    ax.plot(fpr_reg_clf, tpr_reg_clf, lw=1, label='ROC (area = %0.3f)'%(roc_auc))
    ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6)) #, label='luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('receiver operating characteristic')
    plt.legend(loc='lower right')
    plt.grid()

else: #elif model_type == 'nn_binary':
    y_1hot_pred_train = clf.predict(X_train)
    y_pred_train = y_1hot_pred_train.argmax(axis=1)

    y_1hot_pred_test = clf.predict(X_test)
    y_pred_test = y_1hot_pred_test.argmax(axis=1)

    #print(y_1hot_pred_train, y_train, y_pred_train)

    # confusion matrices... plot from test only
    print('\ntesting results...')
    class_labels = ['type 0', 'type 1']
    if model_type == 'relegator':
        class_labels.append('releg.')
        y_test = y_test.append({'label_0': 0, 'label_1': 0, 'label_rel': 1}, ignore_index=True)
        y_1hot_pred_test = np.append(y_1hot_pred_test, [[0,0,1]], axis=0)

    plot_confusion_matrix(y_test.to_numpy().argmax(axis=1),
                          y_1hot_pred_test.argmax(axis=1), class_labels, ax,
                          normalize=True, title='confusion matrix, test')

opt_df = 0.0
if 'regress' in model_type:
    ax = plt.subplot(n_rows,n_cols, 4)
    test_dict = {'x1':X_test['x1'], 'x2':X_test['x2'], 'm':masses_test, 'y':y_test, 'pred':y_pred_test}
    test_df = pd.DataFrame(test_dict)
    min_df, max_df = np.amin(y_pred_test), np.amax(y_pred_test)
    nslices = 100
    dvals = np.linspace(min_df, max_df, num=nslices + 1)
    n_sig, n_bkgd, pwr = [], [], []
    for d in dvals:
        n_sig.append(len(test_df['m'][test_df.y == 1][test_df.pred >= d][np.abs(df.m - mean_mass) < n_sigmas*width_mass]))
        n_bkgd.append(len(test_df['m'][test_df.y == 0][test_df.pred >= d][np.abs(df.m - mean_mass) < n_sigmas*width_mass]))
        pwr.append(n_sig[-1]*sig_frac / np.sqrt(n_sig[-1]*sig_frac + n_bkgd[-1]*(1-sig_frac)))
    opt_pwr = np.nanmax(pwr)
    opt_idx = pwr.index(opt_pwr)
    opt_df  = dvals[opt_idx]
    ax.plot(dvals, pwr, label='sig fraction = ' + str(sig_frac))
    plt.xlabel('decision function value')
    plt.ylabel(r'$s / \sqrt{s+b}$')
    plt.axvline(x=opt_df, color='lightgray', dashes=(1,1))
    plt.legend(loc='lower right')

# if 'pickle' in sys.argv:
#     output_fname += ':opt_df=' + str(np.round(opt_df, 6))
#     with open(output_fname + ".pkl", 'wb') as f:
#         pickle.dump(dvals, f)
#         pickle.dump(pwr, f)
#         f.close()

plt.tight_layout()

# # # # # # plot decision boundaries
fig = plt.figure(figsize=(9,5.5))
ax = plt.subplot(1,1,1)
x1_mesh, x2_mesh, class_mesh = predict_bound_class(clf, df, n_outs) #, opt_df=opt_df)
vmin, vmax = 0, 1
if model_type == 'relegator':
    vmax = 2
ax.contourf(x1_mesh, x2_mesh, class_mesh, alpha=0.4)
# im = ax.contourf(x1_mesh, x2_mesh, class_mesh, alpha=0.4, cmap='PuOr_r', vmin=vmin, vmax=vmax)
# fig.colorbar(im, ax=ax)

plot_xs(raw_df, ax)
plt.title('noise = ' + str(noise) + ', angle = ' + str(angle) + ', epochs = ' + str(n_epochs))

plt.tight_layout()

# # # # # # plot mass histograms after optimal cut
print('\napplying optimal cut to dataset with sig_frac = ' + str(sig_frac) + '...')

fig = plt.figure(figsize=(11,6))
weighted_n_evts = n_evts # 50000
weighted_df = make_moons_mass(weighted_n_evts, min_mass, max_mass, mean=mean_mass, sigma=width_mass,
                              noise=noise, angle=angle, beta=0.60, sig_fraction=sig_frac)
y_weighted = weighted_df['label']
# y_weighted_1hot = pd.concat([weighted_df['label_0'], weighted_df['label_1']], axis=1, sort=False)
xs_weighted = weighted_df.drop(['label', 'label_0', 'label_1', 'm'], axis=1)

# ax = plt.subplot(1,2,1)
ax = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
cents, occs, bkgds = hist_ms(weighted_df, min_mass, max_mass, nbins, ax)
plt.xlim((min_mass, max_mass))
plt.title('masses, sig\_frac = ' + str(sig_frac))
plt.legend(loc='upper right')

ax = plt.subplot2grid((3, 2), (2, 0))
ax.yaxis.grid(True)
hist_diff_signif(cents, occs, bkgds)
plt.ylim((-10, 10))
plt.xlim((min_mass, max_mass))

# ax = plt.subplot(1,2,2)
ax = plt.subplot2grid((3, 2), (0, 1), rowspan=2)

raw_signif, pass_signif, n_raw_bkgd, n_raw_sig, n_pass_bkgd, n_pass_sig = 0, 0, 0, 0, 0, 0
cents, occs, bkgds = 0, 0, 0
if 'regress' in model_type:
    weighted_df['pred'] = clf.predict(xs_weighted).ravel()
    cents, occs, bkgds = hist_cut_ms(weighted_df, opt_df, min_mass, max_mass, nbins, ax)
    raw_signif, pass_signif, n_raw_bkgd, n_raw_sig, n_pass_bkgd, n_pass_sig = compute_signif_regress(weighted_df, opt_df, mean_mass, width_mass, n_sigmas)

else: # model_type == 'nn_binary':
    weighted_df['prob_0'] = clf.predict(xs_weighted)[:,0]
    weighted_df['prob_1'] = clf.predict(xs_weighted)[:,1]
    if model_type == 'relegator':
        weighted_df['prob_rel'] = clf.predict(xs_weighted)[:,1]

    cents, occs, bkgds = hist_softmax_cut_ms(weighted_df, min_mass, max_mass, nbins, ax)
    raw_signif, pass_signif, n_raw_bkgd, n_raw_sig, n_pass_bkgd, n_pass_sig = compute_signif_binary(weighted_df, mean_mass, width_mass, n_sigmas)

plt.xlim((min_mass, max_mass))
title_str = 'masses pass nn'
if opt_df > 0:
    title_str += ', optimal d.f. = %0.3f' % opt_df
plt.title(title_str)
plt.legend(loc='upper right')

ax = plt.subplot2grid((3, 2), (2, 1))
ax.yaxis.grid(True)
hist_diff_signif(cents, occs, bkgds)
plt.ylim((-10, 10))
plt.xlim((min_mass, max_mass))

plt.tight_layout()

print('\nraw analysis significance:\t', str(raw_signif))
print('pass analysis significance:\t', str(pass_signif))

if 'write_results' in sys.argv:
    print('\nwriting results to file ' + './' + model_type + '_results.txt')
    f = open('./' + model_type + '_results.txt', 'a+')
    out_arr = [sys.argv[2], str(noise), str(angle), str(sig_frac), str(n_epochs),
               '%0.4f' % train_results_df['eps'].iloc[-1],
               '%0.4f' % train_results_df['eval_accs'].iloc[-1],
               '%0.4f' % train_results_df['train_accs'].iloc[-1],
               '%0.4f' % train_results_df['test_accs'].iloc[-1],
               '%0.4f' % train_results_df['eval_loss'].iloc[-1],
               '%0.4f' % train_results_df['train_loss'].iloc[-1],
               '%0.4f' % train_results_df['test_loss'].iloc[-1],
               str(n_raw_bkgd), str(n_raw_sig),
               str(n_pass_bkgd), str(n_pass_sig),
               '%0.3f' % raw_signif,
               '%0.3f' % pass_signif]
    line = ','.join(out_arr) + '\n'
    f.write(line)
    f.close()

if 'noplot' not in sys.argv:
    plt.show()
