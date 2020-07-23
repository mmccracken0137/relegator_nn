#!/usr/bon/env python
'''
Fits arbitrary model to moons+mass data.  Command line is

python master_moons.py <config_file>.json

model can take the values: regress, nn_binary, relegator

Last three arguments may be omitted to run with default values.
'''

from colorama import Fore, Back, Style
import pandas as pd
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import sys, os
import pickle
import json
import time

from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.utils.multiclass import unique_labels
import tensorflow as tf

from moons_tools_2 import *
from classifier_classes import *

# check if tf2.0 eager exec is working
print('Executing eagerly today -->  ' + str(tf.executing_eagerly()) + '!\n')
if not tf.executing_eagerly():
    sys.exit('this code works with tf2.0+ (eager execution) only')

pd.options.mode.chained_assignment = None  # default='warn'

# to use latex with matplotlib
from matplotlib import rc
rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# # # # # # # # # # # # # # # # # # # #
# read in parameters from json config file
with open(sys.argv[1], 'r') as f:
    config_pars = json.loads(f.read())

model_type = config_pars['model']['model_type'] # sys.argv[1]
allowed_models = ['regress', 'binary_softmax', 'relegator',
                  'mod_binary', 'relegator_factor', 'relegator_diff']
if model_type not in allowed_models:
    print('error: model type \"' + model_type + '\" undefined')
    sys.exit()

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

# # # # # # # # # # # # # # # # # # # #
# generate train and test datasets...
train_df = None
if 'gen_data' in sys.argv:
    print('generating training dataset...')
    train_df = make_moons_mass(n_evts, min_mass, max_mass,
                               mean=mean_mass, sigma=width_mass,
                               noise=noise, angle=angle, beta=bkgd_beta)
else:
    print('unpickling training dataset from ' + config_pars['data']['train_data_file'])
    with open(config_pars['data']['train_data_file'], 'rb') as f:
        train_df = pickle.load(f)

y = train_df['label']
y_1hot = pd.concat([train_df['label_0'], train_df['label_1']], axis=1, sort=False)
if 'relegator' in model_type:
    y_1hot['label_rel'] = 0
if 'regress' not in model_type:
    y = y_1hot.copy()

train_df.drop(['label', 'label_0', 'label_1'], axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=test_fraction)
                                                    # random_state=ttsplit_random_state)
masses_train = X_train['m']
masses_test = X_test['m']
X_train.drop('m', axis=1, inplace=True)
X_test.drop('m', axis=1, inplace=True)

train_ds = np_to_tfds(X_train, y_train, batch_size=len(X_train))
test_ds = np_to_tfds(X_test, y_test, batch_size=len(X_test))

# # # # # # # # # # # # # # # # # # # #
# model and optimization inits
input_dropout = config_pars['model']['input_dropout'] # 0.05
learning_rate = config_pars['model']['learning_rate'] # 1e-3
hidden_nodes = config_pars['model']['hidden_nodes'] # [40, 60, 40, 20]
bias = config_pars['model']['bias'] # True
signif_type = config_pars['model']['signif_type'] # True

n_epochs = config_pars['run']['n_epochs'] # 100
ot_cutoff_depth = config_pars['run']['ot_cutoff_depth'] # 20
ot_cutoff = config_pars['run']['ot_cutoff'] # True

n_inputs = len(X_train.columns)
n_outputs = 1
if 'regress' not in model_type:
    n_outputs = len(y_train.columns)

n_hidden = len(hidden_nodes)
layers = []
tf.keras.backend.set_floatx('float32')

model_clf = None
n_ouputs = 0
if model_type == 'regress':
    model_clf = RegressorClf()
    n_ouputs = 1
elif model_type == 'binary_softmax':
    model_clf = BinarySoftmaxClf()
    model_clf.set_signal_idx(1)
    model_clf.set_background_idxs([0])
    n_ouputs = 2
elif model_type == 'mod_binary':
    model_clf = ModBinarySoftmaxClf()
    model_clf.set_train_masses(masses_train, mean_mass, n_sigmas * width_mass)
    model_clf.set_test_masses(masses_test)
    model_clf.set_test_fraction(test_fraction)
    model_clf.gen_train_peak_mask()
    model_clf.gen_test_peak_mask()
    model_clf.set_signal_fraction(sig_frac)
    model_clf.set_signif_type(signif_type) # 'categ'
    model_clf.set_signal_idx(1)
    model_clf.set_background_idxs([0])
    n_ouputs = 2
elif 'relegator' in model_type:
    if model_type == 'relegator':
        model_clf = RelegatorClf()
    elif model_type == 'relegator_factor':
        model_clf = RelegatorFactorClf()
    elif model_type == 'relegator_diff':
        model_clf = RelegatorDiffClf()
    model_clf.set_train_masses(masses_train, mean_mass, n_sigmas * width_mass)
    model_clf.set_test_masses(masses_test)
    model_clf.set_test_fraction(test_fraction)
    model_clf.gen_train_peak_mask()
    model_clf.gen_test_peak_mask()
    model_clf.set_signal_fraction(sig_frac)
    model_clf.set_signif_type(signif_type) # 'categ'
    model_clf.set_signal_idx(1)
    model_clf.set_background_idxs([0])
    n_ouputs = 3

print('model type:', model_clf.name)
print('loss function type:', model_clf.loss_type)

model_clf.build_model(nodes=hidden_nodes, bias=bias, n_ins=n_inputs, n_outs=n_outputs,
                      input_dropout=input_dropout)
print(model_clf.model.summary())
model_clf.init_optimizer(lr=learning_rate)

# numpy types for post-train plotting...
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test  = X_test.to_numpy()
y_test  = y_test.to_numpy()

start_time = time.time()
model_clf.train(train_ds, test_ds, n_epochs,
                ot_cutoff=ot_cutoff, ot_cutoff_depth=ot_cutoff_depth)
train_results_df = model_clf.train_results
train_time = time.time() - start_time

print('\n... NN trained, plotting...\n')

fig = plt.figure(figsize=(9,6))
nbins = int(np.sqrt(n_evts)/2)

n_rows, n_cols = 2, 2
ax = plt.subplot(n_rows,n_cols, 1)
plt.plot(train_results_df['eps'], train_results_df['train_accs'], label='train, dropout (' + str(input_dropout) + ')')
plt.plot(train_results_df['eps'], train_results_df['eval_accs'], label='train dataset')
plt.plot(train_results_df['eps'], train_results_df['test_accs'], label='test dataset')
plt.title('model accuracy')
plt.legend(loc='lower right')
plt.ylabel('accuracy')
plt.xlabel('epoch')

ax = plt.subplot(n_rows, n_cols, 2)
plt.plot(train_results_df['eps'], train_results_df['train_loss'], label='train, dropout (' + str(input_dropout) + ')')
plt.plot(train_results_df['eps'], train_results_df['eval_loss'], label='train dataset')
plt.plot(train_results_df['eps'], train_results_df['test_loss'], label='test dataset')
# plt.yscale('log')
plt.title('loss (' + model_clf.loss_type + ')')
plt.legend(loc='upper right')
plt.ylabel('loss')
plt.xlabel('epoch')

y_pred_train, y_pred_test = [], []

ax = plt.subplot(n_rows, n_cols, 3)
if 'regress' in model_type:
    y_pred_train = model_clf.model.predict(X_train).ravel()
    y_pred_test = model_clf.model.predict(X_test).ravel()

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
    y_1hot_pred_train = model_clf.model.predict(X_train)
    y_pred_train = y_1hot_pred_train.argmax(axis=1)

    y_1hot_pred_test = model_clf.model.predict(X_test)
    y_pred_test = y_1hot_pred_test.argmax(axis=1)

    #print(y_1hot_pred_train, y_train, y_pred_train)

    # confusion matrices... plot from test only
    print('\ntesting results...')
    # class_labels = ['type 0', 'type 1']
    class_labels = ['bakcground', 'signal']
    if model_type == 'relegator':
        class_labels.append('releg.')
        # y_test = y_test.append({'label_0': 0, 'label_1': 0, 'label_rel': 1}, ignore_index=True)
        y_test = np.append(y_test, [[0, 0, 1]], axis=0)
        y_1hot_pred_test = np.append(y_1hot_pred_test, [[0,0,1]], axis=0)

    plot_confusion_matrix(y_test.argmax(axis=1),
                          y_1hot_pred_test.argmax(axis=1), class_labels, ax,
                          normalize=True, title='confusion matrix, test')

opt_thr = 0.0
if 'regress' in model_type:
    ax = plt.subplot(n_rows,n_cols, 4)
    test_dict = {'x1':X_test[:,0], 'x2':X_test[:,1], 'm':masses_test, 'y':y_test, 'pred':y_pred_test}
    test_df = pd.DataFrame(test_dict)
    min_df, max_df = np.amin(y_pred_test), np.amax(y_pred_test)
    nslices = 100
    dvals = np.linspace(min_df, max_df, num=nslices + 1)
    n_sig, n_bkgd, pwr = [], [], []
    for d in dvals:
        n_sig.append(len(test_df['m'][test_df.y == 1][test_df.pred >= d][np.abs(test_df.m - mean_mass) < n_sigmas*width_mass]))
        n_bkgd.append(len(test_df['m'][test_df.y == 0][test_df.pred >= d][np.abs(test_df.m - mean_mass) < n_sigmas*width_mass]))
        pwr.append(n_sig[-1]*sig_frac / np.sqrt(n_sig[-1]*sig_frac + n_bkgd[-1]*(1-sig_frac)))
    opt_pwr = np.nanmax(pwr)
    opt_idx = pwr.index(opt_pwr)
    opt_thr  = dvals[opt_idx]
    ax.plot(dvals, pwr, label='sig fraction = ' + str(sig_frac))
    plt.xlabel('decision threshold')
    plt.ylabel(r'$s / \sqrt{s+b}$')
    plt.axvline(x=opt_thr, color='lightgray', dashes=(1,1))
    plt.legend(loc='lower right')

plt.tight_layout()

# # # # # # # # # # # # # # # # # #
# if 'pickle' in sys.argv:
    # output_fname = 'reg_nn_pwrs/train_evts=' + str(n_evts)
    # output_fname += ':noise=' + str(noise)
    # output_fname += ':angle=' + str(angle)
    # output_fname += ':epochs=' + str(n_epochs)
    # output_fname += ':sig_frac=' + str(sig_frac)
    # output_fname += ':opt_thr=' + str(np.round(opt_thr, 6))
    # with open(output_fname + ".pkl", 'wb') as f:
    #     pickle.dump(dvals, f)
    #     pickle.dump(pwr, f)
    #     f.close()


# # # # # # # # # # # # # # # # # #
# plot decision boundaries
fig = plt.figure(figsize=(9,5.5))
ax = plt.subplot(1,1,1)
x1_mesh, x2_mesh, class_mesh = predict_bound_class(model_clf.model, train_df, model_clf.n_outputs, opt_thr=opt_thr)
# custom colormap for contour
vmin, vmax = 0, 2
rel_cmap = relegator_cmap()
plt.register_cmap(cmap=rel_cmap)
cont = ax.contourf(x1_mesh, x2_mesh, class_mesh, alpha=0.45,
                   cmap='RelegatorCMap', vmin=vmin, vmax=vmax,
                   levels=[-0.5, 0.5, 1.5, 2.5])
# cbar = fig.colorbar(cont)

if 'regress' in model_type:
    plot_xs(X_train, y_train, ax)
else:
    plot_xs(X_train, y_train[:,1], ax)
plt.title('noise = ' + str(noise) + ', angle = ' + str(angle) + ', epochs = ' + str(n_epochs))
plt.tight_layout()

fig = plt.figure(figsize=(9,5.5))
ax = plt.subplot(1,1,1)
cont = ax.contourf(x1_mesh, x2_mesh, class_mesh, alpha=0.5,
                   cmap='RelegatorCMap', vmin=vmin, vmax=vmax,
                   levels=[-0.5, 0.5, 1.5, 2.5])
plt.title('noise = ' + str(noise) + ', angle = ' + str(angle) + ', epochs = ' + str(n_epochs))
plt.tight_layout()

# # # # # # # # # # # # # # # # # #
# plot mass histograms after optimal cut
print('\napplying optimal cut to dataset with sig_frac = ' + str(sig_frac) + '...')

weighted_n_evts, weighted_df = 0, None
if 'gen_data' in sys.argv:
    print('generating weighted dataset...')
    weighted_n_evts = config_pars['data']['weighted_n_events'] # 50000
    weighted_df = make_moons_mass(weighted_n_evts, min_mass, max_mass,
                                  mean=mean_mass, sigma=width_mass, noise=noise,
                                  angle=angle, beta=bkgd_beta, sig_fraction=sig_frac)
else:
    print('unpickling weighted dataset from ' + config_pars['data']['weighted_data_file'])
    with open(config_pars['data']['weighted_data_file'], 'rb') as f:
        weighted_df = pickle.load(f)
        weighted_n_evts = len(weighted_df.index)

y_weighted = weighted_df['label']
xs_weighted = weighted_df.drop(['label', 'label_0', 'label_1', 'm'], axis=1)

fig = plt.figure(figsize=(11,6))
ax = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
cents, occs, bkgds = hist_ms(weighted_df, min_mass, max_mass, nbins, ax,
                             sig_limits=(mean_mass - n_sigmas*width_mass, mean_mass + n_sigmas*width_mass))
plt.xlim((min_mass, max_mass))
plt.title('masses, sig\_frac = ' + str(sig_frac))
plt.legend(loc='upper right')

ax = plt.subplot2grid((3, 2), (2, 0))
ax.yaxis.grid(True)
# hist_diff_signif(cents, occs, bkgds)
# plt.ylim((-10, 10))
hist_residuals(cents, occs, bkgds,
               sig_limits=(mean_mass - n_sigmas*width_mass, mean_mass + n_sigmas*width_mass))
plt.xlim((min_mass, max_mass))

ax = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
raw_signif, pass_signif, n_raw_bkgd, n_raw_sig, n_pass_bkgd, n_pass_sig = 0, 0, 0, 0, 0, 0
cents, occs, bkgds = 0, 0, 0
if 'regress' in model_type:
    weighted_df['pred'] = model_clf.model.predict(xs_weighted).ravel()
    cents, occs, bkgds = hist_cut_ms(weighted_df, opt_thr, min_mass, max_mass, nbins, ax,
                                     sig_limits=(mean_mass - n_sigmas*width_mass,
                                                 mean_mass + n_sigmas*width_mass))
    raw_signif, pass_signif, n_raw_bkgd, n_raw_sig, n_pass_bkgd, n_pass_sig = compute_signif_regress(weighted_df, opt_thr, mean_mass, width_mass, n_sigmas)

else: # model_type == 'nn_binary':
    weighted_df['prob_0'] = model_clf.model.predict(xs_weighted)[:,0]
    weighted_df['prob_1'] = model_clf.model.predict(xs_weighted)[:,1]
    if model_type == 'relegator':
        weighted_df['prob_rel'] = model_clf.model.predict(xs_weighted)[:,1]

    cents, occs, bkgds = hist_softmax_cut_ms(weighted_df, min_mass, max_mass, nbins, ax,
                                             sig_limits=(mean_mass - n_sigmas*width_mass,
                                                         mean_mass + n_sigmas*width_mass))
    raw_signif, pass_signif, n_raw_bkgd, n_raw_sig, n_pass_bkgd, n_pass_sig = compute_signif_binary(weighted_df, mean_mass, width_mass, n_sigmas)

plt.xlim((min_mass, max_mass))
title_str = 'post-cut masses'
if opt_thr > 0:
    title_str += ', opt. threshold = %0.3f' % opt_thr
plt.title(title_str)
plt.legend(loc='upper right')

ax = plt.subplot2grid((3, 2), (2, 1))
ax.yaxis.grid(True)
# hist_diff_signif(cents, occs, bkgds)
# plt.ylim((-10, 10))
hist_residuals(cents, occs, bkgds,
               sig_limits=(mean_mass - n_sigmas*width_mass, mean_mass + n_sigmas*width_mass))
plt.xlim((min_mass, max_mass))

plt.tight_layout()

print('\nraw analysis significance:\t', str(raw_signif))
print('pass analysis significance:\t', str(pass_signif))

if 'write_results' in sys.argv:
    # print('\nwriting results to file ' + './fit_results/' + model_type + '_results.txt')
    # f = open('./fit_results/' + model_type + '_results.txt', 'a+')

    # check if results file exists...
    write_header = not os.path.exists('./fit_results/fit_results.txt')
    print('\nwriting results to file ' + './fit_results/fit_results.txt')
    f = open('./fit_results/fit_results.txt', 'a+')
    if write_header:
        out_arr = ["model_type", "\tnoise", "angle", "sig_frac", "n_epochs",
                   "train_time",
                   'epochs',
                   'eval_acc',
                   'train_acc',
                   'test_acc',
                   'eval_loss',
                   'train_loss',
                   'test_loss',
                   'weighted_n_events',
                   'n_raw_bkgd', 'n_raw_sig',
                   'n_pass_bkgd', 'n_pass_sig',
                   'raw_signif',
                   'pass_signif']
        line = ','.join(out_arr) + '\n'
        f.write(line)

    out_arr = [model_type, '\t' + str(noise), str(angle), str(sig_frac), str(n_epochs),
               '%0.4f' % train_time,
               str(train_results_df['eps'].iloc[-1]),
               '%0.4f' % train_results_df['eval_accs'].iloc[-1],
               '%0.4f' % train_results_df['train_accs'].iloc[-1],
               '%0.4f' % train_results_df['test_accs'].iloc[-1],
               '%0.4f' % train_results_df['eval_loss'].iloc[-1],
               '%0.4f' % train_results_df['train_loss'].iloc[-1],
               '%0.4f' % train_results_df['test_loss'].iloc[-1],
               str(weighted_n_evts),
               str(n_raw_bkgd), str(n_raw_sig),
               str(n_pass_bkgd), str(n_pass_sig),
               '%0.3f' % raw_signif,
               '%0.3f' % pass_signif]
    line = ','.join(out_arr) + '\n'
    f.write(line)
    f.close()

if 'noplot' not in sys.argv:
    plt.show()
