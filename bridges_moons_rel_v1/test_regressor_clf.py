#!/usr/bon/env python
'''
This code fits and applies the general regression/relegation classifier to toy-model
moons+mass dataset.  User must supply hyperparameters in this code,
rather than supplying a config file.

Command line accepts/requires tags that will be placed in the results
file output.

Example command:
python test_clf.py write_results tag:noise=0.3 tag:angle=1.4 tag:foo=bar tag:sig_frac=0.01

Relegation classifier code is imported via relegation_clf.py.

This code is intended to be a MWE with relegation_clf.py.

Below, the feature of merit (fom) is the feature that may/may not be fit, but
is used to calculate the significance of signal (equivalent to mass parameter).
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

import sys, os
import pickle
import json
import time, datetime
from colorama import Fore, Back, Style
import socket

from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.utils.multiclass import unique_labels
import tensorflow as tf

# import some helper functions from moons_tools_2
from moons_tools_2 import *

# import relegation_clf module
# sys.path.insert(1, '/Users/mmccracken/office_comp/relegation_clf/relegation_clf_v1')
from relegation_clf_v1 import *

# check if tf2.0 eager exec is working
if not tf.executing_eagerly():
    sys.exit('this code works with tf2.0+ (eager execution) only')

pd.options.mode.chained_assignment = None  # default='warn'

# to use latex with matplotlib
#rc('font', **{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

# user-supplied parameters
model_type = 'regressor'

# fraction of train data split for validation, user-defined
validation_fraction = 0.25
validation_random_seed = 42

# relevant labels/indices from dataframe
# will you use a feature of merit to assess significance?
fom_name = 'mass'
# ... do you want to include it as a training feature?
train_on_fom = False

# parse some command-line info in the form of
# tag:name=val
results_file_info = {}
for i in range(len(sys.argv)):
    if 'tag:' in sys.argv[i]:
        results_file_info[sys.argv[i].split(":")[-1].split("=")[0]] = sys.argv[i].split("=")[-1]


print("HERE", results_file_info)

# truth class information
target_class_name = 'truth_class'
signal_idx = 1
background_idx = [0]
training_batch_size = 0 # TKTKTK need to clean this up later... if this is 0, use all events in a batch

# read in training dataframe...
train_df = pd.read_csv('datasets/train_ds_' + str(results_file_info['dataset']) + '.csv')
# get number of training events from train_df
train_n_evts = train_df.shape[0]

# separate truth targets...
y_cat_idxs = train_df[target_class_name]
# make 1-hot targets...
#y_1hot = pd.get_dummies(train_df[target_class_name], prefix=target_class_name)
# add extra truth_class column for relegator class TKTKTK move this into the module?
#y_1hot[target_class_name + '_relegate'] = 0

# read in the weighted dataset so that we can calculate some
# quantities that are important for fitting
weighted_df = pd.read_csv('datasets/weighted_ds_' + str(results_file_info['dataset']) + '.csv')
y_weighted_cat_idxs = weighted_df[target_class_name]
#y_weighted_1hot = pd.get_dummies(weighted_df[target_class_name], prefix=target_class_name)
#y_weighted_1hot[target_class_name + '_relegate'] = 0
weighted_n_evts = len(weighted_df.index)
weighted_n_sig = len(weighted_df[weighted_df['truth_class'] == 1])
weighted_n_bkgd = len(weighted_df[weighted_df['truth_class'] == 0])
weighted_sig_frac = weighted_n_sig / weighted_n_evts

# parameters for significance calculation on feature of merit...
# calculate mean and standard deviation from signal events in train_df
fom_mean = 0.5 # train_df[fom_name][train_df[target_class_name] == signal_idx].mean()
fom_std  = 0.2 # train_df[fom_name][train_df[target_class_name] == signal_idx].std()
fom_n_sigmas = 2.5
fom_min  = fom_mean - fom_n_sigmas * fom_std
fom_max  = fom_mean + fom_n_sigmas * fom_std

# drop target/class info from dataframe...
to_drop = [target_class_name]
train_df.drop(to_drop, axis=1, inplace=True)

# split training dataframe into train/validation
X_train, X_valid, y_train, y_valid = train_test_split(train_df, y_cat_idxs, #y_1hot,
                                                      test_size=validation_fraction)
                                                      # random_state=validation_random_state)

# make separate dfs for the feature of merit values
fom_train = X_train[fom_name]
fom_valid = X_valid[fom_name]
# if we don't train on the fom (i.e., it's not an input feature), drop it from the X dfs
if not train_on_fom:
    X_train.drop(fom_name, axis=1, inplace=True)
    X_valid.drop(fom_name, axis=1, inplace=True)

# convert the input and target dfs to tensorflow datasets
input_feats_arr = list(X_train.columns)

train_tf_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(X_train[input_feats_arr].values, tf.float32),
                                                       tf.cast(y_train.values, tf.int32)))
valid_tf_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(X_valid[input_feats_arr].values, tf.float32),
                                                       tf.cast(y_valid.values, tf.int32)))

if training_batch_size == 0:
    train_tf_dataset = train_tf_dataset.shuffle(len(X_train)).batch(len(X_train))
    valid_tf_dataset = valid_tf_dataset.shuffle(len(X_valid)).batch(len(X_valid))
else:
    train_tf_dataset = train_tf_dataset.shuffle(len(X_train)).batch(training_batch_size)
    valid_tf_dataset = valid_tf_dataset.shuffle(len(X_valid)).batch(training_batch_size)

# # # # # # # # # # # # # # # # # # # #
# model/NN parameters
# dropout fraction for input layer
input_dropout_fraction = 0.05
# list of numbers of nodes in each hidden layer
hidden_nodes = []
if 'net' in results_file_info:
    for lay in results_file_info['net'].split('x'):
        hidden_nodes.append(int(lay))
else:
    hidden_nodes = [40, 40, 20]

# number of hidden layers
n_hidden_layers = len(hidden_nodes)
# add bias at each hidden layer?
use_bias = True
# type of significance calculation... only proba works for now...
# signif_calc_type = 'proba' #none'
# number of input features
n_input_feats = len(X_train.columns)
# number of output classes/features in the target
n_output_feats = 1 # len(y_train.columns)

# # # # # # # # # # # # # # # # # # # #
# training/optimization parameters
# max number of epochs to run
max_n_epochs = 2000
# learning rate
learning_rate = 5e-4
# apply overtraning cut-off?
apply_ot_cutoff = True
# if applied, the overtraining cutoff looks at the tren of the last N loss function values.
# if the slope is positive (from linear regression), training exits early.
# the variable below sets N
ot_cutoff_depth = 20

# this may speed up training, not sure TKTKTK
#tf.keras.backend.set_floatx('float64')

model_clf = RegressorClf()
# model_clf.set_train_fom(fom_train)
# model_clf.set_test_fom(fom_valid)
model_clf.set_fom_minmax(min=fom_min, max=fom_max)
model_clf.set_validation_fraction(validation_fraction)
# model_clf.gen_train_peak_mask()
# model_clf.gen_valid_peak_mask()
# model_clf.set_signif_type(signif_calc_type)
# model_clf.set_signal_idx(signal_idx)
# model_clf.set_background_idxs(background_idx)

model_clf.set_signal_fraction(weighted_sig_frac)
n_train_events = len(X_train)
model_clf.set_n_train(n_train_events)
n_valid_events = len(X_valid)
model_clf.set_n_valid(n_valid_events)
model_clf.set_n_weighted(weighted_n_evts)

print('model type:', model_clf.name)
print('loss function type:', model_clf.loss_type)

# Build the model.  Can't do the numbers of input and output nodes elegantly b/c
# data hasn't been loaded into tf yet...
model_clf.build_model(nodes=hidden_nodes, bias=use_bias, n_ins=n_input_feats,
                      n_outs=n_output_feats, input_dropout=input_dropout_fraction)
print(model_clf.model.summary())
model_clf.init_optimizer(lr=learning_rate)

start_time = time.time()
model_clf.train(train_tf_dataset, valid_tf_dataset, max_n_epochs,
                ot_cutoff=apply_ot_cutoff, ot_cutoff_depth=ot_cutoff_depth)
train_results_df = model_clf.train_results
train_time = time.time() - start_time

print('\n' + str(train_results_df['eps'].iloc[-1]) + ' training iterations took  ' + str(train_time) + ' seconds \n')

print('\n... NN trained, plotting...\n')

# numpy types for post-train plotting...
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_valid = X_valid.to_numpy()
y_valid = y_valid.to_numpy()

fig = plt.figure(figsize=(9,6))
nbins = int(np.sqrt(train_n_evts)/4)

n_rows, n_cols = 2, 2
ax = plt.subplot(n_rows,n_cols, 1)
plt.plot(train_results_df['eps'], train_results_df['train_accs'], label='train, dropout (' + str(input_dropout_fraction) + ')')
plt.plot(train_results_df['eps'], train_results_df['eval_accs'], label='train dataset')
plt.plot(train_results_df['eps'], train_results_df['test_accs'], label='test dataset')
plt.title('model accuracy')
plt.legend(loc='lower right')
plt.ylabel('accuracy')
plt.xlabel('epoch')

ax = plt.subplot(n_rows, n_cols, 2)
plt.plot(train_results_df['eps'], train_results_df['train_loss'], label='train, dropout (' + str(input_dropout_fraction) + ')')
plt.plot(train_results_df['eps'], train_results_df['eval_loss'], label='train dataset')
plt.plot(train_results_df['eps'], train_results_df['test_loss'], label='test dataset')
# plt.yscale('log')
plt.title('loss (' + model_clf.loss_type + ')')
plt.legend(loc='upper right')
plt.ylabel('loss')
plt.xlabel('epoch')

# ax = plt.subplot(n_rows, n_cols, 3)
# plt.plot(train_results_df['eps'], train_results_df['eval_sig'], label='train dataset')
# plt.plot(train_results_df['eps'], train_results_df['test_sig'], label='test dataset')
# # plt.yscale('log')
# plt.title(r'significance $(\sigma)$')
# plt.legend(loc='lower right')
# plt.ylabel('loss')
# plt.xlabel('epoch')

# # # # # # # #
y_pred_train, y_pred_valid = [], []
conf_matrix_valid = []

# ax = plt.subplot(n_rows, n_cols, 3)
y_pred_train = model_clf.model.predict(X_train).ravel()
y_pred_valid = model_clf.model.predict(X_valid).ravel()

# ROC metrics
fpr_reg_clf, tpr_reg_clf, thresholds_reg_clf = metrics.roc_curve(y_valid, y_pred_valid)
auc_keras = metrics.auc(fpr_reg_clf, tpr_reg_clf)
print('\nauc score on test: %0.4f' % auc_keras, '\n')

ax = plt.subplot(n_rows, n_cols, 3)
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

opt_thr = 0.0
ax = plt.subplot(n_rows,n_cols, 4)
test_dict = {'x1': X_valid[:,0], 'x2': X_valid[:,1],
             fom_name: fom_valid, 'y': y_valid, 'pred': y_pred_valid}
# test_dict = {'x1':X_valid[:,0], 'x2':X_valid[:,1], 'y':y_valid, 'pred':y_pred_valid}
test_df = pd.DataFrame(test_dict)
min_df, max_df = np.amin(y_pred_valid), np.amax(y_pred_valid)
nslices = 100
dvals = np.linspace(min_df, max_df, num=nslices + 1)
n_sig, n_bkgd, pwr = [], [], []
for d in dvals:
    n_sig.append(len(test_df[fom_name][test_df.y == 1][test_df.pred >= d][np.abs(test_df[fom_name] - fom_mean) < fom_n_sigmas*fom_std]))
    n_bkgd.append(len(test_df[fom_name][test_df.y == 0][test_df.pred >= d][np.abs(test_df[fom_name] - fom_mean) < fom_n_sigmas*fom_std]))
    pwr.append(n_sig[-1]*weighted_sig_frac /
               np.sqrt(n_sig[-1]*weighted_sig_frac + n_bkgd[-1]*(1-weighted_sig_frac)))
opt_pwr = np.nanmax(pwr)
opt_idx = pwr.index(opt_pwr)
opt_thr  = dvals[opt_idx]
ax.plot(dvals, pwr, label='sig fraction = ' + str(weighted_sig_frac))
plt.xlabel('decision threshold')
plt.ylabel(r'$s / \sqrt{s+b}$')
plt.axvline(x=opt_thr, color='lightgray', dashes=(1,1))
plt.legend(loc='lower right')

plt.tight_layout()

# plot confusion matrix
# fig = plt.figure(figsize=(9,6))
# ax = plt.subplot(1, 1, 1)
# if 'regress' not in model_type:
#     y_1hot_pred_train = model_clf.model.predict(X_train)
#     y_pred_train = y_1hot_pred_train.argmax(axis=1)
#
#     y_1hot_pred_valid = model_clf.model.predict(X_valid)
#     y_pred_valid = y_1hot_pred_valid.argmax(axis=1)
#
#     #print(y_1hot_pred_train, y_train, y_pred_train)
#
#     # confusion matrices... plot from test only
#     print('\ntesting results...')
#     # class_labels = ['type 0', 'type 1']
#     class_labels = ['bkgd', 'sig.']
#     if model_type == 'relegator':
#         class_labels.append('rel.')
#         # y_valid = y_valid.append({'label_0': 0, 'label_1': 0, 'label_rel': 1}, ignore_index=True)
#         y_valid = np.append(y_valid, [[0, 0, 1]], axis=0)
#         y_1hot_pred_valid = np.append(y_1hot_pred_valid, [[0,0,1]], axis=0)
#
#     cm_valid = gen_confusion_matrix(y_valid.argmax(axis=1),
#                                     y_1hot_pred_valid.argmax(axis=1),
#                                     normalize=False)
#     print(cm_to_str(cm_valid))
#     cm_train = gen_confusion_matrix(y_train.argmax(axis=1),
#                                     y_1hot_pred_train.argmax(axis=1),
#                                     normalize=False)
#     print(cm_to_str(cm_train))
#     ax, conf_matrix_valid = plot_confusion_matrix(y_valid.argmax(axis=1),
#                           y_1hot_pred_valid.argmax(axis=1), class_labels, ax,
#                           normalize=True, title='conf. matrix, val.')
#
# plt.tight_layout()

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
# plt.title('noise = ' + str(noise) + ', angle = ' + str(angle) + ', epochs = ' + str(n_epochs))
plt.tight_layout()

fig = plt.figure(figsize=(9,5.5))
ax = plt.subplot(1,1,1)
cont = ax.contourf(x1_mesh, x2_mesh, class_mesh, alpha=0.5,
                   cmap='RelegatorCMap', vmin=vmin, vmax=vmax,
                   levels=[-0.5, 0.5, 1.5, 2.5])
#plt.title('noise = ' + str(noise) + ', angle = ' + str(angle) + ', epochs = ' + str(n_epochs))
plt.title('moons+mass, epochs = ' + str(train_results_df['eps'].iloc[-1]))
plt.tight_layout()


# # # # # # # # # # # # # # # # # #
# plot mass histograms after optimal cut
# weighted_df = pd.read_csv('datasets/weighted_ds.csv')
#
# y_weighted_cat_idxs = weighted_df[target_class_name]
# y_weighted_1hot = pd.get_dummies(weighted_df[target_class_name], prefix=target_class_name)
# y_weighted_1hot[target_class_name + '_relegate'] = 0
#
# weighted_n_evts = len(weighted_df.index)
# weighted_n_sig = len(weighted_df[weighted_df['truth_class'] == 1])
# weighted_n_bkgd = len(weighted_df[weighted_df['truth_class'] == 0])
# sig_frac = weighted_n_sig / weighted_n_evts

X_weighted = weighted_df.copy()
to_drop = [target_class_name]
X_weighted.drop(to_drop, axis=1, inplace=True)
fom_weighted = X_weighted[fom_name]
if not train_on_fom:
    X_weighted.drop(fom_name, axis=1, inplace=True)

# to_drop = [target_class_name]
# weighted_df.drop(to_drop, axis=1, inplace=True)
# if not train_on_fom:
#     weighted_df.drop(fom_name, axis=1, inplace=True)
# X_weight = weighted_df.copy()

print('\napplying optimal cut to dataset with sig_frac = ' + str(weighted_sig_frac) + '...')

fig = plt.figure(figsize=(11,6))
ax = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
cents, occs, bkgds = hist_fom(weighted_df, fom_name, fom_min, fom_max, nbins, ax,
                             sig_limits=(fom_mean - fom_n_sigmas*fom_std,
                                         fom_mean + fom_n_sigmas*fom_std))
plt.xlim((fom_min, fom_max))
plt.title(fom_name + ', sig_frac = ' + str(weighted_sig_frac))
plt.legend(loc='upper right')

ax = plt.subplot2grid((3, 2), (2, 0))
ax.yaxis.grid(True)
# hist_diff_signif(cents, occs, bkgds)
# plt.ylim((-10, 10))
hist_residuals(cents, occs, bkgds,
               sig_limits=(fom_mean - fom_n_sigmas*fom_std,
                           fom_mean + fom_n_sigmas*fom_std))
plt.xlim((fom_min, fom_max))

ax = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
raw_signif, pass_signif, n_raw_bkgd, n_raw_sig, n_pass_bkgd, n_pass_sig = 0, 0, 0, 0, 0, 0
cents, occs, bkgds = 0, 0, 0

# if 'regress' in model_type:
weighted_df['pred'] = model_clf.model.predict(X_weighted).ravel()
cents, occs, bkgds = hist_cut_ms(weighted_df, fom_name, opt_thr, fom_min, fom_max, nbins, ax,
                                 sig_limits=(fom_mean - fom_n_sigmas*fom_std,
                                             fom_mean + fom_n_sigmas*fom_std))
raw_signif, pass_signif, n_raw_bkgd, n_raw_sig, n_pass_bkgd, n_pass_sig = compute_signif_regress(weighted_df, fom_name, opt_thr, fom_mean, fom_std, fom_n_sigmas)
# cents, occs, bkgds = hist_cut_ms(weighted_df, opt_thr, min_mass, max_mass, nbins, ax,
#                                  sig_limits=(mean_mass - n_sigmas*width_mass,
#                                              mean_mass + n_sigmas*width_mass))
# raw_signif, pass_signif, n_raw_bkgd, n_raw_sig, n_pass_bkgd, n_pass_sig = compute_signif_regress(weighted_df, opt_thr, mean_mass, width_mass, n_sigmas)

# else: # model_type == 'nn_binary':
# weighted_df['prob'] = model_clf.model.predict(X_weighted)[:,0]
# weighted_df['prob_1'] = model_clf.model.predict(X_weighted)[:,1]
# if model_type == 'relegator':
#     weighted_df['prob_rel'] = model_clf.model.predict(X_weighted)[:,1]

y_weighted_cat_idxs = y_weighted_cat_idxs.to_numpy()
# cm_weighted = gen_confusion_matrix(y_weighted_cat_idxs.argmax(axis=1),
#                                    model_clf.model.predict(X_weighted).argmax(axis=1),
#                                    normalize=False)
# print(cm_to_str(cm_weighted))

plt.xlim((fom_min, fom_max))
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
               sig_limits=(fom_mean - fom_n_sigmas*fom_std, fom_mean + fom_n_sigmas*fom_std))
plt.xlim((fom_min, fom_max))

plt.tight_layout()

print('\nraw analysis significance:\t', str(raw_signif))
print('pass analysis significance:\t', str(pass_signif))

# Confusion matrix calculations
def opt_cut(preds, thr):
    for i in range(len(preds)):
        preds[i] = int(preds[i] > thr)
    return preds

print('optimal threshhold is: ' + str(opt_thr))
y_train_pred_opt = model_clf.model.predict(X_train).ravel()
y_train_pred_opt = opt_cut(y_train_pred_opt, opt_thr)
cm_train = metrics.confusion_matrix(y_train, y_train_pred_opt)
print(cm_train)

y_valid_pred_opt = model_clf.model.predict(X_valid).ravel()
y_valid_pred_opt = opt_cut(y_valid_pred_opt, opt_thr)
cm_valid = metrics.confusion_matrix(y_valid, y_valid_pred_opt)
print(cm_valid)

y_weighted_pred_opt = model_clf.model.predict(X_weighted).ravel()
y_weighted_pred_opt = opt_cut(y_weighted_pred_opt, opt_thr)
cm_weighted = metrics.confusion_matrix(y_weighted_cat_idxs, y_weighted_pred_opt)
print(cm_weighted)

if 'write_results' in sys.argv:
    # print('\nwriting results to file ' + './fit_results/' + model_type + '_results.txt')
    # f = open('./fit_results/' + model_type + '_results.txt', 'a+')

    # check if results file exists...
    date_stamp = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    hostname = socket.gethostname().replace('.local', '')
    fname = './fit_results/fit_results_' + hostname
    fname += '_' + date_stamp + '.txt'
    write_header = not os.path.exists(fname)
    print('\nwriting results to file ' + fname)
    f = open(fname, 'a+')
    if write_header:
        out_arr = ["model_type", "host", "date", "dataset",
                   "noise", "angle",
                   "nominal_sig_frac", "sig_frac", "max_epochs",
                   'layers',
                   'epochs',
                   "train_n_events",
                   "valid_n_events",
                   "train_time",
                   'opt_reg_thresh',
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
                   'pass_signif',
                   'cm_train',
                   'cm_valid',
                   'cm_weighted']
        line = ','.join(out_arr) + '\n'
        f.write(line)

    out_arr = [model_type,
               hostname,
               date_stamp,
               str(results_file_info['dataset']),
               str(results_file_info['noise']),
               str(results_file_info['angle']),
               str(results_file_info['nomsigfrac']),
               str(weighted_sig_frac),
               str(max_n_epochs),
               #str(n_epochs),
               ':'.join(str(x) for x in hidden_nodes),
               str(train_results_df['eps'].iloc[-1]),
               str(n_train_events),
               str(n_valid_events),
               '%0.4f' % train_time,
               str(opt_thr),
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
               '%0.3f' % pass_signif,
               cm_to_str(cm_train),
               cm_to_str(cm_valid),
               cm_to_str(cm_weighted)]
    print(out_arr)
    line = ','.join(out_arr) + '\n'
    f.write(line)
    f.close()

if 'noplot' not in sys.argv:
    plt.show()
