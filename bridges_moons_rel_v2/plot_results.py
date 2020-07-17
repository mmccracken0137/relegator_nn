#!/usr/bin/env python
'''

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

df = pd.read_csv(sys.argv[1]) # 'datasets/train_ds.csv')
sfs = df.nominal_sig_frac.unique()

df['final_signif'] = df['n_pass_sig'] / np.sqrt(df['n_pass_sig'] +
                                                df['n_pass_bkgd'])

fig = plt.figure(figsize=(8,7))
for i, s in enumerate(sfs):
    ax = plt.subplot(3,2,i+1)
    sub_df = df[df['nominal_sig_frac'] == s]
    plt.scatter(sub_df[sub_df['model_type'] == 'regressor']['n_pass_bkgd'],
                sub_df[sub_df['model_type'] == 'regressor']['n_pass_sig'],
                marker='.', facecolors='none', edgecolors='b', label='regressor')
    plt.scatter(sub_df[sub_df['model_type'] == 'relegator']['n_pass_bkgd'],
                sub_df[sub_df['model_type'] == 'relegator']['n_pass_sig'],
                marker='.', facecolors='none', edgecolors='orange', label='relegator')

    # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    # ax.text(0.05, 0.95, 'raw sig frac = ' + str(s), transform=ax.transAxes, fontsize=8,
    #         verticalalignment='top', bbox=props)
    ax.legend(loc='lower right', title='raw sig frac = ' + str(s))

plt.tight_layout()

fig = plt.figure(figsize=(8,7))
for i, s in enumerate(sfs):
    ax = plt.subplot(3,2,i+1)
    sub_df = df[df['nominal_sig_frac'] == s]
    plt.scatter(sub_df[sub_df['model_type'] == 'regressor']['n_pass_sig'],
                sub_df[sub_df['model_type'] == 'regressor']['final_signif'],
                marker='.', facecolors='none', edgecolors='b', label='regressor')
    plt.scatter(sub_df[sub_df['model_type'] == 'relegator']['n_pass_sig'],
                sub_df[sub_df['model_type'] == 'relegator']['final_signif'],
                marker='.', facecolors='none', edgecolors='orange', label='relegator')

    # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    # ax.text(0.05, 0.95, 'raw sig frac = ' + str(s), transform=ax.transAxes, fontsize=8,
    #         verticalalignment='top', bbox=props)
    ax.legend(loc='lower right', title='raw sig frac = ' + str(s))

plt.tight_layout()

plt.show()

# train_bkgd = train_df[train_df['truth_class'] == 0]
# train_sig = train_df[train_df['truth_class'] == 1]
# # print(train_df.head(10))
#
# plt.scatter(train_sig['x1'], train_sig['x2'], s=1, marker='.')
# plt.scatter(train_bkgd['x1'], train_bkgd['x2'], s=1, marker='.')
# plt.show()

# truth class information
# target_class_name = 'truth_class'
# signal_idx = 1
# background_idx = [0]
# validation_fraction = 0.01
# train_on_fom = False
#
# # separate truth targets...
# y_cat_idxs = train_df[target_class_name]
# # make 1-hot targets...
# y_1hot = pd.get_dummies(train_df[target_class_name], prefix=target_class_name)
# # add extra truth_class column for relegator class TKTKTK move this into the module?
# y_1hot[target_class_name + '_relegate'] = 0
#
# # drop target/class info from dataframe...
# to_drop = [target_class_name]
# train_df.drop(to_drop, axis=1, inplace=True)
#
# # split training dataframe into train/validation
# X_train, X_valid, y_train, y_valid = train_test_split(train_df, y_1hot,
#                                                       test_size=validation_fraction)
#                                                       # random_state=validation_random_state)
#
# # make separate dfs for the feature of merit values
# # make separate dfs for the feature of merit values
# fom_name = 'mass'
# fom_train = X_train[fom_name]
# fom_valid = X_valid[fom_name]
# # if we don't train on the fom (i.e., it's not an input feature), drop it from the X dfs
# if not train_on_fom:
#     X_train.drop(fom_name, axis=1, inplace=True)
#     X_valid.drop(fom_name, axis=1, inplace=True)
#
# # convert the input and target dfs to tensorflow datasets
# input_feats_arr = list(X_train.columns)
#
# # train_tf_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(X_train[input_feats_arr].values, tf.float32),
# #                                                        tf.cast(y_train.values, tf.int32)))
# # valid_tf_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(X_valid[input_feats_arr].values, tf.float32),
# #                                                        tf.cast(y_valid.values, tf.int32)))
# #
# # if training_batch_size == 0:
# #     train_tf_dataset = train_tf_dataset.shuffle(len(X_train)).batch(len(X_train))
# #     valid_tf_dataset = valid_tf_dataset.shuffle(len(X_valid)).batch(len(X_valid))
# # else:
# #     train_tf_dataset = train_tf_dataset.shuffle(len(X_train)).batch(training_batch_size)
# #     valid_tf_dataset = valid_tf_dataset.shuffle(len(X_valid)).batch(training_batch_size)
#
# # numpy types for post-train plotting...
# X_train = X_train.to_numpy()
# y_train = y_train.to_numpy()
# X_valid = X_valid.to_numpy()
# y_valid = y_valid.to_numpy()
#
#
# # # # # # # # # # # # # # # # # # #
# # plot decision boundaries
# plot_xs(X_train, y_train[:,1], ax)
# # plt.title('noise = ' + str(noise) + ', angle = ' + str(angle) + ', epochs = ' + str(n_epochs))
# plt.tight_layout()
#
# plt.show()
