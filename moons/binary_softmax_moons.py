#!/usr/bon/env python
'''
Code to generate 2-feature moons dataset with additional peaking/exponential feature.
'''

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

n_evts = 2*int(sys.argv[1])
noise = float(sys.argv[2])
angle = 0.0
n_epochs = 100
sig_frac = 0.5
if len(sys.argv) >= 4:
    angle = float(sys.argv[3])
if len(sys.argv) >= 5:
    n_epochs = int(sys.argv[4])
if len(sys.argv) >= 6:
    sig_frac = float(sys.argv[5])

output_fname = 'reg_nn_pwrs/train_evts=' + str(n_evts)
output_fname += ':noise=' + str(noise)
output_fname += ':angle=' + str(angle)
output_fname += ':epochs=' + str(n_epochs)
output_fname += ':sig_frac=' + str(sig_frac)

# parameters for 'mass' distribution
min, max = 0.0, 1.0
mean, width, n_sigmas = 0.5, 0.05, 2.5

# make the data and labels
raw_df = make_moons_mass(n_evts, min, max, mean=mean, sigma=width, noise=noise, angle=angle, beta=0.60)
df = raw_df.copy()

y = df['label']
# generate one-hot label encoding
# y_encoded, y_cats = y.factorize()
# oh_enc = preprocessing.onehotencoder(categories='auto')
# y_1hot = oh_enc.fit_transform(y_encoded.reshape(-1,1)).toarray()
y_1hot = pd.concat([df['label_0'], df['label_1']], axis=1, sort=False)

masses = df['m']
df.drop(['label', 'label_0', 'label_1'], axis=1, inplace=True)
X_train, X_test, y_1hot_train, y_1hot_test = train_test_split(df, y_1hot, test_size=0.25, random_state=42)
masses_train = X_train['m']
masses_test = X_test['m']
X_train.drop('m', axis=1, inplace=True)
X_test.drop('m', axis=1, inplace=True)

dropout_frac = 0.01
hidden_nodes = [20, 20, 10]

# initialize model...
binary_clf = binary_softmax_model(len(X_train.columns), hidden_nodes, input_dropout=dropout_frac)
print(binary_clf.summary())

epochs, eval_loss, eval_accs, train_loss, train_accs = [], [], [], [], []
test_loss, test_accs = [], []

# train model...
train_results_df = train_model(binary_clf, X_train, y_1hot_train, X_test, y_1hot_test,
                               n_epochs, batch_size=100, verbose=1, ot_shutoff=True)

print('\n... NN trained, plotting...\n')

# confusion matrix...
y_1hot_pred_train = binary_clf.predict(X_train)
y_pred_train = y_1hot_pred_train.argmax(axis=1)

y_1hot_pred = binary_clf.predict(X_test)
y_pred = y_1hot_pred.argmax(axis=1)

matrix_train = metrics.confusion_matrix(y_1hot_train.to_numpy().argmax(axis=1),
                                        y_1hot_pred_train.argmax(axis=1))
matrix_train = matrix_train.astype('float') / matrix_train.sum(axis=1)[:, np.newaxis]
matrix_test = metrics.confusion_matrix(y_1hot_test.to_numpy().argmax(axis=1),
                                       y_1hot_pred.argmax(axis=1))
matrix_test = matrix_test.astype('float') / matrix_test.sum(axis=1)[:, np.newaxis]

fig = plt.figure(figsize=(11,6))
nbins = int(np.sqrt(n_evts)/2)

n_rows, n_cols = 2, 3
# ax = plt.subplot(n_rows,n_cols,1)
# hist_xs(raw_df, 'x1', nbins, ax)
#
# ax = plt.subplot(n_rows,n_cols,2)
# hist_xs(raw_df, 'x2', nbins, ax)

ax = plt.subplot(n_rows,n_cols, 1)
plt.plot(train_results_df['eps'], train_results_df['eval_accs'], label='train, dropout=' + str(dropout_frac))
plt.plot(train_results_df['eps'], train_results_df['train_accs'], label='train')
plt.plot(train_results_df['eps'], train_results_df['test_accs'], label='test')
plt.plot(train_results_df['eps'], train_results_df['test_acc_sma5'], label='test, sma5')
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.legend(loc='lower right')
plt.ylabel('accuracy')
plt.xlabel('epoch')

ax = plt.subplot(n_rows, n_cols, 2)
plt.plot(train_results_df['eps'], train_results_df['eval_loss'], label='train, dropout=' + str(dropout_frac))
plt.plot(train_results_df['eps'], train_results_df['train_loss'], label='train')
plt.plot(train_results_df['eps'], train_results_df['test_loss'], label='test')
plt.plot(train_results_df['eps'], train_results_df['test_loss_sma5'], label='test sma5')
#plt.plot(history.history['val_acc'])
plt.title('loss (bin. cross-entropy)')
plt.legend(loc='upper right')
plt.ylabel('loss')
plt.xlabel('epoch')
ax = plt.subplot(n_rows,n_cols, n_rows * n_cols - 2)
plot_xs(raw_df, ax)
plt.title('noise = ' + str(noise) + ', angle = ' + str(angle))

ax = plt.subplot(n_rows,n_cols, n_rows * n_cols - 1)
hist_ms(raw_df, min, max, nbins, ax)
plt.tight_layout()

fig = plt.figure(figsize=(11,5))
ax = plt.subplot(1,2,1)
print('training results...')
class_labels = ['type 0', 'type 1']
plot_confusion_matrix(y_1hot_train.to_numpy().argmax(axis=1),
                      y_1hot_pred_train.argmax(axis=1), class_labels, ax,
                      normalize=True, title='confusion matrix, train')

print('\ntesting results...')
ax = plt.subplot(1,2,2)
plot_confusion_matrix(y_1hot_test.to_numpy().argmax(axis=1),
                      y_1hot_pred.argmax(axis=1), class_labels, ax,
                      normalize=True, title='confusion matrix, test')

# ax = plt.subplot(n_rows,n_cols, n_rows * n_cols)
# test_dict = {'x1':X_test['x1'], 'x2':X_test['x2'], 'm':masses_test, 'y':y_test, 'pred':y_pred_keras}
# test_df = pd.DataFrame(test_dict)
# min_df, max_df = np.amin(y_pred_keras), np.amax(y_pred_keras)
# nslices = 100
# dvals = np.linspace(min_df, max_df, num=nslices + 1)
# n_sig, n_bkgd, pwr = [], [], []
# for d in dvals:
#     n_sig.append(len(test_df['m'][test_df.y == 1][test_df.pred >= d]))
#     n_bkgd.append(len(test_df['m'][test_df.y == 0][test_df.pred >= d]))
#     pwr.append(n_sig[-1]*sig_frac / np.sqrt(n_sig[-1]*sig_frac + n_bkgd[-1]*(1-sig_frac)))
# opt_pwr = np.max(pwr)
# opt_idx = pwr.index(opt_pwr)
# opt_df  = dvals[opt_idx]
# ax.plot(dvals, pwr, label='sig fraction = ' + str(sig_frac))
# plt.xlabel('decision function value')
# plt.ylabel(r'$S / \sqrt{S+B}$')
# plt.axvline(x=opt_df, color='lightgray', dashes=(1,1))
# plt.legend(loc='lower right')
#
# if 'pickle' in sys.argv:
#     output_fname += ':opt_df=' + str(np.round(opt_df, 6))
#     with open(output_fname + ".pkl", 'wb') as f:
#         pickle.dump(dvals, f)
#         pickle.dump(pwr, f)
#         f.close()

plt.tight_layout()

# # # # # plot decision boundaries
fig = plt.figure(figsize=(11,7))
ax = plt.subplot(1,1,1)
x1_mesh, x2_mesh, class_mesh = predict_bound_class(binary_clf, df, 2)
ax.contourf(x1_mesh, x2_mesh, class_mesh, alpha=0.4)

plot_xs(raw_df, ax)
plt.title('noise = ' + str(noise) + ', angle = ' + str(angle) + ', epochs = ' + str(n_epochs))

plt.tight_layout()

# # # # # plot weighted-data histograms after optimal cut

weighted_df = make_moons_mass(n_evts, min, max, mean=mean, sigma=width,
                              noise=noise, angle=angle, beta=0.60, sig_fraction=sig_frac)
y_weighted = weighted_df['label']
# y_weighted_encoded, y_weighted_cats = y_weighted.factorize()
# y_weighted_1hot = oh_enc.transform(y_weighted_encoded.reshape(-1,1)).toarray()
y_weighted_1hot = pd.concat([weighted_df['label_0'], weighted_df['label_1']], axis=1, sort=False)

print('applying optimal cut to dataset with sig_frac = ' + str(sig_frac) + '...')
#X_weighted, _, y_weighted, _ = train_test_split(weighted_df, y_weighted, test_size=0.0, random_state=42)
xs_weighted = weighted_df.drop(['label', 'label_0', 'label_1', 'm'], axis=1)
weighted_df['prob_0'] = binary_clf.predict(xs_weighted)[:,0]
weighted_df['prob_1'] = binary_clf.predict(xs_weighted)[:,1]

# calculate analysis power...
raw_signif, pass_signif = compute_signif_binary(weighted_df, mean, width, n_sigmas)
print('\n\nraw analysis significance:\t', str(raw_signif))
print('pass analysis significance:\t', str(pass_signif))

fig = plt.figure(figsize=(11,4))
ax = plt.subplot(1,2,1)
hist_ms(weighted_df, min, max, nbins, ax)
plt.title('raw masses, sig\_frac = ' + str(sig_frac))
plt.legend(loc='upper right')

ax = plt.subplot(1,2,2)
hist_softmax_cut_ms(weighted_df, min, max, nbins, ax)
plt.title('masses pass nn, sig\_frac = ' + str(sig_frac))
plt.legend(loc='upper right')

plt.tight_layout()

if 'noplot' not in sys.argv:
    plt.show()
