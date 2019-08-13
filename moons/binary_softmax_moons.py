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
mean, width = 0.5, 0.12

# make the data and labels
raw_df = make_moons_mass(n_evts, min, max, mean=mean, sigma=width, noise=noise, angle=angle, beta=0.60)
df = raw_df.copy()

y = df['label']
# generate one-hot label encoding
y_encoded, y_cats = y.factorize()
oh_enc = preprocessing.OneHotEncoder(categories='auto')
y_1hot = oh_enc.fit_transform(y_encoded.reshape(-1,1)).toarray()

masses = df['m']
df.drop('label', axis=1, inplace=True)
X_train, X_test, y_1hot_train, y_1hot_test = train_test_split(df, y_1hot, test_size=0.25, random_state=42)
masses_train = X_train['m']
masses_test = X_test['m']
X_train.drop('m', axis=1, inplace=True)
X_test.drop('m', axis=1, inplace=True)

# model definitions
def binary_softmax_model(n_inputs, n_hidden, hidden_nodes, input_dropout=0.0, biases=True):
    model = Sequential()
    if input_dropout > 0.0:
        model.add(Dropout(input_dropout, input_shape=(n_inputs, )))
        model.add(Dense(hidden_nodes[0], activation='relu', use_bias=biases))
    else:
        model.add(Dense(hidden_nodes[0], input_dim=n_inputs,
                        activation='relu', use_bias=biases))

    for i in range(n_hidden - 1):
        model.add(Dense(hidden_nodes[i+1], activation='relu', use_bias=biases))

    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def pred_to_class(X_in, model, n_classes):
    pred_1hot = model.predict(X_in)
    pred_class = [np.where(p == np.max(p)) for p in pred_1hot]
    pred_class = np.reshape(pred_class, (np.shape(X_in)[0], 1))
    return pred_class

dropout_frac = 0.2
binary_clf = binary_softmax_model(len(X_train.columns), 2, [20, 20, 10], input_dropout=dropout_frac)
print(binary_clf.summary())

epochs, eval_loss, eval_accs, train_loss, train_accs = [], [], [], [], []
test_loss, test_accs = [], []

for i in range(n_epochs):
    print('\nEPOCH ' + str(i) + '/' + str(n_epochs))
    #if i > 0:
    history = binary_clf.fit(X_train, y_1hot_train, epochs=1, batch_size=100, verbose=1)

    epochs.append(i)
    eval_accs.append(history.history['accuracy'][-1])
    eval_loss.append(history.history['loss'][-1])
    loss, acc = binary_clf.evaluate(X_train, y_1hot_train, verbose=2)
    train_accs.append(acc)
    train_loss.append(loss)
    print('training --> loss = %0.4f, \t acc = %0.4f'%(loss, acc))
    loss, acc = binary_clf.evaluate(X_test, y_1hot_test, verbose=2)
    test_accs.append(acc)
    test_loss.append(loss)
    print('testing --> loss = %0.4f, \t acc = %0.4f'%(loss, acc))

    # sig_ot_comp.append(dec_score_comp(keras_model, X_train_scale[y_train>0.5], X_test_scale[y_test>0.5]))
    # bkgd_ot_comp.append(dec_score_comp(keras_model, X_train_scale[y_train<0.5], X_test_scale[y_test<0.5]))

print('\n... NN trained, plotting...\n')

# confusion matrix...
y_1hot_pred_train = binary_clf.predict(X_train)
y_pred_train = y_1hot_pred_train.argmax(axis=1)

y_1hot_pred = binary_clf.predict(X_test)
y_pred = y_1hot_pred.argmax(axis=1)

matrix_train = metrics.confusion_matrix(y_1hot_train.argmax(axis=1), y_1hot_pred_train.argmax(axis=1))
matrix_train = matrix_train.astype('float') / matrix_train.sum(axis=1)[:, np.newaxis]
matrix_test = metrics.confusion_matrix(y_1hot_test.argmax(axis=1), y_1hot_pred.argmax(axis=1))
matrix_test = matrix_test.astype('float') / matrix_test.sum(axis=1)[:, np.newaxis]

fig = plt.figure(figsize=(11,7))
nbins = int(np.sqrt(n_evts))

n_rows, n_cols = 3, 3
ax = plt.subplot(n_rows,n_cols,1)
hist_xs(raw_df, 'x1', nbins, ax)

ax = plt.subplot(n_rows,n_cols,2)
hist_xs(raw_df, 'x2', nbins, ax)

ax = plt.subplot(n_rows, n_cols,3)
print('training results...')
class_labels = ['type 0', 'type 1']
plot_confusion_matrix(y_1hot_train.argmax(axis=1), y_1hot_pred_train.argmax(axis=1), class_labels, ax,
                          normalize=True,
                          title='confusion matrix, train')

print('\ntesting results...')
ax = plt.subplot(2,2,4)
plot_confusion_matrix(y_1hot_test.argmax(axis=1), y_1hot_pred.argmax(axis=1), class_labels, ax,
                          normalize=True,
                          title='confusion matrix, test')



ax = plt.subplot(n_rows,n_cols, 4)
plt.plot(epochs, eval_accs, label='train, dropout=' + str(dropout_frac))
plt.plot(epochs, train_accs, label='train')
plt.plot(epochs, test_accs, label='test')
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.legend(loc='lower right')
plt.ylabel('accuracy')
plt.xlabel('epoch')

ax = plt.subplot(n_rows, n_cols, 5)
plt.plot(epochs, eval_loss, label='train, dropout=' + str(dropout_frac))
plt.plot(epochs, train_loss, label='train')
plt.plot(epochs, test_loss, label='test')
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
# plot decision boundaries
x1_min, x1_max = df['x1'].min() - 0.25, df['x1'].max() + 0.25
x2_min, x2_max = df['x2'].min() - 0.25, df['x2'].max() + 0.25
x1_range = x1_max - x1_min
x2_range = x2_max - x2_min
x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, x1_range/100),
                               np.arange(x2_min, x2_max, x2_range/100))

mesh_xs = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]
#print(pred_to_class(mesh_xs, binary_clf, 2))
#print(np.shape(mesh_xs), np.shape(pred_to_class(mesh_xs, binary_clf, 2)))
bounds = pred_to_class(mesh_xs, binary_clf, 2)
bounds = bounds.reshape(x1_mesh.shape)
ax.contourf(x1_mesh, x2_mesh, bounds, alpha=0.4)

plot_xs(raw_df, ax)
plt.title('noise = ' + str(noise) + ', angle = ' + str(angle) + ', epochs = ' + str(n_epochs))

plt.tight_layout()

# # # # # plot weighted-data histograms after optimal cut

# weighted_df = make_moons_mass(n_evts, min, max, mean=mean, sigma=width, noise=noise, angle=angle, beta=0.60, sig_fraction=sig_frac)
# y_weighted = weighted_df['label']
# print('\noptimal decision function cut is at ' + str(opt_df))
# print('applying optimal cut to dataset with sig_frac = ' + str(sig_frac) + '...')
# #X_weighted, _, y_weighted, _ = train_test_split(weighted_df, y_weighted, test_size=0.0, random_state=42)
# xs_weighted = weighted_df.drop(['label', 'm'], axis=1)
# weighted_df['pred'] = binary_clf.predict(xs_weighted).ravel()
#
# fig = plt.figure(figsize=(11,7))
# ax = plt.subplot(1,1,1)
# hist_ms(weighted_df, min, max, nbins, ax)
# hist_cut_ms(weighted_df, opt_df, min, max, nbins, ax)
# plt.title('masses, sig\_frac = ' + str(sig_frac))
# plt.legend(loc='upper right')
#
# plt.tight_layout()

if 'noplot' not in sys.argv:
    plt.show()
