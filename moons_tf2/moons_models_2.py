#!/usr/bon/env python
'''
'''

from colorama import Fore, Back, Style
import pandas as pd
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import sys
import pickle

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.utils.multiclass import unique_labels
import tensorflow as tf  ## this code runs with tf2.0-cpu!!!
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import tensorflow.keras.backend as K
from moons_tools_2 import *

# def regressor_model(n_inputs, hidden_nodes, input_dropout=0.0, biases=True, learning_rate=0.001):
#     n_hidden = len(hidden_nodes)
#     model = Sequential()
#     if input_dropout > 0.0:
#         model.add(Dropout(input_dropout, input_shape=(n_inputs, )))
#         model.add(Dense(hidden_nodes[0], activation='relu', use_bias=biases))
#     else:
#         model.add(Dense(hidden_nodes[0], input_dim=n_inputs,
#                         activation='relu', use_bias=biases))
#
#     for i in range(n_hidden - 1):
#         model.add(Dense(hidden_nodes[i+1], activation='relu', use_bias=biases))
#
#     model.add(Dense(1, activation='sigmoid'))
#     # Compile model
#     opt = Adam(lr=learning_rate)
#     model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#     return model


# def regressor_model2(n_inputs, hidden_nodes, input_dropout=0.0, biases=True, learning_rate=0.001):
#     n_hidden = len(hidden_nodes)
#     layers = []
#     if input_dropout > 0.0:
#         layers.append(tf.keras.layers.Dropout(input_dropout, input_shape=(n_inputs, )))
#         layers.append(tf.keras.layers.Dense(hidden_nodes[0], activation='relu', use_bias=biases))
#     else:
#         layers.append(tf.keras.layers.Dense(hidden_nodes[0], input_dim=n_inputs,
#                                          activation='relu', use_bias=biases))
#
#     for i in range(n_hidden - 1):
#         layers.append(tf.keras.layers.Dense(hidden_nodes[i+1], activation='relu', use_bias=biases))
#
#     layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))
#     model = tf.keras.Sequential(layers)
#     # Compile model
#     # opt = Adam(lr=learning_rate)
#     # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#     return model

# def train_step2(model, X, y_truth):
#     loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#     with tf.GradientTape() as tape:
#         y_pred = model(X, training=True).numpy().ravel()
#
#         # Add asserts to check the shape of the output.
#         # tf.debugging.assert_equal(y_pred.shape, (1024, ))
#
#         loss_value = loss_object(y_truth, y_pred)
#         acc_value = tf.keras.metrics.categorical_accuracy(y_truth, y_pred)
#
#     grads = tape.gradient(loss_value, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     return loss_value, acc_value
#
# def eval_step2(model, X, y_truth):
#     y_pred = model(X, training=True)
#     loss_value = loss_object(y_truth, y_pred)
#     acc_value = acc_object(y_truth, y_pred)
#     return loss_value, acc_value

def train_model2(clf, opt, X_train, y_train, X_test, y_test, n_epochs,
                 batch_size=1024, verbose=1, ot_shutoff=False,
                 ot_shutoff_depth=5):
    epochs, eval_loss, eval_accs, train_loss, train_accs = [], [], [], [], []
    test_loss, test_accs = [], []
    test_acc_sma = []
    test_loss_sma = []

    n_evts = len(X_train)
    train_ds = np_to_tfds(X_train, y_train, batch_size=batch_size)
    test_ds = np_to_tfds(X_test, y_test, batch_size=batch_size)

    for i in range(n_epochs):
        print('\nEPOCH ' + str(i+1) + '/' + str(n_epochs) , ', learning rate: ' + str(opt.lr))
        lv, av = 0, 0
        for (batch, (xs, ys)) in enumerate(train_ds):
            lv, av = train_step2(clf, xs, ys)
        print ('Epoch {} finished'.format(epoch))

        epochs.append(i)
        eval_accs.append(av)
        eval_loss.append(lv)
        lv, av = eval_step2(clf, X_train, y_train)
        train_accs.append(av)
        train_loss.append(lv)
        lv, av = eval_step2(clf, X_test, y_test)
        test_accs.append(av)
        test_loss.append(lv)

        if len(test_accs) < ot_shutoff_depth:
            test_acc_sma.append(np.sum(test_accs[0:len(test_accs)])/len(test_accs))
            test_loss_sma.append(np.sum(test_loss[0:len(test_loss)])/len(test_loss))
        else:
            test_acc_sma.append(np.round(np.sum(test_accs[-ot_shutoff_depth:])/float(ot_shutoff_depth), 4))
            test_loss_sma.append(np.round(np.sum(test_loss[-ot_shutoff_depth:])/float(ot_shutoff_depth), 4))

        if verbose > 0:
            print('training --> loss = %0.4f, \t acc = %0.4f'%(loss, acc))
            print('testing --> loss = %0.4f, \t acc = %0.4f'%(loss, acc))

        loss_slope, loss_sma_slope = 0, 0
        if i > ot_shutoff_depth:
            epos = np.linspace(1, ot_shutoff_depth, ot_shutoff_depth)
            loss_sma_slope, _, _, _, _ = stats.linregress(epos, test_loss_sma[-ot_shutoff_depth:])
            loss_slope, _, _, _, _ = stats.linregress(epos, test_loss[-ot_shutoff_depth:])

        # if ot_shutoff and loss_sma_slope > 0:
        if ot_shutoff and i > 10 and loss_slope >= 0:
            break

    dict = {'eps': epochs,
            'eval_accs': eval_accs, 'eval_loss': eval_loss,
            'train_loss': train_loss, 'train_accs': train_accs,
            'test_loss': test_loss, 'test_accs': test_accs,
            'test_acc_sma': test_acc_sma,
            'test_loss_sma': test_loss_sma}

    train_results_df = pd.DataFrame(dict)
    print('\nmodel trained for ' + str(len(epochs)) + ' epochs')
    print('final train accuracy:\t' + str(train_accs[-1]))
    print('final test accuracy:\t' + str(test_accs[-1]))
    return train_results_df

def regress_signif_loss(sig_frac, reg_min=0, reg_max=1):
    def loss(y_truth, y_pred):
        # n_S = K.sum(y_truth * y_pred) # total signal prob of truth signal events
        n_S = K.sum(y_truth) # total signal prob of truth signal events
        # n_tot = tf.size(y_truth)
        # n_B = K.sum(y_pred[:,sig_idx] # total prob of truth bkgd events

        # TKTKTK Not sure that we really care about probabilities here, just max probability???
        #signif = signif_function(sig_frac * n_S, n_B)
        n_tot = K.sum(y_truth)
        print("NTOT", n_tot)

        sum = 0
        sum += K.binary_crossentropy(y_truth, y_pred) # cce term for accuracy
        # sum += 1 / signif #/ n_tot # term for significance
        # sum += n_S + n_tot
        # sum += n_S
        return sum
    return loss

def signif_regressor_model(n_inputs, hidden_nodes, loss_fcn,
                           input_dropout=0.0, biases=True, learning_rate=0.001):
    n_hidden = len(hidden_nodes)
    model = Sequential()
    if input_dropout > 0.0:
        model.add(Dropout(input_dropout, input_shape=(n_inputs, )))
        model.add(Dense(hidden_nodes[0], activation='relu', use_bias=biases))
    else:
        model.add(Dense(hidden_nodes[0], input_dim=n_inputs,
                        activation='relu', use_bias=biases))

    for i in range(n_hidden - 1):
        model.add(Dense(hidden_nodes[i+1], activation='relu', use_bias=biases))

    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    opt = Adam(lr=learning_rate)
    model.compile(loss=loss_fcn, optimizer=opt, metrics=['accuracy'])
    return model

# model definitions
def binary_softmax_model(n_inputs, hidden_nodes, input_dropout=0.0, biases=True, learning_rate=0.001):
    n_hidden = len(hidden_nodes)
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
    opt = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def train_model(clf, X_train, y_train, X_test, y_test, n_epochs,
                batch_size=1024, verbose=1, ot_shutoff=False,
                ot_shutoff_depth=5):
    epochs, eval_loss, eval_accs, train_loss, train_accs = [], [], [], [], []
    test_loss, test_accs = [], []
    test_acc_sma = []
    test_loss_sma = []

    n_evts = len(X_train)
    train_ds = np_to_tfds(X_train, y_train, batch_size=batch_size)
    test_ds = np_to_tfds(X_test, y_test, batch_size=batch_size)

    for i in range(n_epochs):
        print('\nEPOCH ' + str(i+1) + '/' + str(n_epochs), ', learning rate: ' + str(K.eval(clf.optimizer.lr)))
        history = clf.fit(train_ds, epochs=1, verbose=1)

        epochs.append(i)
        eval_accs.append(history.history['accuracy'][-1])
        eval_loss.append(history.history['loss'][-1])
        loss, acc = clf.evaluate(X_train, y_train, verbose=2)
        train_accs.append(acc)
        train_loss.append(loss)
        loss, acc = clf.evaluate(X_test, y_test, verbose=2)
        test_accs.append(acc)
        test_loss.append(loss)

        if len(test_accs) < ot_shutoff_depth:
            test_acc_sma.append(np.sum(test_accs[0:len(test_accs)])/len(test_accs))
            test_loss_sma.append(np.sum(test_loss[0:len(test_loss)])/len(test_loss))
        else:
            test_acc_sma.append(np.round(np.sum(test_accs[-ot_shutoff_depth:])/float(ot_shutoff_depth), 4))
            test_loss_sma.append(np.round(np.sum(test_loss[-ot_shutoff_depth:])/float(ot_shutoff_depth), 4))

        if verbose > 0:
            print('training --> loss = %0.4f, \t acc = %0.4f'%(loss, acc))
            print('testing --> loss = %0.4f, \t acc = %0.4f'%(loss, acc))

        loss_slope, loss_sma_slope = 0, 0
        if i > ot_shutoff_depth:
            epos = np.linspace(1, ot_shutoff_depth, ot_shutoff_depth)
            loss_sma_slope, _, _, _, _ = stats.linregress(epos, test_loss_sma[-ot_shutoff_depth:])
            loss_slope, _, _, _, _ = stats.linregress(epos, test_loss[-ot_shutoff_depth:])

        # if ot_shutoff and loss_sma_slope > 0:
        if ot_shutoff and i > 10 and loss_slope >= 0:
            break

    dict = {'eps': epochs,
            'eval_accs': eval_accs, 'eval_loss': eval_loss,
            'train_loss': train_loss, 'train_accs': train_accs,
            'test_loss': test_loss, 'test_accs': test_accs,
            'test_acc_sma': test_acc_sma,
            'test_loss_sma': test_loss_sma}

    train_results_df = pd.DataFrame(dict)
    print('\nmodel trained for ' + str(len(epochs)) + ' epochs')
    print('final train accuracy:\t' + str(train_accs[-1]))
    print('final test accuracy:\t' + str(test_accs[-1]))
    return train_results_df

def cce_loss():
    def loss(y_true, y_pred):
        sum = K.categorical_crossentropy(y_true, y_pred)
        return sum
    return loss

def relegator_loss(sig_frac, sig_idx=1, reg_min=0, reg_max=1):
    def loss(y_truth, y_pred):
        sig_mask = y_truth[:,sig_idx]
        bkgd_mask = y_truth[:,0]
        diff_01 = K.abs(y_pred[:,0] - y_pred[:,sig_idx])
        n_S = K.sum(y_pred[:,sig_idx] * sig_mask) # total prob of truth signal events
        n_B = K.sum(y_pred[:,sig_idx] * bkgd_mask) # total prob of truth bkgd events

        # TKTKTK Not sure that we really care about probabilities here, just max probability???
        signif = signif_function(sig_frac * n_S, n_B)
        n_tot = K.sum(y_truth)
        sum = 0
        # sum += K.categorical_crossentropy(y_truth, y_pred) # cce term for accuracy
        # sum -= sig_frac * signif #/ n_tot # term for significance

        sum += K.categorical_crossentropy(y_truth, y_pred) # cce term for accuracy
        sum += 1 / signif #/ n_tot # term for significance
        # sum -= K.log(signif)
        return sum
    return loss

def relegator_model(n_inputs, hidden_nodes, loss_fcn, input_dropout=0.0, biases=True, learning_rate=0.001):
    n_hidden = len(hidden_nodes)
    model = Sequential()
    if input_dropout > 0.0:
        model.add(Dropout(input_dropout, input_shape=(n_inputs, )))
        model.add(Dense(hidden_nodes[0], activation='relu', use_bias=biases))
    else:
        model.add(Dense(hidden_nodes[0], input_dim=n_inputs,
                        activation='relu', use_bias=biases))

    for i in range(n_hidden - 1):
        model.add(Dense(hidden_nodes[i+1], activation='relu', use_bias=biases))

    out_layer = model.add(Dense(3, activation='softmax'))
    # Compile model
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    opt = Adam(lr=learning_rate)
    model.compile(loss=loss_fcn, optimizer=opt, metrics=['accuracy'])
    return model

# def pred_1hot_to_class(X_in, model, n_classes):
#     # converts 1hot model output to class label
#     pred_1hot = model.predict(X_in)
#     pred_class = [np.where(p == np.max(p)) for p in pred_1hot]
#     pred_class = np.reshape(pred_class, (np.shape(X_in)[0], 1))
#     return pred_class
#
# def predict_bound_class(model, df, n_outputs, opt_thr=0):
#     x1_min, x1_max = df['x1'].min() - 0.25, df['x1'].max() + 0.25
#     x2_min, x2_max = df['x2'].min() - 0.25, df['x2'].max() + 0.25
#     x1_range = x1_max - x1_min
#     x2_range = x2_max - x2_min
#     x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, x1_range/100),
#                                    np.arange(x2_min, x2_max, x2_range/100))
#
#     mesh_xs = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]
#     class_mesh = []
#     if n_outputs == 1:
#         class_mesh = model.predict(mesh_xs)
#         if opt_thr > 0.0:
#             class_mesh[class_mesh > opt_thr]  = 1
#             class_mesh[class_mesh <= opt_thr] = 0
#     else:
#         class_mesh = pred_1hot_to_class(mesh_xs, model, n_outputs)
#     class_mesh = class_mesh.reshape(x1_mesh.shape)
#     return x1_mesh, x2_mesh, class_mesh

def df_to_tfdataset(df, feats_arr, labels_arr):
    tf_ds = (tf.data.Dataset.from_tensor_slices((tf.cast(df[feats_arr].values, tf.float32),
                                                 tf.cast(df[labels_arr].values, tf.int32))))
    return tf_ds

# def np_to_tfds(X, y, batch_size=1024):
#     feats_arr = ['x1', 'x2']
#     n_evts = len(X)
#     tf_ds = tf.data.Dataset.from_tensor_slices((tf.cast(X[feats_arr].values, tf.float32),
#                                                 tf.cast(y.values, tf.int32)))
#     tf_ds = tf_ds.shuffle(n_evts).batch(batch_size)
#     return tf_ds

def get_ns_truth(y_1hot, sig_idx=1):
    n_sig, n_bkgd = 0, 0
    for snip in y_1hot:
        if snip[sig_idx] == 1:
            n_sig += 1
        else:
            n_bkgd += 1
    return n_sig, n_bkgd

def get_ns_pred(y_1hot, sig_idx=1):
    n_sig, n_bkgd = 0, 0
    for snip in y_1hot:
        if np.argmax(snip) == sig_idx:
            n_sig += 1
        else:
            n_bkgd += 1
    return n_sig, n_bkgd
