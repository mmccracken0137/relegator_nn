'''
Class def for classifiers with modified loss functions
'''

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy import stats
import sys

class ModClf:
    def __init__(self):
        self.optimizer = None
        self.layers = None
        self.model = None

        self.learning_rate = None
        self.hidden_layers_nodes = None
        self.bias = None
        self.n_inputs = 0
        self.n_outputs = 0

        self.loss_object = None
        self.acc_object = None

        self.name = None
        self.loss_type = None

    def build_model(self, nodes=[20,20,10], bias=True, n_ins=2, n_outs=1,
                    input_dropout=0.05):
        self.hidden_layers_nodes = nodes
        self.n_hidden_layers = len(self.hidden_layers_nodes)
        self.bias = bias
        self.n_inputs = n_ins
        self.n_outputs = n_outs
        self.input_dropout_frac = input_dropout
        self.layers = []

        if input_dropout > 0.0:
            self.layers.append(tf.keras.layers.Dropout(self.input_dropout_frac,
                                                       input_shape=(self.n_inputs, )))
            self.layers.append(tf.keras.layers.Dense(self.hidden_layers_nodes[0],
                                                     activation='relu',use_bias=self.bias))
        else:
            self.layers.append(tf.keras.layers.Dense(self.hidden_layers_nodes[0],
                                                     input_dim=self.n_inputs,
                                                     activation='relu', use_bias=self.bias))

        for i in range(self.n_hidden_layers - 1):
            self.layers.append(tf.keras.layers.Dense(self.hidden_layers_nodes[i+1],
                                                     activation='relu', use_bias=self.bias))

        self.layers.append(tf.keras.layers.Dense(self.n_outputs,
                                                 activation=self.output_activation))
        self.model = tf.keras.Sequential(self.layers)

    def init_optimizer(self, lr=1e-3):
        self.learning_rate = lr
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)

    def train_step(self, xs, y_truth):
        with tf.GradientTape() as tape:
            y_pred = self.model(xs, training=True)
            loss_val = self.loss_object(y_truth, y_pred)
            acc_val = self.acc_object(y_truth, y_pred)

        grads = tape.gradient(loss_val, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_val.numpy().mean(), acc_val.numpy().mean()

    def predict_step(self, x, y_t):
        y_p = self.model(x, training=False)
        loss_val = self.loss_object(y_t, y_p).numpy().mean()
        acc_val = self.acc_object(y_t, y_p).numpy().mean()
        return loss_val, acc_val

    def train(self, train_ds, test_ds, max_epochs, ot_cutoff=True, ot_cutoff_depth=10,
              verbose=True):
        self.ot_cutoff = ot_cutoff
        self.ot_cutoff_depth = ot_cutoff_depth
        self.max_epochs = max_epochs
        epochs, eval_loss, eval_accs, train_loss, train_accs = [], [], [], [], []
        test_loss, test_accs = [], []
        test_acc_sma = []
        test_loss_sma = []

        for epoch in range(self.max_epochs):
            out_text = ''
            epochs.append(epoch)
            lv, av = 0, 0
            for (batch, (xs, ys)) in enumerate(train_ds):
                lv, av = self.train_step(xs, ys)
            # print('Epoch {}/{} finished, learning rate: {:0.4f}'.format(epoch+1, self.max_epochs, self.optimizer.lr.numpy()))
            out_text += 'Epoch {}/{} finished, learning rate: {:0.4f}'.format(epoch+1, self.max_epochs, self.optimizer.lr.numpy()) + '\n'
            # print('train loss: \t{:0.4f} \t|\ttrain acc: \t{:0.4f}'.format(lv, av))
            out_text += 'train loss: \t{:0.4f} \t|\ttrain acc: \t{:0.4f}'.format(lv, av) + '\n'
            train_loss.append(lv)
            train_accs.append(av)
            for (batch, (xs, ys)) in enumerate(train_ds):
                lv, av = self.predict_step(xs, ys)
            # print('eval loss: \t{:0.4f} \t|\teval acc: \t{:0.4f}'.format(lv, av))
            out_text += 'eval loss: \t{:0.4f} \t|\teval acc: \t{:0.4f}'.format(lv, av) + '\n'
            eval_loss.append(lv)
            eval_accs.append(av)
            for (batch, (xs, ys)) in enumerate(test_ds):
                lv, av = self.predict_step(xs, ys)
            # print('test loss: \t{:0.4f} \t|\ttest acc: \t{:0.4f}'.format(lv, av))
            out_text += 'test loss: \t{:0.4f} \t|\ttest acc: \t{:0.4f}'.format(lv, av) + '\n'
            test_loss.append(lv)
            test_accs.append(av)
            # print()
            if verbose:
                print(out_text)

            loss_slope = 0
            epos = []
            if epoch + 1 > ot_cutoff_depth:
                epos = np.linspace(1, ot_cutoff_depth, ot_cutoff_depth)
                loss_slope, _, _, _, _ = stats.linregress(epos, test_loss[-ot_cutoff_depth:])

            if self.ot_cutoff and epoch + 1 > self.ot_cutoff_depth and loss_slope >= 0:
                break

        dict = {'eps': epochs,
                'eval_accs': eval_accs, 'eval_loss': eval_loss,
                'train_loss': train_loss, 'train_accs': train_accs,
                'test_loss': test_loss, 'test_accs': test_accs}

        train_results_df = pd.DataFrame(dict)
        print('\nmodel trained for ' + str(len(epochs)) + ' epochs')
        print('final train accuracy:\t' + str(train_accs[-1]))
        print('final test accuracy:\t' + str(test_accs[-1]))
        self.train_results = train_results_df

    # # # # # # # # # # # # # # # # #
    def set_signal_idx(self, idx):
        self.signal_idx = idx

    def set_background_idxs(self, idxs):
        self.background_idxs = idxs

    # # # # # # # # # # # # # # # # #
    # functions for significance calculation
    def set_signal_fraction(self, sig_frac):
        self.signal_fraction = sig_frac

    def set_validation_fraction(self, validation_frac):
        self.validation_fraction = validation_frac

    def set_n_train(self, n):
        self.train_n_events = n

    def set_n_valid(self, n):
        self.train_n_valid = n

    def set_n_weighted(self, n):
        self.train_n_weighted = n


    # def set_train_masses(self, masses, mass_mean, mass_width):
    #     self.train_masses = masses
    #     self.mass_mean = mass_mean
    #     self.mass_width = mass_width
    #     self.mass_peak_range = (self.mass_mean - self.mass_width,
    #                             self.mass_mean + self.mass_width)

    def set_train_fom(self, foms):
        self.train_masses = foms

    def set_test_fom(self, foms):
        self.test_masses = foms

    def set_fom_minmax(self, min=-1e7, max=1e7):
        self.fom_min = min
        self.fom_max = max
        self.fom_mean = (self.fom_max - self.fom_min) / 2.0
        self.fom_width = self.fom_max - self.fom_mean
        self.fom_peak_range = (self.fom_min, self.fom_max)

    def gen_train_peak_mask(self):
        self.train_peak_mask = self.make_peak_mask(self.train_masses)

    def gen_valid_peak_mask(self):
        self.validation_peak_mask = self.make_peak_mask(self.test_masses)

    def make_peak_mask(self, masses):
        # gets the indices of events in the mass peak
        # and then makes a mask array of 0s and 1s for dotting with masses array
        peak_idxs = np.where(np.abs(masses-self.fom_mean) <= self.fom_width)
        peak_mask = np.zeros_like(masses)
        peak_mask[peak_idxs] = 1
        return peak_mask

    def signif_proba(self, y_truth, y_pred, masses=[], peak_mask=[], data_frac=1.0):
        sig_mask  = tf.cast(tf.slice(y_truth, [0, self.signal_idx,], [len(y_truth), 1]), tf.float32)
        bkgd_mask = tf.cast(tf.slice(y_truth, [0, self.background_idxs[0],], [len(y_truth), 1]), tf.float32)
        sig_probs = tf.slice(y_pred, [0, self.signal_idx,], [len(y_truth), 1])
        peak_mask = tf.cast(peak_mask, tf.float32)

        sig_as_sig_probs = tf.reshape(tf.math.multiply(sig_probs, sig_mask), (len(y_truth),))
        bkgd_as_sig_probs = tf.reshape(tf.math.multiply(sig_probs, bkgd_mask), (len(y_truth),))

        data_frac = tf.constant(data_frac)
        n_S = (1/data_frac) * tf.math.reduce_sum(tf.math.multiply(sig_as_sig_probs, peak_mask), axis=0)
        n_B = (1/data_frac) * tf.math.reduce_sum(tf.math.multiply(bkgd_as_sig_probs, peak_mask), axis=0)

        signif = self.signif_function(n_S, n_B, tf.constant(self.signal_fraction))
        return signif

    def signif_categ(self, y_truth, y_pred, masses=[], peak_mask=[], data_frac=1.0):
        sig_mask  = tf.cast(tf.slice(y_truth, [0, self.signal_idx,], [len(y_truth), 1]), tf.float32)
        bkgd_mask = tf.cast(tf.slice(y_truth, [0, self.background_idxs[0],], [len(y_truth), 1]), tf.float32)

        categories = tf.reshape(tf.cast(tf.math.argmax(y_pred, axis=1), tf.float32), (len(y_truth), 1))

        sig_pred_cats = tf.reshape(tf.math.multiply(categories, sig_mask), (len(y_truth),))
        bkgd_pred_cats = tf.reshape(tf.math.multiply(categories, bkgd_mask), (len(y_truth),))

        data_frac = tf.constant(data_frac)
        n_S = (1/data_frac) * tf.math.reduce_sum(tf.math.multiply(sig_pred_cats, peak_mask), axis=0)
        n_B = (1/data_frac) * tf.math.reduce_sum(tf.math.multiply(bkgd_pred_cats, peak_mask), axis=0)

        signif = self.signif_function(n_S, n_B, tf.constant(self.signal_fraction))
        return signif

    # def signif_function(self, n_S, n_B, sig_frac):
    #     signif = tf.math.divide(n_S * sig_frac, tf.math.sqrt(n_S * sig_frac + n_B * (1 - sig_frac)))
    #     return signif

    def signif_function(self, n_S, n_B, sig_frac,
                        n_train = 1, n_eval = 1, n_weighted = 1):
        corr_fac = tf.math.divide(n_weighted, n_train * n_eval)
        signif = tf.math.divide(n_S * sig_frac * corr_fac,
                                tf.math.sqrt(n_S * sig_frac * corr_fac +
                                             n_B * (1 - sig_frac) * corr_fac))
        return signif

class RegressorClf(ModClf):
    def __init__(self):
        self.optimizer = None
        self.layers = None
        self.model = None

        self.learning_rate = None
        self.hidden_layers_nodes = None
        self.bias = None
        self.n_inputs = None

        self.output_activation = 'sigmoid'
        self.name = 'regression classifier'
        self.loss_type = 'binary CE'

        self.loss_object = tf.keras.losses.BinaryCrossentropy()
        self.acc_object = tf.keras.metrics.BinaryAccuracy()

class BinarySoftmaxClf(ModClf):
    def __init__(self):
        self.optimizer = None
        self.layers = None
        self.model = None

        self.learning_rate = None
        self.hidden_layers_nodes = None
        self.bias = None
        self.n_inputs = None

        self.output_activation = 'softmax'
        self.name = 'binary softmax classifier'
        self.loss_type = 'categorical CE'

        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.acc_object = tf.keras.metrics.CategoricalAccuracy()

class ModBinarySoftmaxClf(ModClf):
    def __init__(self):
        self.optimizer = None
        self.layers = None
        self.model = None

        self.learning_rate = None
        self.hidden_layers_nodes = None
        self.bias = None
        self.n_inputs = None

        self.signif_type = 'proba'
        self.output_activation = 'softmax'
        self.name = 'binary softmax classifier'
        self.loss_type = 'CCE + 1/sigma'

        self.acc_object = tf.keras.metrics.CategoricalAccuracy()

    def set_signif_type(self, type_str):
        if type_str not in ['proba', 'none']: # , 'categ']:
            print('\nError: significance type \'' + type_str + '\' not defined')
            sys.exit()
        else:
            self.signif_type = type_str

    def loss_object(self, y_truth, y_pred, train=True):
        signif = 0.0
        if train:
            if self.signif_type == 'proba':
                signif = self.signif_proba(y_truth, y_pred, self.train_masses,
                                           self.train_peak_mask, data_frac=1-self.validation_fraction)
            elif self.signif_type == 'categ':
                signif = self.signif_categ(y_truth, y_pred, self.train_masses,
                                           self.train_peak_mask, data_frac=1-self.validation_fraction)

        else:
            if self.signif_type == 'proba':
                signif = self.signif_proba(y_truth, y_pred, self.test_masses,
                                           self.validation_peak_mask, data_frac=self.validation_fraction)
            elif self.signif_type == 'categ':
                signif = self.signif_categ(y_truth, y_pred, self.test_masses,
                                           self.validation_peak_mask, data_frac=self.validation_fraction)

        signif_term = 0.0
        if self.signif_type != 'none':
            signif_term = + tf.math.divide(1, signif)

        return tf.keras.losses.categorical_crossentropy(y_truth, y_pred) + signif_term

    def predict_step(self, x, y_t, train=True):
        y_p = self.model(x, training=False)
        loss_val = self.loss_object(y_t, y_p, train).numpy().mean()
        acc_val = self.acc_object(y_t, y_p).numpy().mean()
        signif_val = 0
        if train:
            if self.signif_type == 'proba':
                signif_val = self.signif_proba(y_t, y_p, self.train_masses,
                                               self.train_peak_mask, data_frac=1-self.validation_fraction)
            elif self.signif_type == 'categ':
                signif_val = self.signif_categ(y_t, y_p, self.train_masses,
                                               self.train_peak_mask, data_frac=1-self.validation_fraction)
        else:
            if self.signif_type == 'proba':
                signif_val = self.signif_proba(y_t, y_p, self.test_masses,
                                               self.validation_peak_mask, data_frac=self.validation_fraction)
            elif self.signif_type == 'categ':
                signif_val = self.signif_categ(y_t, y_p, self.test_masses,
                                               self.validation_peak_mask, data_frac=self.validation_fraction)
        return loss_val, acc_val, signif_val

    def train(self, train_ds, test_ds, max_epochs, ot_cutoff=True, ot_cutoff_depth=10,
              verbose=True):
        self.ot_cutoff = ot_cutoff
        self.ot_cutoff_depth = ot_cutoff_depth
        self.max_epochs = max_epochs
        epochs, eval_loss, eval_accs, train_loss, train_accs = [], [], [], [], []
        test_loss, test_accs = [], []
        eval_sig, test_sig = [], []
        test_acc_sma = []
        test_loss_sma = []

        for epoch in range(self.max_epochs):
            out_text = ''
            epochs.append(epoch)
            lv, av, sv = 0, 0, 0
            for (batch, (xs, ys)) in enumerate(train_ds):
                lv, av = self.train_step(xs, ys)
            # print('Epoch {}/{} finished, learning rate: {:0.4f}'.format(epoch+1, self.max_epochs, self.optimizer.lr.numpy()))
            out_text += 'Epoch {}/{} finished, learning rate: {:0.4f}'.format(epoch+1, self.max_epochs, self.optimizer.lr.numpy()) + '\n'
            # print('train loss: \t{:0.4f} \t|\ttrain acc: \t{:0.4f}'.format(lv, av))
            out_text += 'train loss: \t{:0.4f} \t|\ttrain acc: \t{:0.4f}'.format(lv, av) + '\n'
            train_loss.append(lv)
            train_accs.append(av)
            for (batch, (xs, ys)) in enumerate(train_ds):
                lv, av, sv = self.predict_step(xs, ys)
            # print('eval loss: \t{:0.4f} \t|\teval acc: \t{:0.4f} \t|\teval signif: \t{:0.4f}'.format(lv, av, sv))
            out_text += 'eval loss: \t{:0.4f} \t|\teval acc: \t{:0.4f} \t|\teval signif: \t{:0.4f}'.format(lv, av, sv) + '\n'
            eval_loss.append(lv)
            eval_accs.append(av)
            eval_sig.append(sv)
            for (batch, (xs, ys)) in enumerate(test_ds):
                lv, av, sv = self.predict_step(xs, ys, train=False)
            # print('test loss: \t{:0.4f} \t|\ttest acc: \t{:0.4f} \t|\ttest signif: \t{:0.4f}'.format(lv, av, sv))
            out_text += 'test loss: \t{:0.4f} \t|\ttest acc: \t{:0.4f} \t|\ttest signif: \t{:0.4f}'.format(lv, av, sv) + '\n'
            test_loss.append(lv)
            test_accs.append(av)
            test_sig.append(sv)
            # print()
            if verbose:
                print(out_text)

            loss_slope = 0
            epos = []
            if epoch + 1 > ot_cutoff_depth:
                epos = np.linspace(1, ot_cutoff_depth, ot_cutoff_depth)
                loss_slope, _, _, _, _ = stats.linregress(epos, test_loss[-ot_cutoff_depth:])

            if self.ot_cutoff and epoch + 1 > self.ot_cutoff_depth and loss_slope >= 0:
                break

        dict = {'eps': epochs,
                'eval_accs': eval_accs, 'eval_loss': eval_loss,
                'eval_sig': eval_sig,
                'train_loss': train_loss, 'train_accs': train_accs,
                'test_loss': test_loss, 'test_accs': test_accs,
                'test_sig': test_sig}

        train_results_df = pd.DataFrame(dict)
        print('\nmodel trained for ' + str(len(epochs)) + ' epochs')
        print('final train accuracy:\t' + str(train_accs[-1]))
        print('final test accuracy:\t' + str(test_accs[-1]))
        self.train_results = train_results_df


class RelegatorClf(ModBinarySoftmaxClf):
    def __init__(self):
        self.optimizer = None
        self.layers = None
        self.model = None

        self.learning_rate = None
        self.hidden_layers_nodes = None
        self.bias = None
        self.n_inputs = None

        self.output_activation = 'softmax'
        self.name = 'relegation classifier'
        self.loss_type = 'rel. entropy + 1/sigma'

        self.acc_object = tf.keras.metrics.CategoricalAccuracy()

    def loss_object(self, y_truth, y_pred, train=True):
        signif = 0
        if train:
            if self.signif_type == 'proba':
                signif = self.signif_proba(y_truth, y_pred, self.train_masses,
                                           self.train_peak_mask, data_frac=1-self.validation_fraction)
            elif self.signif_type == 'categ':
                signif = self.signif_categ(y_truth, y_pred, self.train_masses,
                                           self.train_peak_mask, data_frac=1-self.validation_fraction)

        else:
            if self.signif_type == 'proba':
                signif = self.signif_proba(y_truth, y_pred, self.test_masses,
                                           self.validation_peak_mask, data_frac=self.validation_fraction)
            elif self.signif_type == 'categ':
                signif = self.signif_categ(y_truth, y_pred, self.test_masses,
                                           self.validation_peak_mask, data_frac=self.validation_fraction)

        rel_ent = self.relegator_cce(y_truth, y_pred)

        signif_term = 0.0
        if self.signif_type != 'none':
            signif_term = tf.math.divide(1, signif)

        return rel_ent + signif_term
        # return tf.keras.losses.categorical_crossentropy(y_truth, y_pred)

    def relegator_cce(self, y_truth, y_pred):
        y_p = []
        for i in range(self.n_outputs - 1):
            y_p.append(tf.transpose(tf.math.add(y_pred[:,i], y_pred[:,self.n_outputs-1])))
        y_p = tf.transpose(y_p)
        y_t = tf.slice(y_truth, [0, 0,], [len(y_truth), self.n_outputs - 1])
        return tf.keras.losses.categorical_crossentropy(y_t, y_p)

class RelegatorFactorClf(RelegatorClf):
    def __init__(self):
        self.optimizer = None
        self.layers = None
        self.model = None

        self.learning_rate = None
        self.hidden_layers_nodes = None
        self.bias = None
        self.n_inputs = None

        self.output_activation = 'softmax'
        self.name = 'relegation classifier'
        self.loss_type = 'rel. entropy + 1/sigma'

        self.acc_object = tf.keras.metrics.CategoricalAccuracy()

    def loss_object(self, y_truth, y_pred, train=True):
        signif = 0
        if train:
            if self.signif_type == 'proba':
                signif = self.signif_proba(y_truth, y_pred, self.train_masses,
                                           self.train_peak_mask, data_frac=1-self.validation_fraction)
            elif self.signif_type == 'categ':
                signif = self.signif_categ(y_truth, y_pred, self.train_masses,
                                           self.train_peak_mask, data_frac=1-self.validation_fraction)

        else:
            if self.signif_type == 'proba':
                signif = self.signif_proba(y_truth, y_pred, self.test_masses,
                                           self.validation_peak_mask, data_frac=self.validation_fraction)
            elif self.signif_type == 'categ':
                signif = self.signif_categ(y_truth, y_pred, self.test_masses,
                                           self.validation_peak_mask, data_frac=self.validation_fraction)

        rel_ent = self.relegator_cce(y_truth, y_pred)
        return tf.math.divide(rel_ent, signif)

class RelegatorDiffClf(RelegatorClf):
    def __init__(self):
        self.optimizer = None
        self.layers = None
        self.model = None

        self.learning_rate = None
        self.hidden_layers_nodes = None
        self.bias = None
        self.n_inputs = None

        self.output_activation = 'softmax'
        self.name = 'relegation classifier'
        self.loss_type = 'rel. entropy + 1/sigma'

        self.acc_object = tf.keras.metrics.CategoricalAccuracy()

    def loss_object(self, y_truth, y_pred, train=True):
        signif = 0
        if train:
            if self.signif_type == 'proba':
                signif = self.signif_proba(y_truth, y_pred, self.train_masses,
                                           self.train_peak_mask, data_frac=1-self.validation_fraction)
            elif self.signif_type == 'categ':
                signif = self.signif_categ(y_truth, y_pred, self.train_masses,
                                           self.train_peak_mask, data_frac=1-self.validation_fraction)

        else:
            if self.signif_type == 'proba':
                signif = self.signif_proba(y_truth, y_pred, self.test_masses,
                                           self.validation_peak_mask, data_frac=self.validation_fraction)
            elif self.signif_type == 'categ':
                signif = self.signif_categ(y_truth, y_pred, self.test_masses,
                                           self.validation_peak_mask, data_frac=self.validation_fraction)

        rel_ent = self.relegator_cce(y_truth, y_pred)
        return rel_ent - signif
