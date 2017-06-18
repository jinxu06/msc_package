import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
import time
from contrib import enumerate_parameters

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers
from keras import initializers

from keras.callbacks import EarlyStopping


def log_sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    x_max = K.max(x, axis=axis, keepdims=True)
    return K.log(K.sum(K.exp(x - x_max),
                       axis=axis, keepdims=True))+x_max



class ConditionalModel(object):

    def __init__(self, attr_type):
        pass

    def fit(self, X, y, verbose=1):
        pass

    def query_proba(self, X):
        pass

    def evaluate(self, X, y):
        pass

    def train_K_fold(self, K, X, y, K_max_run=None, verbose=1):
        if K_max_run is None:
            K_max_run = K
        errs = []
        for k in range(min(K, K_max_run)):
            valid_index = range(k, X.shape[0],K)
            t_inputs = np.delete(X, valid_index, axis=0)
            t_targets = np.delete(y, valid_index, axis=0)
            v_inputs = X[valid_index,:]
            v_targets = y[valid_index]
            self.fit(t_inputs, t_targets, verbose=verbose)
            e = self.evaluate(v_inputs, v_targets)
            errs.append(e)
        return np.mean(errs)


    def search_hyper_params(self, K, train_inputs, train_targets, K_max_run=None, random_max_run=None, verbose=1):
        if random_max_run is None:
            random_max_run = len(self.hyper_params_choices)
        best_hyper_params = None
        best_err = 1e10
        for i in range(random_max_run):
            hyper_params = self.hyper_params_choices[i]
            self.set_model(hyper_params)
            e = self.train_K_fold(K, train_inputs, train_targets, K_max_run=K_max_run, verbose=verbose)
            if e < best_err:
                best_err = e
                best_hyper_params = hyper_params
        self.set_model(best_hyper_params)
        self.fit(train_inputs, train_targets, verbose=verbose)
        return best_hyper_params, best_err

class MixtureDensityNetwork(ConditionalModel):

    def __init__(self, base_model, hyper_params, inputs_dim, random=False):
        self.inputs_dim = inputs_dim
        self.base_model = base_model
        self.hyper_params_choices = enumerate_parameters(hyper_params)
        if random:
            num_choices = len(self.hyper_params_choices)
            sel = np.random.choice(range(num_choices),
                                        size=(num_choices,), replace=False)
            self.hyper_params_choices = [self.hyper_params_choices[s] for s in sel]

    def set_model(self, hyper_params):
        self.model = Sequential()
        num_hidden_units = hyper_params['num_hidden_units']
        num_hidden_layers = hyper_params['num_hidden_layers']
        activation = hyper_params['activation']
        l2_scale = hyper_params['l2_scale']
        self.n_components = hyper_params['n_components']

        kernel_initializer = initializers.glorot_uniform(seed=123)
        kernel_regularizer = regularizers.l2(l2_scale)


        self.model.add(Dense(hyper_params['num_hidden_units'], activation=activation,
                                        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, input_shape=(self.inputs_dim,)))
        for l in range(hyper_params['num_hidden_layers']-1):
            self.model.add(Dense(hyper_params['num_hidden_units'], activation=activation,
                                            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, input_shape=(hyper_params['num_hidden_units'],)))
        if self.base_model=='Gaussian':
            self.model.add(Dense(self.n_components*3, kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer, input_shape=(hyper_params['num_hidden_units'],)))
            self.model.compile(loss=self._mdn_gaussian_loss, optimizer='adam')
        elif self.base_model=='Poisson':
            self.model.add(Dense(self.n_components*2, kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer, input_shape=(hyper_params['num_hidden_units'],)))
            self.model.compile(loss=self._mdn_poisson_loss, optimizer='adam')



    def _mdn_gaussian_loss(self, y_true, y_pred):
        self.mus = y_pred[:, :self.n_components]
        self.sigmas = K.exp(y_pred[:, self.n_components:self.n_components*2])
        self.alphas = K.softmax(y_pred[:, self.n_components*2:])

        exponent = K.log(self.alphas) + tf.contrib.distributions.Normal(loc=self.mus, scale=self.sigmas).log_prob(y_true)
        #Z = (2 * np.pi * (self.sigmas**2))**0.5
        #normal = lambda x: tf.exp(-0.5 * (x - self.mus)**2 / (self.sigmas**2)) / Z
        res = log_sum_exp(exponent, axis=1)
        res = - K.mean(res)
        return res

    def _mdn_poisson_loss(self, y_true, y_pred):
        self.lambdas = K.exp(y_pred[:, :self.n_components])
        self.alphas = K.softmax(y_pred[:, self.n_components:])
        exponent = K.log(self.alphas) + tf.contrib.distributions.Poisson(rate=self.lambdas).log_prob(y_true)
        res = log_sum_exp(exponent, axis=1)
        res = - K.mean(res)
        return res

    def fit(self, X, y, max_num_epochs=500, validation_split=0.2, batch_size=100, verbose=1):
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        self.model.fit(X, y, validation_split=validation_split, callbacks=[early_stopping], epochs=max_num_epochs,  batch_size=batch_size, verbose=verbose)

    def evaluate(self, X, y, batch_size=100):
        return self.model.evaluate(X, y, batch_size=batch_size)

    def query_proba(self, X):
        pred = self.model.predict(X)
        pred = pred.astype(np.float64)
        if self.base_model == 'Gaussian':
            mus = pred[:, :self.n_components]
            sigmas = np.exp(pred[:, self.n_components:self.n_components*2])
            exponent = np.exp(pred[:, self.n_components*2:] - np.max(pred[:, self.n_components*2:], axis=1)[:, None])
            alphas = exponent / np.sum(exponent, axis=1)[:, None]
            return mus, sigmas, alphas
        elif self.base_model == 'Poisson':
            lambdas = np.exp(pred[:, :self.n_components])
            exponent = np.exp(pred[:, self.n_components:] - np.max(pred[:, self.n_components:], axis=1)[:, None])
            alphas = exponent / np.sum(exponent, axis=1)[:, None]
            return lambdas, alphas


class SklearnConditionalModel(ConditionalModel):

    def __init__(self, method, hyper_params, num_classes, random=False):
        self.method = method
        self.num_classes = num_classes

        self.hyper_params_choices = enumerate_parameters(hyper_params)
        if random:
            num_choices = len(self.hyper_params_choices)
            sel = np.random.choice(range(num_choices),
                                        size=(num_choices,), replace=False)
            self.hyper_params_choices = [self.hyper_params_choices[s] for s in sel]

    def set_model(self, hyper_params):
        if self.method==xgb.XGBClassifier:
            hyper_params['num_class'] = self.num_classes
        self.model = self.method(**hyper_params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        proba = self.query_proba(X)
        error = []
        for p, t in zip(proba, y):
            error.append(-np.log(p[t]))
        return np.mean(error)#, np.std(error, ddof=1) / len(error)

    def query_proba(self, X):
        min_proba = 1e-8
        proba = self.model.predict_proba(X)
        proba = np.maximum(proba, min_proba)
        classes = self.model.classes_
        predictions = np.zeros((X.shape[0], self.num_classes))
        for c in range(self.num_classes):
            if c in classes:
                idx = list(classes).index(c)
                predictions[:, c:c+1] = proba[:, idx:idx+1]
            else:
                predictions[:, c:c+1] = np.ones((X.shape[0],1)) * min_proba
        predictions = predictions.astype(np.float64)
        predictions /= np.sum(predictions, axis=1)[:, None]
        return predictions

    """
    def train_K_fold(self, K, X, y, K_max_run=None):
        if K_max_run is None:
            K_max_run = K
        errs = []
        for k in range(min(K, K_max_run)):
            valid_index = range(k, X.shape[0],K)
            t_inputs = np.delete(X, valid_index, axis=0)
            t_targets = np.delete(y, valid_index, axis=0)
            v_inputs = X[valid_index,:]
            v_targets = y[valid_index]
            self.fit(t_inputs, t_targets)
            e = self.evaluate(v_inputs, v_targets)
            errs.append(e)
        return np.mean(errs)

    def search_hyper_params(self, K, train_inputs, train_targets, K_max_run=None, random_max_run=None):
        if random_max_run is None:
            random_max_run = len(self.hyper_params_choices)
        best_hyper_params = None
        best_err = 1e10
        for i in range(random_max_run):
            hyper_params = self.hyper_params_choices[i]
            self.set_model(hyper_params)
            e = self.train_K_fold(K, train_inputs, train_targets, K_max_run=K_max_run)
            if e < best_err:
                best_err = e
                best_hyper_params = hyper_params
        self.set_model(best_hyper_params)
        self.fit(train_inputs, train_targets)
        return best_hyper_params, best_err

    """

class NDependencyNetwork(object):

    def __init__(self, model_settings, inputs_block, attr_types, name=None):
        self.attr_types = attr_types
        self.num_attr = len(self.attr_types)
        self.inputs_block = inputs_block
        self.num_inputs = self.inputs_block[-1][1]
        self.name = name

        self.model_settings = model_settings
        self.hyper_params_choices = []
        self.methods = []
        for a in attr_types:
            self.methods.append(model_settings[a])
            self.hyper_params_choices.append(model_settings[model_settings[a][1]])

        self._define_masks()

    def _define_masks(self):
        self.masks = []
        for block in self.inputs_block:
            mask = np.ones((self.num_inputs,), dtype=np.int64)
            mask[block[0]:block[1]] = 0
            self.masks.append(mask)
        self.masks = np.array(self.masks)

    #def set_models(self, methods, hyper_params):
    #    self.models = []
    #    for m, p in zip(methods, hyper_params):
    #        self.models.append(m(**p))

    def train(self, train_inputs, K=10, K_max_run=None, random_max_run=None, verbose=1):
        self.models = []
        for mask, block, attr_type, method, hyper_params in zip(self.masks, self.inputs_block, self.attr_types,
                        self.methods, self.hyper_params_choices):
            inputs = train_inputs * mask
            targets = train_inputs[:, block[0]:block[1]]
            if attr_type=='c':
                targets = np.argmax(targets, axis=1)
                num_classes = block[1]-block[0]
            elif attr_type=='b':
                targets = targets[:, 0]
                num_classes = 2
            elif attr_type=='i' or attr_type=='r':
                targets = targets[:, 0]

            if str(method[0])==str(SklearnConditionalModel):
                model = SklearnConditionalModel(method[1], hyper_params, num_classes, random=True)
            elif str(method[0])==str(MixtureDensityNetwork):
                model = MixtureDensityNetwork(method[1], hyper_params, train_inputs.shape[-1], random=True)
            else:
                raise Exception("model not found")
            p, e = model.search_hyper_params(K, inputs, targets, K_max_run=K_max_run, random_max_run=random_max_run, verbose=verbose)
            print p, e
            self.models.append(model)

    def query(self, query_inputs):
        ret = []
        for model, mask in zip(self.models, self.masks):
            proba = model.query_proba(query_inputs * mask)
            ret.append(proba)
        return ret
