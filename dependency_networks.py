import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
import time
import cPickle as pkl
import os
from contrib import enumerate_parameters
from classifiers import SklearnClassifier
from synthetic_data_discriminators import SyntheticDataDiscriminator

from keras import backend as Kb
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers
from keras import initializers

from keras.callbacks import EarlyStopping
from keras.models import load_model
from scipy.stats import poisson
from contrib import resample_with_replacement


def log_sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    x_max = Kb.max(x, axis=axis, keepdims=True)
    return Kb.log(Kb.sum(Kb.exp(x - x_max),
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

class GANDependencyNetwork(object):

    def __init__(self, hyper_params, noise_generator, inputs_dim, prior_dim, inputs_block, attr_types, random=False, name="GDN"):
        self.hyper_params = hyper_params
        self.noise_generator = noise_generator
        self.inputs_dim = inputs_dim
        self.prior_dim = prior_dim
        self.inputs_block = inputs_block
        self.attr_types = attr_types
        self.name = name

        self.hyper_params_choices = enumerate_parameters(hyper_params)
        if random:
            num_choices = len(self.hyper_params_choices)
            sel = np.random.choice(range(num_choices),
                                        size=(num_choices,), replace=False)
            self.hyper_params_choices = [self.hyper_params_choices[s] for s in sel]

        self.set_model(self.hyper_params_choices[0])

    def set_dis_l2_scale(self, value, ratio=None):
        for model in self.models:
            model.set_dis_l2_scale(value, ratio)

    def set_model(self, hyper_params):
        self.models = []
        for k, block in enumerate(self.inputs_block):
            model = GenerativeAdversarialNetwork(self.hyper_params, self.noise_generator, self.inputs_dim, self.prior_dim, block, name="{0}GDN-{1}".format(self.name, k))
            model.set_model(hyper_params)
            model.init_variables()
            self.models.append(model)

    def fit(self, train_inputs, num_epochs=100):
        for model, block in zip(self.models, self.inputs_block):
            model.train(train_inputs, num_epochs)

    def run_sampling(self, initial_samples, num_round, max_step=None, reject=True, same_size=True):
        sampling_order = np.random.permutation(range(len(self.inputs_block)))
        samples = initial_samples.copy()
        step = 0
        for o in sampling_order:
            model = self.models[o]
            begin, end = self.inputs_block[o][0], self.inputs_block[o][1]
            gen_col, perturbed_inputs = model.generate(samples, reject, same_size)
            samples = perturbed_inputs
            samples[:, begin:end] = gen_col
            step += 1
            if max_step is not None and step>=max_step:
                break
        return samples

class GenerativeAdversarialNetwork(object):

    def __init__(self, hyper_params, noise_generator, inputs_dim, prior_dim, block, name="GAN{0}".format(np.random.randint(1e6)), random=False):
        self.name = name
        self.inputs_dim = inputs_dim
        self.prior_dim = prior_dim
        self.block = block
        self.noise_generator = noise_generator
        self.best_eval = 1.
        self.best_params = None
        self.mask = np.ones((self.inputs_dim,))
        self.mask[block[0]:block[1]] = 0
        self.session = tf.Session()

    def __del__(self):
        self.session.close()

    def init_variables(self):
        self.session.run(self.init)

    def set_dis_l2_scale(self, value, ratio=None):
        if ratio is not None:
            self.cur_dis_l2_scale *= ratio
        else:
            self.cur_dis_l2_scale = value

    def _get_all_params(self):
        gen_vals = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+"-generator")
        dis_vals = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+"-discriminator")
        return self.session.run([gen_vals, dis_vals])

    def _assign_all_params(self, gen_values, dis_values):
        gen_vals = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+"-generator")
        dis_vals = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+"-discriminator")
        with tf.variable_scope(self.name+"-generator"):
            opts = [val.assign(value) for val, value in zip(gen_vals, gen_values)]
            opt = tf.group(*opts)
            self.session.run(opt)
        with tf.variable_scope(self.name+"-discriminator"):
            opts = [val.assign(value) for val, value in zip(dis_vals, dis_values)]
            opt = tf.group(*opts)
            self.session.run(opt)

    def train_discriminator_step(self, X):
        batch_size = X.shape[0]
        noise = self.noise_generator(size=(batch_size, self.prior_dim))
        feed_dict = {}
        feed_dict[self.inputs] = X
        feed_dict[self.prior_noise] = noise
        feed_dict[self.masks] = np.broadcast_to(self.mask, shape=X.shape)
        feed_dict[self.targets] = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))], axis=0)
        feed_dict[self.is_training] = True
        feed_dict[self.dis_l2_scale] = self.cur_dis_l2_scale
        self.session.run(self.discriminator_optimizer, feed_dict=feed_dict)
        return self.session.run(self.error, feed_dict=feed_dict)

    def train_generator_step(self, X):
        batch_size = X.shape[0]
        noise = self.noise_generator(size=(batch_size, self.prior_dim))
        feed_dict = {}
        feed_dict[self.inputs] = X
        feed_dict[self.prior_noise] = noise
        feed_dict[self.masks] = np.broadcast_to(self.mask, shape=X.shape)
        feed_dict[self.targets] = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))], axis=0)
        feed_dict[self.is_training] = False
        feed_dict[self.dis_l2_scale] = self.cur_dis_l2_scale
        self.session.run(self.generator_optimizer, feed_dict=feed_dict)
        return self.session.run(self.error, feed_dict=feed_dict)

    def discriminate(self, train_inputs, reject=True):
        gen_col, perturbed_inputs = self.generate(train_inputs, reject=reject, same_size=False)
        gen_inputs = perturbed_inputs.copy()
        gen_inputs[:, self.block[0]:self.block[1]] = gen_col
        hyper_params = {
            "max_depth": [4],
            "n_estimators": [4, 8, 16, 32, 64]
        }
        cls = SklearnClassifier(xgb.XGBClassifier, hyper_params)
        D = SyntheticDataDiscriminator(cls, sampling_size=min(gen_inputs.shape[0], train_inputs.shape[0]))
        D.feed_data(train_inputs, np.zeros((train_inputs.shape[0],)), gen_inputs, np.zeros((gen_inputs.shape[0],)))
        return D.discriminate()[0]

    def train_epoch(self, train_inputs, K, batch_size=100, verbose=1):
        train_inputs = train_inputs.copy()
        np.random.shuffle(train_inputs)
        start = 0
        dis_errs = []
        gen_errs = []
        while start < train_inputs.shape[0]:
            end = min(start+batch_size, train_inputs.shape[0])
            batch = train_inputs[start:end, :]
            for k in range(K):
                dis_e = self.train_discriminator_step(batch)
            dis_errs.append(dis_e)
            gen_e = self.train_generator_step(batch)
            gen_errs.append(gen_e)
            start = end
        if verbose > 0:
            print "DIS>>", np.array(dis_errs).mean()
            print "GEN>>", np.array(gen_errs).mean()
        return np.array(dis_errs).mean(), np.array(gen_errs).mean()

    def train(self, train_inputs, num_epochs, K=1, evaluate_freq=10, batch_size=100, verbose=1):
        for i in range(num_epochs):
            if verbose > 0:
                print "epoch {0} ------".format(i)
            dis_err, gen_err = self.train_epoch(train_inputs, K=K, batch_size=batch_size, verbose=verbose)

            if (i+1) % evaluate_freq == 0:
                T = 3
                acc = 0.
                for t in range(T):
                    acc += self.discriminate(train_inputs, reject=True)
                acc /= T
                if acc < self.best_eval:
                    self.best_eval = acc
                    self.best_params = self._get_all_params()
                    if self.best_eval < 0.51:
                        break
        print "Best Eval: ", self.best_eval
        self._assign_all_params(*self.best_params)
        return self.best_eval

    def generate(self, X, reject=True, same_size=True):
        batch_size = X.shape[0]
        arr = []
        inputs = X.copy()
        perturbed_inputs = []
        new_samples = []
        while inputs.shape[0]>0:
            noise = self.noise_generator(size=(inputs.shape[0], self.prior_dim))
            feed_dict = {}
            feed_dict[self.inputs] = inputs
            feed_dict[self.prior_noise] = noise
            feed_dict[self.masks] = np.broadcast_to(self.mask, shape=inputs.shape)
            feed_dict[self.is_training] = False
            feed_dict[self.dis_l2_scale] = self.cur_dis_l2_scale
            samples, proba = self.session.run([self.predictions, self.proba], feed_dict=feed_dict)
            if not reject:
                new_samples = samples
                perturbed_inputs = X.copy()
                break
            proba = proba[inputs.shape[0]:, :]
            new_inputs = []
            for p, s, inp in zip(proba[:, 0], samples, inputs):
                if np.random.uniform() < p:
                    new_samples.append(s)
                    perturbed_inputs.append(inp)
                else:
                    new_inputs.append(inp)
            inputs = np.array(new_inputs)
            if not same_size:
                break
        samples = np.array(new_samples)
        perturbed_inputs = np.array(perturbed_inputs)

        return samples, perturbed_inputs

    def set_model(self, hyper_params):

        self.cur_dis_l2_scale = hyper_params['dis_l2_scale']
        self.inputs = tf.placeholder(tf.float64, [None, self.inputs_dim], 'inputs')
        self.prior_noise = tf.placeholder(tf.float64, [None, self.prior_dim], 'prior')
        self.targets = tf.placeholder(tf.float64, [None,  1], 'targets')
        self.masks = tf.placeholder(tf.float64, [None, self.inputs_dim], 'masks')
        self.is_training = tf.placeholder(tf.bool, (), "is_training")
        self.dis_l2_scale = tf.placeholder(tf.float64, (), "dis_l2_scale")
        self.predictions = self._build_generator(hyper_params, self.inputs, self.prior_noise, self.block)
        self.discriminator_optimizer = self._build_discriminator(hyper_params, self.inputs, self.predictions, self.block, self.targets)
        self.generator_optimizer = self._build_generator_optimizer(hyper_params, self.error)

        var_list_gen = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+"-generator")
        var_list_dis = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+"-discriminator")
        var_list = var_list_gen+var_list_dis
        self.init = tf.variables_initializer(var_list=var_list)


    def _build_generator(self, hyper_params, inputs, prior_noise, block):
        num_hidden_units = hyper_params['num_hidden_units']
        num_hidden_layers = hyper_params['num_hidden_layers']
        activation = hyper_params["activation"]
        l2_scale = hyper_params["l2_scale"]
        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_scale)
        kernel_initializer = tf.contrib.layers.xavier_initializer()

        comb_inputs = tf.concat([inputs * self.mask, prior_noise], axis=1)
        layers = comb_inputs

        with tf.variable_scope(self.name+"-generator"):
            for l in range(num_hidden_layers):
                layers = tf.layers.dense(layers, num_hidden_units, activation,
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer)
            layers = tf.layers.dense(layers, self.block[1]-self.block[0],
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer)
            return layers


    def _build_generator_optimizer(self, hyper_params, error):
        learning_rate = hyper_params['learning_rate']

        with tf.variable_scope(self.name+"-generator"):
            l2_loss = tf.losses.get_regularization_loss(scope=self.name+"-generator")
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+"-generator")
            generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-error+l2_loss, var_list=var_list)
            return generator_optimizer

    def _build_discriminator(self, hyper_params, inputs, predictions, block, targets):
        num_hidden_units = hyper_params['dis_num_hidden_units']
        num_hidden_layers = hyper_params['dis_num_hidden_layers']
        activation = hyper_params["activation"]
        learning_rate = hyper_params['learning_rate']
        dropout_rate = hyper_params['dropout_rate']
        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=self.dis_l2_scale)
        kernel_initializer = tf.contrib.layers.xavier_initializer()

        gen_data = inputs*self.mask + tf.pad(predictions, paddings=[[0, 0], [block[0], self.inputs_dim-block[1]]])
        gen_data = tf.concat([gen_data, 1-self.masks], axis=1)
        ori_data = inputs
        ori_data = tf.concat([ori_data, 1-self.masks], axis=1)
        dis_inputs = tf.concat([ori_data, gen_data], axis=0)
        layers = dis_inputs

        with tf.variable_scope(self.name+"-discriminator"):
            layers = tf.layers.dropout(layers, rate=dropout_rate, training=self.is_training)
            for l in range(num_hidden_layers):
                layers = tf.layers.dense(layers, num_hidden_units, activation,
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer)
                layers = tf.layers.dropout(layers, rate=dropout_rate, training=self.is_training)
            layers = tf.layers.dense(layers, 1,
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer)
            self.proba = tf.nn.sigmoid(layers)
            error = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=layers)
            self.error = tf.reduce_mean(error)
            l2_loss = tf.losses.get_regularization_loss(scope=self.name+"-discriminator")
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+"-discriminator")
            discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.error+l2_loss, var_list=var_list)
            return discriminator_optimizer




class PoissonConditionalModel(ConditionalModel):

    def __init__(self, hyper_params, random=True, name=None):
        self.name = name
        self.hyper_params_choices = enumerate_parameters(hyper_params)
        if random:
            num_choices = len(self.hyper_params_choices)
            sel = np.random.choice(range(num_choices),
                                        size=(num_choices,), replace=False)
            self.hyper_params_choices = [self.hyper_params_choices[s] for s in sel]

    def set_model(self, hyper_params):
        self.params = hyper_params

    def fit(self, X, y, verbose=1):
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(self.params, dtrain)

    def evaluate(self, X, y):
        preds = self.model.predict(xgb.DMatrix(X))
        arr = []
        for lam, t in zip(preds, y):
            p = poisson(lam)
            arr.append(-np.log(p.pmf(t)))
        return np.mean(arr)

    def query_proba(self, X, temperature=1.):
        preds = self.model.predict(xgb.DMatrix(X))
        return preds[:, None], np.ones((preds.shape[0],1))

class BaggingPoissonConditionalModel(ConditionalModel):

    def __init__(self, hyper_params, random=True, name=None):
        self.name = name
        self.hyper_params_choices = enumerate_parameters(hyper_params)
        if random:
            num_choices = len(self.hyper_params_choices)
            sel = np.random.choice(range(num_choices),
                                        size=(num_choices,), replace=False)
            self.hyper_params_choices = [self.hyper_params_choices[s] for s in sel]

    def set_model(self, hyper_params):
        self.params = hyper_params

    def fit(self, X, y, verbose=1):
        self.n_bagging = self.params["n_bagging"]
        self.models = []
        for i in range(self.n_bagging):
            t_inputs, t_targets, v_inputs, v_targets = resample_with_replacement(X, y)
            dtrain = xgb.DMatrix(t_inputs, label=t_targets)
            model = xgb.train(self.params, dtrain)
            self.models.append(model)

    def evaluate(self, X, y):
        all_preds = []
        for model in self.models:
            preds = model.predict(xgb.DMatrix(X))
            all_preds.append(preds)
        all_preds = np.array(all_preds).T
        arr = []
        for lams, t in zip(all_preds, y):
            ps = []
            for lam in lams:
                p = poisson(lam).pmf(t)
                ps.append(p)
            prob = np.mean(ps)
            arr.append(-np.log(prob))
        return np.mean(arr)

    def query_proba(self, X, temperature=1.):
        all_preds = []
        for model in self.models:
            preds = model.predict(xgb.DMatrix(X))
            all_preds.append(preds)
        all_preds = np.array(all_preds).T
        alphas = np.ones(all_preds.shape, dtype=np.float64) / all_preds.shape[1]
        return all_preds, alphas



class MixtureDensityNetwork(ConditionalModel):

    def __init__(self, base_model, hyper_params, inputs_dim, random=True, name=None):
        self.inputs_dim = inputs_dim
        self.base_model = base_model
        if self.base_model == "Gaussian":
            self.loss_func = "_mdn_gaussian_loss"
        elif self.base_model == "Poisson":
            self.loss_func = "_mdn_poisson_loss"
        elif self.base_model == "NegativeBinomial":
            self.loss_func = "_mdn_negative_binomial_loss"
        self.hyper_params_choices = enumerate_parameters(hyper_params)
        self.name = name
        self.n_components = None
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

        elif self.base_model=='NegativeBinomial':
            self.model.add(Dense(self.n_components*3, kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer, input_shape=(hyper_params['num_hidden_units'],)))
            self.model.compile(loss=self._mdn_negative_binomial_loss, optimizer='adam')




    def _mdn_gaussian_loss(self, y_true, y_pred):
        if self.n_components is None:
            self.n_components = int(y_pred.shape[1]/3)
        self.mus = y_pred[:, :self.n_components]
        self.sigmas = Kb.exp(y_pred[:, self.n_components:self.n_components*2])
        self.alphas = Kb.softmax(y_pred[:, self.n_components*2:])

        exponent = Kb.log(self.alphas) + tf.contrib.distributions.Normal(loc=self.mus, scale=self.sigmas).log_prob(y_true)
        #Z = (2 * np.pi * (self.sigmas**2))**0.5
        #normal = lambda x: tf.exp(-0.5 * (x - self.mus)**2 / (self.sigmas**2)) / Z
        res = log_sum_exp(exponent, axis=1)
        res = - Kb.mean(res)
        return res

    def _mdn_poisson_loss(self, y_true, y_pred):
        if self.n_components is None:
            self.n_components = int(y_pred.shape[1]/2)
        self.lambdas = Kb.exp(y_pred[:, :self.n_components])
        self.alphas = Kb.softmax(y_pred[:, self.n_components:])
        exponent = Kb.log(self.alphas) + tf.contrib.distributions.Poisson(rate=self.lambdas).log_prob(y_true)
        res = log_sum_exp(exponent, axis=1)
        res = - Kb.mean(res)
        return res

    def _mdn_negative_binomial_loss(self, y_true, y_pred):
        if self.n_components is None:
            self.n_components = int(y_pred.shape[1]/3)
        self.total_counts = Kb.exp(y_pred[:, :self.n_components])
        self.probs = Kb.sigmoid(y_pred[:, self.n_components:2*self.n_components])
        self.alphas = Kb.softmax(y_pred[:, 2*self.n_components:])

        exponent = Kb.log(self.alphas) + tf.contrib.distributions.NegativeBinomial(total_count=self.total_counts, probs=self.probs).log_prob(y_true)
        res = log_sum_exp(exponent, axis=1)
        res = - Kb.mean(res)
        return res

    def fit(self, X, y, max_num_epochs=500, validation_split=0.2, batch_size=100, verbose=1):
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        self.model.fit(X, y, validation_split=validation_split, callbacks=[early_stopping], epochs=max_num_epochs,  batch_size=batch_size, verbose=verbose)

    def evaluate(self, X, y, batch_size=100):
        return self.model.evaluate(X, y, batch_size=batch_size)

    def query_proba(self, X, temperature=1.):
        pred = self.model.predict(X)
        pred = pred.astype(np.float64)
        if self.base_model == 'Gaussian':
            mus = pred[:, :self.n_components]
            sigmas = np.exp(pred[:, self.n_components:self.n_components*2])
            exponent = np.exp(pred[:, self.n_components*2:] - np.max(pred[:, self.n_components*2:], axis=1)[:, None])
            alphas = exponent / np.sum(exponent, axis=1)[:, None]
            sigmas /= temperature
            return mus, sigmas, alphas
        elif self.base_model == 'Poisson':
            lambdas = np.exp(pred[:, :self.n_components])
            exponent = np.exp(pred[:, self.n_components:] - np.max(pred[:, self.n_components:], axis=1)[:, None])
            alphas = exponent / np.sum(exponent, axis=1)[:, None]
            lambdas /= temperature
            return lambdas, alphas
        elif self.base_model == 'NegativeBinomial':
            total_counts = np.exp(pred[:, :self.n_components])
            probs = 1. / (1. + np.exp(-pred[:, self.n_components:2*self.n_components]))
            exponent = np.exp(pred[:, 2*self.n_components:] - np.max(pred[:, 2*self.n_components:], axis=1)[:, None])
            alphas = exponent / np.sum(exponent, axis=1)[:, None]
            #lambdas /= temperature
            return total_counts, probs, alphas

    def save_model(self):
        if self.name is None:
            self.name = "MDN{0}".format(hash(self.model))
        self.model.save("../models/{0}.h5".format(self.name))

    def load_model(self):
        self.model = load_model("../models/{0}.h5".format(self.name), custom_objects={self.loss_func: eval("self.{0}".format(self.loss_func))})

    def delete_model(self):
        os.remove("../models/{0}.h5".format(self.name))


class SklearnConditionalModel(ConditionalModel):

    def __init__(self, method, hyper_params, num_classes, random=False, name=None):
        self.method = method
        self.num_classes = num_classes
        self.name = name
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

    def fit(self, X, y, verbose=1.):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        proba = self.query_proba(X)
        error = []
        for p, t in zip(proba, y):
            error.append(-np.log(p[t]))
        return np.mean(error)#, np.std(error, ddof=1) / len(error)

    def query_proba(self, X, temperature=1.):
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
        predictions = predictions** temperature
        predictions /= np.sum(predictions, axis=1)[:, None]
        return predictions

    def save_model(self):
        with open("../models/{0}.pkl".format(self.name), 'w') as f:
            pkl.dump(self.model, f)


    def load_model(self):
        with open("../models/{0}.pkl".format(self.name), 'r') as f:
            self.model = pkl.load(f)

    def delete_model(self):
        os.remove("../models/{0}.pkl".format(self.name))

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

    def train(self, train_inputs, K=10, K_max_run=None, random_max_run=None, verbose=1, save=False):
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
            elif attr_type=='i' or attr_type=='r' or attr_type=='ib':
                targets = targets[:, 0]

            if str(method[0])==str(SklearnConditionalModel):
                model = SklearnConditionalModel(method[1], hyper_params, num_classes, random=True, name=self.name+str(block[0]))
            elif str(method[0])==str(MixtureDensityNetwork):
                model = MixtureDensityNetwork(method[1], hyper_params, train_inputs.shape[-1], random=True, name=self.name+str(block[0]))
            elif str(method[0])==str(PoissonConditionalModel):
                model = PoissonConditionalModel(hyper_params)
            elif str(method[0])==str(BaggingPoissonConditionalModel):
                model = BaggingPoissonConditionalModel(hyper_params)
            else:
                raise Exception("model not found")
            cur = time.time()
            p, e = model.search_hyper_params(K, inputs, targets, K_max_run=K_max_run, random_max_run=random_max_run, verbose=verbose)
            if save:
                model.save_model()
                Kb.clear_session()
            print p, e, time.time()-cur
            self.models.append(model)

    def restore_models(self):
        self.models = []
        for mask, block, attr_type, method, hyper_params in zip(self.masks, self.inputs_block, self.attr_types,
                        self.methods, self.hyper_params_choices):
            if attr_type=='c':
                num_classes = block[1]-block[0]
            elif attr_type=='b':
                num_classes = 2

            if str(method[0])==str(SklearnConditionalModel):
                model = SklearnConditionalModel(method[1], hyper_params, num_classes, random=True, name=self.name+str(block[0]))
            elif str(method[0])==str(MixtureDensityNetwork):
                model = MixtureDensityNetwork(method[1], hyper_params, self.inputs_block[-1][-1], random=True, name=self.name+str(block[0]))
            elif str(method[0])==str(PoissonConditionalModel):
                model = PoissonConditionalModel(hyper_params)
            elif str(method[0])==str(BaggingPoissonConditionalModel):
                model = BaggingPoissonConditionalModel(hyper_params)
            else:
                raise Exception("model not found")
            self.models.append(model)
        for model in self.models:
            model.load_model()

    def delete_models(self):
        for model in self.models:
            model.delete_model()

    def query(self, query_inputs, temperature=1.):
        ret = []
        for model, mask, method in zip(self.models, self.masks, self.methods):
            proba = model.query_proba(query_inputs * mask, temperature=temperature)
            ret.append(proba)
        return ret
