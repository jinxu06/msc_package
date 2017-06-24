import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
import time
from contrib import enumerate_parameters, one_hot_encoding
from classifiers import SklearnClassifier
from synthetic_data_discriminators import SyntheticDataDiscriminator

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

    def __init__(self, hyper_params, noise_generator, inputs_dim, num_classes, prior_dim, block, name="GAN{0}".format(np.random.randint(1e6)), random=False):
        self.name = name
        self.inputs_dim = inputs_dim
        self.prior_dim = prior_dim
        self.num_classes = num_classes
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

    def train_discriminator_step(self, X, y):
        batch_size = X.shape[0]
        feed_dict = {}
        feed_dict[self.inputs] = X
        #noise = self.noise_generator(size=(batch_size, self.prior_dim))
        self.noise_generator.reset(X, y)
        noise, _ = self.noise_generator.generate(1, shuffle=False, include_original_data=False)
        noise = noise[:, self.block[0]:self.block[1]]
        feed_dict[self.prior_noise] = noise
        feed_dict[self.targets] = one_hot_encoding(y, self.num_classes)
        feed_dict[self.masks] = np.broadcast_to(self.mask, shape=X.shape)
        feed_dict[self.is_training] = True
        feed_dict[self.dis_l2_scale] = self.cur_dis_l2_scale

        feed_dict[self.is_gen] = 1.
        feed_dict[self.source_targets] = np.zeros((batch_size, 1))
        self.session.run(self.discriminator_optimizer, feed_dict=feed_dict)
        feed_dict[self.is_gen] = 0.
        feed_dict[self.source_targets] = np.ones((batch_size, 1))

        #feed_dict[self.source_targets] = np.concatenate([np.zeros((batch_size, 1)), np.ones((batch_size, 1))], axis=0)
        self.session.run(self.discriminator_optimizer, feed_dict=feed_dict)

        return self.session.run([self.source_error, self.classification_error], feed_dict=feed_dict)

    def train_generator_step(self, X, y):
        batch_size = X.shape[0]
        feed_dict = {}
        feed_dict[self.inputs] = X
        #noise = self.noise_generator(size=(batch_size, self.prior_dim))
        self.noise_generator.reset(X, y)
        noise, _ = self.noise_generator.generate(1, shuffle=False, include_original_data=False)
        noise = noise[:, self.block[0]:self.block[1]]
        feed_dict[self.prior_noise] = noise
        feed_dict[self.targets] = one_hot_encoding(y, self.num_classes)
        feed_dict[self.masks] = np.broadcast_to(self.mask, shape=X.shape)
        feed_dict[self.is_training] = True
        feed_dict[self.dis_l2_scale] = self.cur_dis_l2_scale
        feed_dict[self.is_gen] = 1.
        #feed_dict[self.source_targets] = np.zeros((batch_size, 1))
        feed_dict[self.source_targets] = np.ones((batch_size, 1)) # min log(1-D) has vanishing gradients, use max logD
        self.session.run(self.generator_optimizer, feed_dict=feed_dict)
        return self.session.run([self.source_error, self.classification_error], feed_dict=feed_dict)


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

    def train_epoch(self, train_inputs, train_targets, K, batch_size=100, verbose=1):
        train_inputs = train_inputs.copy()
        np.random.shuffle(train_inputs)
        start = 0
        dis_errs = []
        gen_errs = []
        cls_errs = []
        while start < train_inputs.shape[0]:
            end = min(start+batch_size, train_inputs.shape[0])
            batch = train_inputs[start:end, :]
            batch_targets = train_targets[start:end]
            for k in range(K):
                dis_e, cls_e = self.train_discriminator_step(batch, batch_targets)
            dis_errs.append(dis_e)
            gen_e, cls_e = self.train_generator_step(batch, batch_targets)
            cls_errs.append(cls_e)
            gen_errs.append(gen_e)
            start = end
        if verbose > 0:
            print "DIS>>", np.array(dis_errs).mean()
            print "GEN>>", np.array(gen_errs).mean()
            print "CLS>>", np.array(cls_errs).mean()
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

    def generate(self, X, y, reject=True, same_size=True):
        inputs = X.copy()
        targets = one_hot_encoding(y, self.num_classes)
        #noise = self.noise_generator(size=(inputs.shape[0], self.prior_dim))
        self.noise_generator.reset(X, y)
        noise, _ = self.noise_generator.generate(1, shuffle=False, include_original_data=False)
        noise = noise[:, self.block[0]:self.block[1]]
        feed_dict = {}
        feed_dict[self.inputs] = inputs
        feed_dict[self.targets] = targets
        feed_dict[self.prior_noise] = noise
        feed_dict[self.masks] = np.broadcast_to(self.mask, shape=inputs.shape)
        feed_dict[self.is_training] = False
        feed_dict[self.dis_l2_scale] = self.cur_dis_l2_scale

        return self.session.run(self.gen_inputs, feed_dict=feed_dict)
        """
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
        """

    def set_model(self, hyper_params):

        self.cur_dis_l2_scale = hyper_params['dis_l2_scale']
        self.inputs = tf.placeholder(tf.float32, [None, self.inputs_dim], 'inputs')
        self.prior_noise = tf.placeholder(tf.float32, [None, self.prior_dim], 'prior')
        self.source_targets = tf.placeholder(tf.float32, [None,  1], 'source_targets')
        self.targets = tf.placeholder(tf.float32, [None,  self.num_classes], 'targets')

        self.is_gen = tf.placeholder(tf.float32, (), "is_gen")
        self.masks = tf.placeholder(tf.float32, [None, self.inputs_dim], 'masks')
        self.is_training = tf.placeholder(tf.bool, (), "is_training")
        self.dis_l2_scale = tf.placeholder(tf.float32, (), "dis_l2_scale")


        self.gen_inputs = self._build_generator(hyper_params, self.inputs * self.mask, self.targets, self.prior_noise)
        self.dis_inputs = self.gen_inputs * self.is_gen + self.inputs * (1-self.is_gen)
        inputs = tf.concat([self.dis_inputs, 1-self.masks, self.targets], axis=1)
        #inputs = tf.concat([self.gen_inputs, self.inputs], axis=0)
        self.discriminator_optimizer, self.source_proba, self.source_error, self.classification_error = self._build_discriminator(hyper_params, inputs, self.source_targets, self.targets)
        self.generator_optimizer = self._build_generator_optimizer(hyper_params, self.source_error, self.classification_error)

        var_list_gen = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+"-generator")
        var_list_dis = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+"-discriminator")
        var_list = var_list_gen+var_list_dis
        self.init = tf.variables_initializer(var_list=var_list)



    def _build_generator(self, hyper_params, inputs, targets, prior_noise):
        num_hidden_units = hyper_params['gen_num_hidden_units']
        num_hidden_layers = hyper_params['gen_num_hidden_layers']
        if "activation" in hyper_params:
            activation = hyper_params["activation"]
        else:
            activation = tf.contrib.keras.layers.LeakyReLU()
        l2_scale = hyper_params["gen_l2_scale"]
        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_scale)
        kernel_initializer = tf.contrib.layers.xavier_initializer()

        #layer = tf.concat([inputs, targets, prior_noise], axis=1)
        layer = tf.concat([inputs, targets, prior_noise], axis=1)


        with tf.variable_scope(self.name+"-generator"):
            #layer = tf.layers.batch_normalization(layer, training=self.is_training)
            for l in range(num_hidden_layers):
                layer = tf.layers.dense(layer, num_hidden_units, activation,
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer)
                #layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = tf.layers.dense(layer, num_hidden_units*self.num_classes,
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer)
            sel = tf.reshape(tf.transpose(tf.stack([targets for i in range(self.num_hidden_units)])), shape=-1)
            layer = tf.reduce_sum(layer * sel, axis=1, keep_dims=True)

        gen_inputs = inputs*self.mask + tf.pad(layer, paddings=[[0, 0], [self.block[0], self.inputs_dim-self.block[1]]])
        return gen_inputs

    def _build_discriminator(self, hyper_params, inputs, source_targets, targets):
        num_hidden_units = hyper_params['dis_num_hidden_units']
        num_hidden_layers = hyper_params['dis_num_hidden_layers']
        if "activation" in hyper_params:
            activation = hyper_params["activation"]
        else:
            activation = tf.contrib.keras.layers.LeakyReLU()
        if "learning_rate" in hyper_params:
            learning_rate = hyper_params['learning_rate']
        else:
            learning_rate = 1e-3

        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=self.dis_l2_scale)
        kernel_initializer = tf.contrib.layers.xavier_initializer()

        layer = inputs

        with tf.variable_scope(self.name+"-discriminator"):
            #layer = tf.layers.batch_normalization(layer, training=self.is_training)
            for l in range(num_hidden_layers):
                layer = tf.layers.dense(layer, num_hidden_units, activation,
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer)
                #layer = tf.layers.batch_normalization(layer, training=self.is_training)

            source_logits = tf.layers.dense(layer, 1,
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer)
            classification_logits = tf.layers.dense(layer, self.num_classes,
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer)
            source_proba = tf.nn.sigmoid(source_logits)
            source_error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=source_targets, logits=source_logits))
            classification_error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=classification_logits))
            l2_loss = tf.losses.get_regularization_loss(scope=self.name+"-discriminator")
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+"-discriminator")
            discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(source_error+l2_loss, var_list=var_list)
            return discriminator_optimizer, source_proba, source_error, classification_error

    def _build_generator_optimizer(self, hyper_params, source_error, classification_error):
        if "learning_rate" in hyper_params:
            learning_rate = hyper_params['learning_rate']
        else:
            learning_rate = 1e-3

        with tf.variable_scope(self.name+"-generator"):
            l2_loss = tf.losses.get_regularization_loss(scope=self.name+"-generator")
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+"-generator")
            generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(source_error+l2_loss, var_list=var_list)
            return generator_optimizer

    """
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
    """
