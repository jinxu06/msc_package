import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import MultinomialNB
import itertools
import time
from contrib import enumerate_parameters








class DependencyNetwork(object):

    def __init__(self, inputs_block, attr_types, graph_config=None, graph=None, name="DN", random_seed=123):

        self.random_seed = random_seed
        if graph is None:
            self.graph = tf.Graph()
        else:
            self.graph = graph
        config = {}
        config['num_hidden_units'] = 50
        config['num_hidden_layers'] = 1
        config['hidden_layer_nonlinearity'] = tf.nn.sigmoid
        config['l2_scale'] = 1e-4
        config['weights_sharing'] = False
        config['mask_as_inputs'] = False
        config['adam_learning_rate'] = 1e-3
        if graph_config is not None:
            for key in graph_config:
                config[key] = graph_config[key]
        self.graph_config = config

        self.attr_types = attr_types
        self.num_attr = len(self.attr_types)

        self.inputs_block = inputs_block
        self.num_inputs = self.inputs_block[-1][1]
        self.name = name
        self.graph_nodes = {}
        self._define_masks()
        with self.graph.as_default():
            self._build_graph()

        self.historical_params = []
        self.historical_valid_error = []

    def _define_masks(self):
        self.masks = []
        for block in self.inputs_block:
            mask = np.ones((self.num_inputs,), dtype=np.int64)
            mask[block[0]:block[1]] = 0
            self.masks.append(mask)
        self.masks = np.array(self.masks)

    def _build_graph(self):
        with tf.variable_scope(self.name):
            nodes = self.graph_nodes
            nodes["inputs"] = tf.placeholder(tf.float64, [None, self.masks.shape[1]], 'inputs')
            nodes["mask_inputs"] = tf.placeholder(tf.float64, [None, self.masks.shape[0], self.masks.shape[1]], 'mask_inputs')

            num_hidden_units = self.graph_config['num_hidden_units']
            num_hidden_layers = self.graph_config['num_hidden_layers']
            nonlinearity = self.graph_config['hidden_layer_nonlinearity']
            l2_scale = self.graph_config['l2_scale']
            sharing = self.graph_config['weights_sharing']
            mask_as_inputs = self.graph_config['mask_as_inputs']
            learning_rate = self.graph_config['adam_learning_rate']

            kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_scale)
            kernel_initializer = tf.contrib.layers.xavier_initializer(seed=self.random_seed)


            outputs_collection = []
            for i in range(self.masks.shape[0]):
                masked_inputs = nodes['inputs'] * self.masks[i]
                if mask_as_inputs:
                    masked_inputs = tf.concat([masked_inputs, nodes['mask_inputs'][:, i, :]], axis=1)
                outputs_collection.append(masked_inputs)

            for i in range(num_hidden_layers):
                if not sharing:
                    for k in range(self.masks.shape[0]):
                        hidden_units = outputs_collection[k]
                        hidden_units = tf.layers.dense(hidden_units, num_hidden_units, nonlinearity,
                                             kernel_initializer=kernel_initializer,
                                             kernel_regularizer=kernel_regularizer)
                        #hidden_units = tf.layers.batch_normalization(hidden_units, training=True)
                        outputs_collection[k] = hidden_units
                else:
                    reuse = None
                    for k in range(self.masks.shape[0]):
                        hidden_units = outputs_collection[k]
                        hidden_units = tf.layers.dense(hidden_units, num_hidden_units, nonlinearity,
                                             kernel_initializer=kernel_initializer,
                                             kernel_regularizer=kernel_regularizer, reuse=reuse, name="{0}-share-{1}".format(self.name, i+1))
                        #hidden_units = tf.layers.batch_normalization(hidden_units, training=True, reuse=reuse, name="{0}-bn-{1}".format(self.name, i+1))
                        reuse = True
                        outputs_collection[k] = hidden_units

            errors = []
            predictions = []
            accuracies = []
            for i, t in enumerate(self.attr_types):
                if t=='b':
                    targets = tf.slice(nodes["inputs"], [0, i], [-1, 1]) # This seems to be wrong when attr_type is mixed
                    targets = tf.concat([1-targets, targets], axis=1)
                    num_outputs = 2
                elif t=='c':
                    num_outputs = self.inputs_block[i][1] - self.inputs_block[i][0]
                    targets = tf.slice(nodes["inputs"], [0, self.inputs_block[i][0]], [-1, num_outputs])
                elif t=='r':
                    pass
                else:
                    raise Exception("attr_type not found")
                outputs_collection[i] = tf.layers.dense(outputs_collection[i], num_outputs,
                                             kernel_initializer=kernel_initializer,
                                             kernel_regularizer=kernel_regularizer)
                error = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=outputs_collection[i], labels=targets))
                errors.append(error)

                prediction = tf.nn.softmax(outputs_collection[i])
                predictions.append(prediction)

                accuracy = tf.reduce_mean(tf.cast(
                    tf.equal(tf.argmax(outputs_collection[i], 1), tf.argmax(targets, 1)),
                    tf.float64))
                accuracies.append(accuracy)

            nodes['errors'] = errors
            nodes['predictions'] = predictions
            nodes['accuracies'] = accuracies

        nodes['l2_loss'] = tf.losses.get_regularization_loss(scope=self.name)

        nodes['optimizer'] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(tf.reduce_sum(errors)+nodes['l2_loss'])




    def set_session(self, sess):
        self.session = sess

    def train_epoch(self, train_data, batch_size=None):
        if batch_size is None:
            batch_size = train_data.shape[0]
        start = 0
        errors = []
        accs = []
        while start < train_data.shape[0]:
            end = min(start+batch_size, train_data.shape[0])
            batch = train_data[start:end, :]
            feed_dict = {self.graph_nodes['inputs']:batch}
            if self.graph_config['mask_as_inputs']:
                feed_dict[self.graph_nodes['mask_inputs']] = 1-np.broadcast_to(self.masks, shape=(batch.shape[0], self.masks.shape[0], self.masks.shape[1]))
            self.session.run(self.graph_nodes['optimizer'], feed_dict=feed_dict)
            errors_batch = self.session.run(self.graph_nodes['errors'], feed_dict=feed_dict)
            accs_batch = self.session.run(self.graph_nodes['accuracies'], feed_dict=feed_dict)
            errors.append(errors_batch)
            accs.append(accs_batch)
            start = end
        return np.array(errors).mean(0), np.array(accs).mean(0)

    def validate_epoch(self, valid_data, batch_size=None):
        if batch_size is None:
            batch_size = valid_data.shape[0]
        start = 0
        errors = []
        accs = []
        while start < valid_data.shape[0]:
            end = min(start+batch_size, valid_data.shape[0])
            batch = valid_data[start:end, :]
            feed_dict = {self.graph_nodes['inputs']:batch}
            if self.graph_config['mask_as_inputs']:
                feed_dict[self.graph_nodes['mask_inputs']] = 1-np.broadcast_to(self.masks, shape=(batch.shape[0], self.masks.shape[0], self.masks.shape[1]))
            errors_batch = self.session.run(self.graph_nodes['errors'], feed_dict=feed_dict)
            accs_batch = self.session.run(self.graph_nodes['accuracies'], feed_dict=feed_dict)
            errors.append(errors_batch)
            accs.append(accs_batch)
            start = end
        return np.array(errors).mean(0), np.array(accs).mean(0)

    def _get_all_params(self):
        with self.graph.as_default():
            vals = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            return self.session.run(vals)

    def _assign_all_params(self, values):
        with self.graph.as_default():
            vals = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            with tf.variable_scope(self.name):
                opts = [val.assign(value) for val, value in zip(vals, values)]
                opt = tf.group(*opts)
                self.session.run(opt)

    def _monitor(self, valid_err, early_stopping_lookahead):
        if not len(self.historical_params) < early_stopping_lookahead+1:
            del self.historical_params[0]
            del self.historical_valid_error[0]
        self.historical_params.append(self._get_all_params())
        self.historical_valid_error.append(valid_err)
        if len(self.historical_params) == early_stopping_lookahead+1 and np.argmin(self.historical_valid_error)==0:
            return True
        return False

    def train(self, train_data, valid_data, num_epoch, batch_size=None, valid_freq=5, early_stopping_lookahead=None, quiet=False):
        errors, accs = self.validate_epoch(valid_data, batch_size)
        if not quiet:
            print "epoch {0} -- valid -- error:{1}, acc:{2}".format(0, errors.mean(), accs.mean())

        for i in range(num_epoch):
            errors, accs = self.train_epoch(train_data, batch_size)
            #print "epoch {0} -- train -- error:{1}, acc:{2}".format(i+1, errors.mean(), accs.mean())
            if (i+1)%valid_freq==0:
                if not quiet:
                    print "epoch {0} -- train -- error:{1}, acc:{2}".format(i+1, errors.mean(), accs.mean())
                valid_errors, valid_accs = self.validate_epoch(valid_data, batch_size)
                if early_stopping_lookahead is not None:
                    stop = self._monitor(valid_errors.mean(), early_stopping_lookahead)
                    if stop:
                        self._assign_all_params(self.historical_params[0])
                        print "Early Stopping at {0}, Valid Err {1}".format(i+1-valid_freq*early_stopping_lookahead, self.historical_valid_error[0])
                        return self.historical_valid_error[0]
                if not quiet:
                    print "epoch {0} -- valid -- error:{1}, acc:{2}".format(i+1, valid_errors.mean(), valid_accs.mean())
        return valid_errors.mean()

    def query(self, query_data):
        feed_dict = {self.graph_nodes['inputs']:query_data}
        if self.graph_config['mask_as_inputs']:
            feed_dict[self.graph_nodes['mask_inputs']] = 1-np.broadcast_to(self.masks, shape=(query_data.shape[0], self.masks.shape[0], self.masks.shape[1]))
        predictions = self.session.run(self.graph_nodes['predictions'], feed_dict=feed_dict)
        return predictions


class MLPClassifier(object):

    def __init__(self, config, inputs_node, mask, block, attr_type, graph, session ,scope):
        self.config = config
        self.mask = mask
        self.block = block
        self.attr_type = attr_type
        self.graph = graph
        self.session = session
        self.scope = scope
        self.graph_nodes = {}
        self.graph_nodes['inputs'] = inputs_node
        self.init = None

        self._build_graph()

        self.historical_params = []
        self.historical_valid_error = []

    def init_variables(self):
        if self.init is None:
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
            self.init = tf.variables_initializer(var_list=var_list)
        self.session.run(self.init)

    def _build_graph(self):
        nodes = self.graph_nodes
        with tf.variable_scope(self.scope):

            masked_inputs = nodes['inputs'] * self.mask

            num_hidden_units = self.config['num_hidden_units']
            nonlinearity = self.config['hidden_layer_nonlinearity']
            l2_scale = self.config['l2_scale']
            kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_scale)
            kernel_initializer = tf.contrib.layers.xavier_initializer()
            learning_rate = self.config['adam_learning_rate']
            block = self.block
            attr_type = self.attr_type

            hidden_units = masked_inputs
            for l in range(self.config['num_hidden_layers']):
                hidden_units = tf.layers.dense(hidden_units, num_hidden_units, nonlinearity,
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer)


            if self.attr_type=='b':
                targets = tf.slice(nodes['inputs'], [0, block[0]], [-1, 1])
                targets = tf.concat([1-targets, targets], axis=1)
                num_outputs = 2
            elif self.attr_type=='c':
                num_outputs = block[1] - block[0]
                targets = tf.slice(nodes['inputs'], [0, block[0]], [-1, num_outputs])
            elif self.attr_type=='r':
                pass
            else:
                raise Exception("attr_type not found")

            logits = tf.layers.dense(hidden_units, num_outputs,
                                         kernel_initializer=kernel_initializer,
                                         kernel_regularizer=kernel_regularizer)

            nodes['error'] = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets))
            nodes['prediction'] = tf.nn.softmax(logits)
            nodes['accuracy'] = tf.reduce_mean(tf.cast(
                    tf.equal(tf.argmax(logits, 1), tf.argmax(targets, 1)),
                    tf.float64))

            nodes['l2_loss'] = tf.losses.get_regularization_loss(scope=self.scope)

            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
            nodes['optimizer'] = tf.train.AdamOptimizer(learning_rate=learning_rate)\
                                .minimize(nodes['error']+nodes['l2_loss'], var_list=var_list)



    def train_epoch(self, train_inputs, batch_size=None):
        if batch_size is None:
            batch_size = 100
        start = 0
        errors = []
        accs = []
        while start < train_inputs.shape[0]:
            end = min(start+batch_size, train_inputs.shape[0])
            batch = train_inputs[start:end, :]
            feed_dict = {self.graph_nodes['inputs']:batch}
            self.session.run(self.graph_nodes['optimizer'], feed_dict=feed_dict)
            error_batch, acc_batch = self.session.run([self.graph_nodes['error'], self.graph_nodes['accuracy']], feed_dict=feed_dict)
            errors.append(error_batch)
            accs.append(acc_batch)
            start = end
        return np.array(errors).mean(), np.array(accs).mean()

    def validate_epoch(self, valid_inputs, batch_size=None):
        if batch_size is None:
            batch_size = 100
        start = 0
        errors = []
        accs = []
        while start < valid_inputs.shape[0]:
            end = min(start+batch_size, valid_inputs.shape[0])
            batch = valid_inputs[start:end, :]
            feed_dict = {self.graph_nodes['inputs']:batch}
            error_batch, acc_batch = self.session.run([self.graph_nodes['error'], self.graph_nodes['accuracy']], feed_dict=feed_dict)
            errors.append(error_batch)
            accs.append(acc_batch)
            start = end
        return np.array(errors).mean(), np.array(accs).mean()



    def _get_all_params(self):
        with self.graph.as_default():
            vals = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
            return self.session.run(vals)

    def _assign_all_params(self, values):
        with self.graph.as_default():
            vals = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
            with tf.variable_scope(self.scope):
                opts = [val.assign(value) for val, value in zip(vals, values)]
                opt = tf.group(*opts)
                self.session.run(opt)

    def _monitor(self, valid_err, early_stopping_lookahead):
        if not len(self.historical_params) < early_stopping_lookahead+1:
            del self.historical_params[0]
            del self.historical_valid_error[0]
        self.historical_params.append(self._get_all_params())
        self.historical_valid_error.append(valid_err)
        if len(self.historical_params) == early_stopping_lookahead+1 and np.argmin(self.historical_valid_error)==0:
            return True
        return False




    def fit(self, max_num_epoch, train_inputs, valid_inputs, batch_size, valid_freq=5, early_stopping_lookahead=5, quiet=False):
        err, acc = self.validate_epoch(valid_inputs, batch_size)
        if not quiet:
            print "epoch {0} -- valid -- error:{1}, acc:{2}".format(0, err, acc)

        for i in range(max_num_epoch):
            err, acc = self.train_epoch(train_inputs, batch_size)
            #print "epoch {0} -- train -- error:{1}, acc:{2}".format(i+1, errors.mean(), accs.mean())
            if (i+1)%valid_freq==0:
                if not quiet:
                    print "epoch {0} -- train -- error:{1}, acc:{2}".format(i+1, err, acc)
                valid_error, valid_acc = self.validate_epoch(valid_inputs, batch_size)
                if early_stopping_lookahead is not None:
                    stop = self._monitor(valid_error, early_stopping_lookahead)
                    if stop:
                        self._assign_all_params(self.historical_params[0])
                        print "Early Stopping at {0}, Valid Err {1}".format(i+1-valid_freq*early_stopping_lookahead, self.historical_valid_error[0])
                        return self.historical_valid_error[0]
                if not quiet:
                    print "epoch {0} -- valid -- error:{1}, acc:{2}".format(i+1, valid_error, valid_acc)
        return valid_error

    def query(self, query_inputs):

        feed_dict = {self.graph_nodes['inputs']:query_inputs}
        prediction = self.session.run(self.graph_nodes['prediction'], feed_dict=feed_dict)
        return prediction


class FlexibleDependencyNetwork(DependencyNetwork):

    def __init__(self, inputs_block, attr_types, configs, graph=None, session=None, name="FDN{0}".format(hash(time.time()))):
        if graph is None:
            self.graph = tf.Graph()
        else:
            self.graph = graph
        if session is None:
            self.session = tf.Session(graph=self.graph)
        else:
            self.session = session
        self.name = name
        self.attr_types = attr_types
        self.num_attr = len(self.attr_types)
        self.inputs_block = inputs_block
        self.num_inputs = self.inputs_block[-1][1]

        self._define_masks()

        self.configs = configs
        self._build_graph(self.configs)



    def _build_graph(self, attr_configs):
        self.models = []
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.float64, [None, self.masks.shape[1]], 'inputs')

            for i in range(self.num_attr):
                attr_type = self.attr_types[i]
                block = self.inputs_block[i]
                scope = "{0}-{1}".format(self.name, i)
                config = attr_configs[i]
                mask = self.masks[i]

                model = MLPClassifier(config, self.inputs, mask, block, attr_type, self.graph, self.session ,scope)
                model.init_variables()
                self.models.append(model)

    def train(self, train_inputs, valid_inputs, max_num_epoch, batch_size=None, valid_freq=5, early_stopping_lookahead=5, quiet=False):

        errors = []
        for model in self.models:
            err = model.fit(max_num_epoch, train_inputs, valid_inputs,
                        valid_freq=valid_freq, batch_size=batch_size,
                        early_stopping_lookahead=early_stopping_lookahead, quiet=quiet)
            errors.append(err)
        return errors


    def query(self, query_inputs):
        return [model.query(query_inputs) for model in self.models]







class SklearnDependencyNetwork(DependencyNetwork):

    def __init__(self, method, hyper_params, inputs_block, attr_types, name="DN", random_seed=123):
        assert 'r' not in attr_types, "real value attributes are not allowd here"
        self.attr_types = attr_types
        self.num_attr = len(self.attr_types)
        self.inputs_block = inputs_block
        self.method = method
        self.num_inputs = self.inputs_block[-1][1]
        self.name = name
        self.models = []
        for i, block in enumerate(self.inputs_block):
            self.models.append(self.method())
            try:
                self.models[-1].set_params(probability=True)
            except:
                pass

        self._list_hyper_params(hyper_params)
        self._define_masks()

    def _list_hyper_params(self, hyper_params):
        all_params = []
        arr = []
        for param in hyper_params:
            arr.append(hyper_params[param])
        for item in itertools.product(*arr):
            params = {}
            for i, param in enumerate(hyper_params):
                params[param] = item[i]
            all_params.append(params)

        self.hyper_params = all_params


    def _predict_proba(self, idx, train_inputs):
        min_prob = 1e-5 / train_inputs.shape[0]
        pred = self.models[idx].predict_proba(train_inputs)
        pred = np.maximum(pred, min_prob)
        classes = self.models[idx].classes_
        if self.attr_types[idx]=='c':
            num_classes = self.inputs_block[idx][1]-self.inputs_block[idx][0]
        elif self.attr_types[idx]=='b':
            num_classes = 2
        predictions = np.zeros((train_inputs.shape[0], num_classes))
        for c in range(num_classes):
            if c in classes:
                idx = list(classes).index(c)
                predictions[:, c:c+1] = pred[:, idx:idx+1]
            else:
                predictions[:, c:c+1] = np.ones((pred.shape[0],1)) * min_prob
        predictions = predictions.astype(np.float64)
        predictions /= np.sum(predictions, axis=1)[:, None]
        return predictions

    def _train(self, train_data, batch_size=None):
        errors = []
        accs = []
        for i, block in enumerate(self.inputs_block):
            mask = self.masks[i]
            train_inputs = train_data * mask
            if self.attr_types[i]=='c':
                train_targets = np.argmax(train_data[:, block[0]:block[1]], axis=1)
            elif self.attr_types[i]=='b':
                train_targets = train_data[:, block[0]]
            self.models[i].fit(train_inputs, train_targets)
            proba = self._predict_proba(i, train_inputs)
            pred = self.models[i].predict(train_inputs)
            error = 0.
            for p, t in zip(proba, train_targets):
                error += -np.log(p[t])
            error /= train_inputs.shape[0]
            acc = np.sum(pred==train_targets) / float(len(pred))
            errors.append(error)
            accs.append(acc)
        return errors, accs


    def _validate(self, valid_data):
        errors = []
        accs = []
        for i, block in enumerate(self.inputs_block):
            mask = self.masks[i]
            valid_inputs = valid_data * mask
            if self.attr_types[i]=='c':
                valid_targets = np.argmax(valid_data[:, block[0]:block[1]], axis=1)
            elif self.attr_types[i]=='b':
                valid_targets = valid_data[:, block[0]]
            proba = self._predict_proba(i, valid_inputs)
            pred = self.models[i].predict(valid_inputs)
            error = 0.
            for p, t in zip(proba, valid_targets):
                error += -np.log(p[t])
            error /= valid_inputs.shape[0]
            acc = np.sum(pred==valid_targets) / float(len(pred))
            errors.append(error)
            accs.append(acc)
        return errors, accs

    def train(self, train_data, valid_data, quiet=True):
        valid_errors = []
        valid_accs = []
        params_choices = []
        for params in self.hyper_params:
            for model in self.models:
                model.set_params(**params)
            self._train(train_data)
            errors, accs = self._validate(valid_data)
            valid_errors.append(errors)
            valid_accs.append(accs)

        for i in np.argmin(valid_errors, axis=0):
            params_choices.append(self.hyper_params[i])

        for i, model in enumerate(self.models):
            model.set_params(**params_choices[i])

        self._train(train_data)
        errors, accs = self._validate(valid_data)
        print "valid -- error:{0}, acc:{1}".format(np.mean(errors), np.mean(accs))
        return errors, accs

    def query(self, query_data):
        predictions = []
        for i, model in enumerate(self.models):
            mask = self.masks[i]
            query_inputs = query_data * mask
            predictions.append(self._predict_proba(i, query_inputs))
        return predictions
