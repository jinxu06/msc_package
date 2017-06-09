import numpy as np
import tensorflow as tf


class DependencyNetwork:

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
                    targets = tf.slice(nodes["inputs"], [0, i], [-1, 1])
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
        vals = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return self.session.run(vals)

    def _assign_all_params(self, values):
        vals = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
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

    def train(self, train_data, valid_data, num_epoch, batch_size=None, valid_freq=5, early_stopping_lookahead=None):
        errors, accs = self.validate_epoch(valid_data, batch_size)
        print "epoch {0} -- valid -- error:{1}, acc:{2}".format(0, errors.mean(), accs.mean())

        for i in range(num_epoch):
            errors, accs = self.train_epoch(train_data, batch_size)
            #print "epoch {0} -- train -- error:{1}, acc:{2}".format(i+1, errors.mean(), accs.mean())
            if (i+1)%valid_freq==0:
                print "epoch {0} -- train -- error:{1}, acc:{2}".format(i+1, errors.mean(), accs.mean())
                valid_errors, valid_accs = self.validate_epoch(valid_data, batch_size)
                if early_stopping_lookahead is not None:
                    stop = self._monitor(valid_errors.mean(), early_stopping_lookahead)
                    if stop:
                        self._assign_all_params(self.historical_params[0])
                        print "Early Stopping at {0}, Valid Err {1}".format(i+1-valid_freq*early_stopping_lookahead, self.historical_valid_error[0])
                        return self.historical_valid_error[0]
                print "epoch {0} -- valid -- error:{1}, acc:{2}".format(i+1, valid_errors.mean(), valid_accs.mean())
        return valid_errors.mean()

    def query(self, query_data):
        feed_dict = {self.graph_nodes['inputs']:query_data}
        if self.graph_config['mask_as_inputs']:
            feed_dict[self.graph_nodes['mask_inputs']] = 1-np.broadcast_to(self.masks, shape=(query_data.shape[0], self.masks.shape[0], self.masks.shape[1]))
        predictions = self.session.run(self.graph_nodes['predictions'], feed_dict=feed_dict)
        return predictions
