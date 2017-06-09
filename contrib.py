import numpy as np
import tensorflow as tf
from dependency_network import DependencyNetwork


def train_dependency_networks(config, num_classes, train_inputs, train_targets, valid_inputs, valid_targets, inputs_block, attr_types, max_num_epoch=500, early_stopping_lookahead=5):

    graph = tf.Graph()
    dns = []
    for i in range(num_classes):
        dn = DependencyNetwork(inputs_block, attr_types, graph=graph, graph_config=config, name="DN{0}".format(i+1))
        dns.append(dn)
    sess = tf.InteractiveSession(graph=graph)
    with graph.as_default():
        init = tf.global_variables_initializer()
    sess.run(init)

    for dn in dns:
        dn.set_session(sess)

    valid_errors = []
    for i, dn in enumerate(dns):
        e = dn.train(train_inputs[train_targets==i, :], valid_inputs[valid_targets==i, :],
                 num_epoch=max_num_epoch, batch_size=100, early_stopping_lookahead=early_stopping_lookahead)
        valid_errors.append(e)

    return dns, sess, valid_errors
