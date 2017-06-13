import numpy as np
from dependency_network import *
import tensorflow as tf


def train_dependency_networks(config, num_classes, train_inputs, train_targets, valid_inputs, valid_targets, inputs_block, attr_types, max_num_epoch=500, early_stopping_lookahead=5, quiet=False):

    config['mask_as_inputs'] = config['weights_sharing']
    graph = tf.Graph()
    dns = []
    for i in range(num_classes):
        dn = DependencyNetwork(inputs_block, attr_types, graph=graph, graph_config=config, name="DN{0}-{1}".format(hash(time.time()), i+1))
        dns.append(dn)
    sess = tf.Session(graph=graph)
    with graph.as_default():
        init = tf.global_variables_initializer()
    sess.run(init)

    for dn in dns:
        dn.set_session(sess)

    valid_errors = []
    for i, dn in enumerate(dns):
        e = dn.train(train_inputs[train_targets==i, :], valid_inputs[valid_targets==i, :],
                 num_epoch=max_num_epoch, batch_size=100, early_stopping_lookahead=early_stopping_lookahead, quiet=quiet)
        valid_errors.append(e)

    return dns, sess, valid_errors


def train_flexible_dependency_networks(configs, num_classes, train_inputs, train_targets, valid_inputs, valid_targets, inputs_block, attr_types, max_num_epoch=500, valid_freq=5, early_stopping_lookahead=5, quiet=False):
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    fdns = []
    for i in range(num_classes):
        fdn = FlexibleDependencyNetwork(inputs_block, attr_types, configs, graph=graph, session=sess, name="FDN{0}-{1}".format(hash(time.time()), i))
        fdns.append(fdn)

    errors = []
    for i, fdn in enumerate(fdns):
        e = fdn.train(train_inputs[train_targets==i, :], valid_inputs[valid_targets==i, :],
                max_num_epoch=max_num_epoch, valid_freq=valid_freq, early_stopping_lookahead=early_stopping_lookahead, quiet=quiet)
        errors.append(e)
    return fdns, sess, errors

def K_fold_train_flexible_dependency_networks(K, max_k, configs, num_classes, train_inputs, train_targets, inputs_block, attr_types, max_num_epoch=500, valid_freq=5, early_stopping_lookahead=5, quiet=False):
    dns_ens = []
    sess_ens = []
    err_ens = []
    for k in range(min(K, max_k)):
        valid_index = range(k,train_inputs.shape[0],K)
        t_inputs, t_targets = np.delete(train_inputs, valid_index, axis=0), np.delete(train_targets, valid_index, axis=0)
        v_inputs, v_targets = train_inputs[valid_index,:], train_targets[valid_index]
        dns, sess, valid_errors = train_flexible_dependency_networks(configs, num_classes, t_inputs, t_targets, v_inputs, v_targets, inputs_block, attr_types, max_num_epoch, valid_freq, early_stopping_lookahead, quiet)
        dns_ens.append(dns)
        sess_ens.append(sess)
        err_ens.append(valid_errors)
    return dns_ens, sess_ens, err_ens


def K_fold_train_dependency_networks(K, config, num_classes, train_inputs, train_targets, inputs_block, attr_types, max_num_epoch=500, early_stopping_lookahead=5, quiet=False):
    dns_ens = []
    sess_ens = []
    err_ens = []
    for k in range(K):
        valid_index = range(k,train_inputs.shape[0],K)
        t_inputs, t_targets = np.delete(train_inputs, valid_index, axis=0), np.delete(train_targets, valid_index, axis=0)
        v_inputs, v_targets = train_inputs[valid_index,:], train_targets[valid_index]
        dns, sess, valid_errors = train_dependency_networks(config, num_classes, t_inputs, t_targets, v_inputs, v_targets, inputs_block, attr_types, max_num_epoch, early_stopping_lookahead, quiet)
        dns_ens.append(dns)
        sess_ens.append(sess)
        err_ens.append(np.mean(valid_errors))
    return dns_ens, sess_ens, err_ens

def bagging_train_dependency_networks(n_estimators, config, num_classes, train_inputs, train_targets, inputs_block, attr_types, max_num_epoch=500, early_stopping_lookahead=5, quiet=False):
    dns_ens = []
    sess_ens = []
    err_ens = []
    for n in range(n_estimators):
        t_inputs, t_targets, v_inputs, v_targets = resample_with_replacement(train_inputs, train_targets)
        dns, sess, valid_errors = train_dependency_networks(config, num_classes, t_inputs, t_targets, v_inputs, v_targets, inputs_block, attr_types, max_num_epoch, early_stopping_lookahead, quiet)
        dns_ens.append(dns)
        sess_ens.append(sess)
        err_ens.append(np.mean(valid_errors))
    return dns_ens, sess_ens, err_ens
