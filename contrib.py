import numpy as np
import tensorflow as tf
from dependency_network import DependencyNetwork
import time

def train_dependency_networks(config, num_classes, train_inputs, train_targets, valid_inputs, valid_targets, inputs_block, attr_types, max_num_epoch=500, early_stopping_lookahead=5, quiet=False):

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

def compare_datasets(inputs, targets, n_inputs, n_targets):
    data = np.concatenate([inputs, targets[:,None], np.zeros((inputs.shape[0],1), dtype=np.int64)], axis=1)
    n_data = np.concatenate([n_inputs, n_targets[:,None], np.ones((n_inputs.shape[0],1), dtype=np.int64)], axis=1)
    all_data = np.concatenate([data, n_data], axis=0)
    idx = np.lexsort(np.fliplr(all_data).T)
    all_data = all_data[idx, :]
    last_old = np.ones((all_data.shape[1],), dtype=np.int64)*2
    new_data = []
    for d in all_data:
        is_new = d[-1]==1
        if not is_new:
            last_old = d
        else:
            if not (d[:-1]==last_old[:-1]).all():
                new_data.append(d)
    new_data = np.array(new_data)
    ratio = new_data.shape[0] / float(n_inputs.shape[0])
    return ratio, new_data[:,:-2], new_data[:,-2]


def resample_with_replacement(inputs, targets):
    idx = np.random.choice(np.arange(inputs.shape[0]), size=(inputs.shape[0],))
    new_inputs = inputs[idx, :]
    new_targets = targets[idx]
    remain_inputs, remain_targets = compare_datasets(new_inputs, new_targets, inputs, targets)[1:]
    return new_inputs, new_targets, remain_inputs, remain_targets


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


def ensemble_query(dns_ens):
    def ret_func(data, c):
        results = []
        for dns in dns_ens:
            results.append(dns[c].query(data))
        mean_results = []
        for i in range(len(results[0])):
            mean_results.append(np.stack([r[i] for r in results]).mean(0))
        return mean_results
    return ret_func
