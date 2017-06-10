import numpy as np
import tensorflow as tf
from dependency_network import DependencyNetwork


def train_dependency_networks(config, num_classes, train_inputs, train_targets, valid_inputs, valid_targets, inputs_block, attr_types, max_num_epoch=500, early_stopping_lookahead=5, quiet=False):

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
    return new_data, ratio
