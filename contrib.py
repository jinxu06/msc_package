import numpy as np
import tensorflow as tf
import time
import itertools



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




class QueryFuncEnsemble(object):

    def __init__(self, dns_ens, class_index):
        self.dns_ens = dns_ens
        self.class_index = class_index

    def query(self, inputs):
        results = []
        for dns in self.dns_ens:
            results.append(dns[self.class_index].query(inputs))
        mean_results = []
        for i in range(len(results[0])):
            mean_results.append(np.stack([r[i] for r in results]).mean(0))
        return mean_results


def ensemble_query(dns_ens):
    num_classes = len(dns_ens[0])
    return [QueryFuncEnsemble(dns_ens, c).query for c in range(num_classes)]




def enumerate_parameters(params_choices, key_order=None):
    if key_order is None:
        keys = list(params_choices.keys())
    else:
        keys = key_order
    arr = []
    for key in keys:
        arr.append(params_choices[key])
    all_choices = []
    for params in list(itertools.product(*arr)):
        dic = {}
        for i in range(len(keys)):
            dic[keys[i]] = params[i]
        all_choices.append(dic)
    return all_choices
