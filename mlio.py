import numpy as np
import cPickle as pkl

def load_dataset(name, which):
    with open("../data/{0}-{1}.pkl".format(name, which), 'r') as f:
        data = pkl.load(f)
        inputs = data['inputs'].todense()
        targets = data['targets']
    with open("../data/{0}-{1}.pkl".format(name, "meta"), 'r') as f:
        meta = pkl.load(f)
    meta['inputs'] = np.array(inputs)
    meta['targets'] = targets
    meta['which'] = which
    meta['size'] = inputs.shape[0]
    del meta['train_size'], meta['valid_size'], meta['test_size']
    rng = np.random.RandomState(123)
    d = rng.permutation(np.concatenate([meta['inputs'], meta['targets'][:,None]], axis=1))
    meta['inputs'], meta['targets'] = d[:, :-1], d[:, -1]
    return meta
