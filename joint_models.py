import numpy as np
from contrib import enumerate_parameters
from sklearn.neural_network import BernoulliRBM
from sklearn.mixture import GaussianMixture

class JointModel(object):

    def __init__(self, hyper_params, random=True, name=None):
        self.model = None
        self.name = name
        self.hyper_params_choices = enumerate_parameters(hyper_params)
        if random:
            num_choices = len(self.hyper_params_choices)
            sel = np.random.choice(range(num_choices),
                                        size=(num_choices,), replace=False)
            self.hyper_params_choices = [self.hyper_params_choices[s] for s in sel]

    def set_params(self, params):
        pass

    def fit(self, X):
        return self.model.fit(X)

    def evaluate(self, X):
        pass

    def train_K_fold(self, K, X, K_max_run=None, verbose=1):
        if K_max_run is None:
            K_max_run = K
        scores = []
        for k in range(min(K, K_max_run)):
            valid_index = range(k, X.shape[0],K)
            t_inputs = np.delete(X, valid_index, axis=0)
            v_inputs = X[valid_index,:]
            self.fit(t_inputs)
            s = self.evaluate(v_inputs)
            scores.append(s)
        return np.mean(scores)


    def search_hyper_params(self, K, train_inputs, K_max_run=None, random_max_run=None, verbose=1):
        if random_max_run is None:
            random_max_run = len(self.hyper_params_choices)
        best_hyper_params = None
        best_score = -1e10
        for i in range(min(random_max_run, len(self.hyper_params_choices))):
            hyper_params = self.hyper_params_choices[i]
            self.set_params(hyper_params)
            s = self.train_K_fold(K, train_inputs, K_max_run=K_max_run, verbose=verbose)
            if verbose >= 2:
                print hyper_params, s
            if s > best_score:
                best_score = s
                best_hyper_params = hyper_params
        self.set_params(best_hyper_params)
        self.fit(train_inputs)
        return best_hyper_params, best_score

    def save_model(self):
        with open("../models/{0}.pkl".format(self.name), 'w') as f:
            pkl.dump(self.model, f)


    def load_model(self):
        with open("../models/{0}.pkl".format(self.name), 'r') as f:
            self.model = pkl.load(f)

    def delete_model(self):
        os.remove("../models/{0}.pkl".format(self.name))


class RBMModel(JointModel):

    def __init__(self, hyper_params, random=True):

        super(RBMModel, self).__init__(hyper_params, random)

        self.model = BernoulliRBM()

    def set_params(self, params):
        self.model = BernoulliRBM(**params)

    def evaluate(self, X):
        return self.model.score_samples(X)
