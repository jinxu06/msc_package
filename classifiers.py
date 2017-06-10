import numpy as np
import itertools
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFClassifier
from sklearn.neural_network import MLPClassifier

class HighPerformanceClassifier(object):

    def __init__(self, hyper_params=None):
        if self.hyper_params is None:
            self.hyper_params = {}
        if hyper_params is not None:
            for param in hyper_params:
                self.hyper_params[param] = hyper_params[param]
        self.best_valid_acc = 0.
        self.best_valid_acc_std = 0.

        self.grid_search_params = {}
        all_lists = []
        for param in self.hyper_params:
            all_lists.append(self.hyper_params[param])
            self.grid_search_params[param] = []
        all_lists = list(itertools.product(*all_lists))
        for values in all_lists:
            for i, param in enumerate(self.hyper_params):
                self.grid_search_params[param].append(values[i])
        self.grid_search_pos = -1

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def grid_search(self):
        self.grid_search_pos += 1
        key = self.grid_search_params.keys()[0]
        if self.grid_search_pos < len(self.grid_search_params[key]):
            return True
        return False


    def rebuild_model(self):
        pass

    def experiment(self, train_inputs, train_targets, valid_inputs, valid_targets, hyper_params_grid=None, quiet=False):

        while self.grid_search():
            self.rebuild_model()
            self.fit(train_inputs, train_targets)
            pred = self.predict(train_inputs)
            train_acc = np.sum(pred == train_targets) / float(len(pred))
            train_std = np.sqrt(np.var(pred==train_targets, ddof=1) / len(pred))
            pred = self.predict(valid_inputs)
            valid_acc = np.sum(pred == valid_targets) / float(len(pred))
            valid_std = np.sqrt(np.var(pred==valid_targets, ddof=1) / len(pred))
            if self.best_valid_acc < valid_acc:
                self.best_valid_acc = valid_acc
                self.best_valid_acc_std = valid_std
            if not quiet:
                s = ""
                for param in self.hyper_params:
                    s += param+"="
                    s += str(self.grid_search_params[param][self.grid_search_pos]) + " "
                    s = type(self).__name__ + "  " + s
                    print s+"--------"
                    print "train acc:{0}({1}), valid acc:{2}({3})".format(train_acc, train_std, valid_acc, valid_std)
        print "\n{0} -- Best Accuracy: {1}({2})\n".format(type(self).__name__, self.best_valid_acc, self.best_valid_acc_std)




class SVMClassifier(HighPerformanceClassifier):


    def __init__(self, hyper_params=None):

        self.hyper_params = {}
        self.hyper_params['C'] = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        super(SVMClassifier, self).__init__(hyper_params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def rebuild_model(self):

        C = self.grid_search_params['C'][self.grid_search_pos]
        self.model = SVC(C=C)

class RandomForestClassifier(HighPerformanceClassifier):

    def __init__(self, hyper_params=None):

        self.hyper_params = {}
        self.hyper_params['n_estimators'] = [1, 2, 4, 8, 16, 32]
        super(RandomForestClassifier, self).__init__(hyper_params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def rebuild_model(self):

        n = self.grid_search_params['n_estimators'][self.grid_search_pos]
        self.model = RFClassifier(n)


class XGBoostClassifier(HighPerformanceClassifier):

    def __init__(self, hyper_params=None):

        self.hyper_params = {}
        self.hyper_params['max_depth'] = [2, 3, 4]
        self.hyper_params['n_estimators'] = [1, 2, 4, 8, 16, 32]
        super(XGBoostClassifier, self).__init__(hyper_params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def rebuild_model(self):
        max_depth = self.grid_search_params['max_depth'][self.grid_search_pos]
        n_estimators = self.grid_search_params['n_estimators'][self.grid_search_pos]
        self.model = xgb.XGBClassifier(max_depth=max_depth, learning_rate=0.1, n_estimators=n_estimators)


class NeuralNetworkClassifier(HighPerformanceClassifier):

    def __init__(self, hyper_params=None):

        self.hyper_params = {}
        self.hyper_params['num_hidden_layers'] = [1, 2, 3]
        self.hyper_params['num_hidden_units'] = [20, 50, 100, 200]
        self.hyper_params['alpha'] = [0.0001, 0.001, 0.01]
        super(NeuralNetworkClassifier, self).__init__(hyper_params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def rebuild_model(self):
        num_hidden_layers = self.grid_search_params['num_hidden_layers'][self.grid_search_pos]
        num_hidden_units = self.grid_search_params['num_hidden_units'][self.grid_search_pos]
        hidden_layer_sizes = [num_hidden_units for i in range(num_hidden_layers)]
        alpha = self.grid_search_params['alpha'][self.grid_search_pos]
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, learning_rate='adaptive')
