import numpy as np
import itertools
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from contrib import enumerate_parameters
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
import warnings

class SklearnEstimator(object):

    def __init__(self, estimator):
        self.estimator = estimator

    def print_params(self):
        print self.estimator.get_params()

    def set_params(self, params):
        self.estimator.set_params(**params)

    def fit(self, X, y, sample_weight=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.estimator.fit(X, y, sample_weight)

    def evaluate(self, X, y, metric, average="macro"):
        if metric=='log_loss':
            return log_loss(y, self.predict_proba(X))
        metric = eval(metric)
        try:
            return metric(y, self.predict(X), average=average)
        except TypeError:
            return metric(y, self.predict(X))

    def precision_recall_report(self, X, y_true, y_pred=None):
        if y_pred is None:
            y_pred = self.predict(X)
        print classification_report(y_true, y_pred, digits=5)

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def train_valid_search(self, X, y, v_X, v_y, param_grid, sample_weight=None, v_sample_weight=None, max_num_run=None, refit=False, verbose=1):
        hyper_params = enumerate_parameters(param_grid)
        hyper_params = np.random.choice(hyper_params, size=len(hyper_params), replace=False)
        if max_num_run is None:
            max_num_run = len(hyper_params)
        best_params = None
        best_score = 0
        for r in range(max_num_run):
            self.set_params(hyper_params[r])
            self.fit(X, y, sample_weight=sample_weight)
            print hyper_params[r]
            score = self.estimator.score(v_X, v_y, sample_weight=v_sample_weight)
            print score
            if score > best_score:
                best_score = score
                best_params = hyper_params[r]
        print "best ------"
        print best_params
        print best_score
        if sample_weight is None:
            sample_weight = np.ones((X.shape[0], ))
        if v_sample_weight is None:
            v_sample_weight = np.ones((v_X.shape[0], ))
        train_data = np.concatenate([X, y[:, None], sample_weight[:, None]], axis=1)
        valid_data = np.concatenate([v_X, v_y[:, None], v_sample_weight[:, None]], axis=1)
        all_data = np.concatenate([train_data, valid_data], axis=0)
        np.random.shuffle(all_data)
        self.set_params(best_params)
        if refit:
            self.fit(all_data[:, :-2], all_data[:, -2], sample_weight=all_data[:, -1])

    def grid_search(self, X, y, param_grid, sample_weight=None, cv=3, scoring=None, n_jobs=1, verbose=1):
        if scoring =="log_loss":
            scoring = lambda e, x, y: log_loss(y, e.predict_proba(x))
        gs = GridSearchCV(self.estimator, param_grid, scoring=scoring, cv=cv, refit=True, n_jobs=n_jobs, fit_params={"sample_weight":sample_weight})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rs.fit(X, y)
        self.estimator = gs.best_estimator_
        result = gs.cv_results_
        if verbose>=2:
            print "CV {0}".format(cv)
        for params, score, std in zip(result['params'], result['mean_test_score'], result['std_test_score']):
            if verbose >= 2:
                print params
                print "mean_test_score={0}, std_test_score={1}".format(score, std)
        if verbose >=1:
            print "best params -- {0}".format(gs.best_params_)
            print "best_score -- {0}".format(gs.best_score_)

    def randomized_search(self, X, y, param_distributions, n_iter, sample_weight=None, cv=3, scoring=None, n_jobs=1, verbose=1):
        if scoring == "log_loss":
            scoring = lambda e, x, y: log_loss(y, e.predict_proba(x))
        rs = RandomizedSearchCV(self.estimator, param_distributions, n_iter=n_iter, scoring=scoring, cv=cv, refit=True, n_jobs=n_jobs, fit_params={"sample_weight":sample_weight})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rs.fit(X, y)
        self.estimator = rs.best_estimator_
        result = rs.cv_results_
        if verbose>=2:
            print "CV {0}".format(cv)
        for params, score, std in zip(result['params'], result['mean_test_score'], result['std_test_score']):
            if verbose >= 2:
                print params
                print "mean_test_score={0}, std_test_score={1}".format(score, std)
        if verbose >=1:
            print "best params -- {0}".format(rs.best_params_)
            print "best_score -- {0}".format(rs.best_score_)




class SklearnClassifier(object):

    def __init__(self, method, hyper_params, random=False, random_max_run=1):
        self.hyper_params_choices = enumerate_parameters(hyper_params)
        if random:
            num_choices = len(self.hyper_params_choices)
            assert random_max_run <= num_choices
            sel = np.random.choice(range(num_choices),
                                        size=(random_max_run,), replace=False)
            self.hyper_params_choices = [self.hyper_params_choices[s] for s in sel]
        self.method = method
        self.performance_records = []

    def build_model(self, hyper_params):
        self.model = self.method(**hyper_params)

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def experiment(self, train_inputs, train_targets, valid_inputs, valid_targets, sample_weight=None, verbose=1):
        best_acc, best_std = 0., 0.
        best_model = None
        for hyper_params in self.hyper_params_choices:
            self.build_model(hyper_params)
            self.fit(train_inputs, train_targets, sample_weight=sample_weight)
            pred = self.predict(train_inputs)
            train_acc = np.sum(pred == train_targets) / float(len(pred))
            train_std = np.sqrt(np.var(pred==train_targets, ddof=1) / len(pred))
            pred = self.predict(valid_inputs)
            valid_acc = np.sum(pred == valid_targets) / float(len(pred))
            valid_std = np.sqrt(np.var(pred==valid_targets, ddof=1) / len(pred))
            performance = {"train_acc":train_acc, "train_std":train_std, "valid_acc":valid_acc, "valid_std":valid_std}
            self.performance_records.append((hyper_params, performance))
            if valid_acc > best_acc:
                best_acc, best_std = valid_acc, valid_std
                best_model = self.model
            if verbose > 1:
                s = ""
                for param in hyper_params:
                    s += param+"="
                    s += str(hyper_params[param]) + " "
                s = type(self).__name__ + "  " + s
                print s+"--------"
                print "train acc:{0}({1}), valid acc:{2}({3})".format(train_acc, train_std, valid_acc, valid_std)

        if verbose >= 1:
            print "\n{0} -- Best Accuracy: {1}({2})\n".format(type(self).__name__, best_acc, best_std)
        return best_acc, best_std, best_model

"""


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
        return self.best_valid_acc, self.best_valid_acc_std


class LogisticRegressionClassifier(HighPerformanceClassifier):


    def __init__(self, hyper_params=None):

        self.hyper_params = {}
        self.hyper_params['C'] = [0.01, 0.1, 1., 10., 100.]
        super(LogisticRegressionClassifier, self).__init__(hyper_params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def rebuild_model(self):

        C = self.grid_search_params['C'][self.grid_search_pos]
        self.model = LogisticRegression(C=C)



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
        self.hyper_params['max_depth'] = [2, 4, 6]
        super(RandomForestClassifier, self).__init__(hyper_params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def rebuild_model(self):

        n = self.grid_search_params['n_estimators'][self.grid_search_pos]
        max_depth = self.grid_search_params['max_depth'][self.grid_search_pos]
        self.model = RFClassifier(n, max_depth=max_depth)


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

"""
