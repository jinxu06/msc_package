import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
import time
from contrib import enumerate_parameters


class ConditionalModel(object):

    def __init__(self, attr_type):
        pass

    def fit(self, X, y):
        pass

    def query_proba(self, X):
        pass

class SklearnConditionalModel(ConditionalModel):

    def __init__(self, method, hyper_params, num_classes, random=False):
        self.method = method
        self.num_classes = num_classes

        self.hyper_params_choices = enumerate_parameters(hyper_params)
        if random:
            num_choices = len(self.hyper_params_choices)
            sel = np.random.choice(range(num_choices),
                                        size=(num_choices,), replace=False)
            self.hyper_params_choices = [self.hyper_params_choices[s] for s in sel]

    def set_model(self, hyper_params):
        if self.method==xgb.XGBClassifier:
            hyper_params['num_class'] = self.num_classes
        self.model = self.method(**hyper_params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        proba = self.query_proba(X)
        error = []
        for p, t in zip(proba, y):
            error.append(-np.log(p[t]))
        return np.mean(error), np.std(error, ddof=1) / len(error)

    def query_proba(self, X):
        min_proba = 1e-8
        proba = self.model.predict_proba(X)
        proba = np.maximum(proba, min_proba)
        classes = self.model.classes_
        predictions = np.zeros((X.shape[0], self.num_classes))
        for c in range(self.num_classes):
            if c in classes:
                idx = list(classes).index(c)
                predictions[:, c:c+1] = proba[:, idx:idx+1]
            else:
                predictions[:, c:c+1] = np.ones((X.shape[0],1)) * min_proba
        predictions = predictions.astype(np.float64)
        predictions /= np.sum(predictions, axis=1)[:, None]
        return predictions

    def train_K_fold(self, K, X, y, K_max_run=None):
        if K_max_run is None:
            K_max_run = K
        errs = []
        for k in range(min(K, K_max_run)):
            valid_index = range(k, X.shape[0],K)
            t_inputs = np.delete(X, valid_index, axis=0)
            t_targets = np.delete(y, valid_index, axis=0)
            v_inputs = X[valid_index,:]
            v_targets = y[valid_index]
            self.fit(t_inputs, t_targets)
            e, _ = self.evaluate(v_inputs, v_targets)
            errs.append(e)
        return np.mean(errs)

    def search_hyper_params(self, K, train_inputs, train_targets, K_max_run=None, random_max_run=None):
        if random_max_run is None:
            random_max_run = len(self.hyper_params_choices)
        best_hyper_params = None
        best_err = 1e10
        for i in range(random_max_run):
            hyper_params = self.hyper_params_choices[i]
            self.set_model(hyper_params)
            e = self.train_K_fold(K, train_inputs, train_targets, K_max_run=K_max_run)
            if e < best_err:
                best_err = e
                best_hyper_params = hyper_params
        self.set_model(best_hyper_params)
        self.fit(train_inputs, train_targets)
        return best_hyper_params, best_err



class NDependencyNetwork(object):

    def __init__(self, model_settings, inputs_block, attr_types, name=None):
        self.attr_types = attr_types
        self.num_attr = len(self.attr_types)
        self.inputs_block = inputs_block
        self.num_inputs = self.inputs_block[-1][1]
        self.name = name

        self.model_settings = model_settings
        self.hyper_params_choices = []
        self.methods = []
        for a in attr_types:
            self.methods.append(model_settings[a])
            self.hyper_params_choices.append(model_settings[model_settings[a]])

        self._define_masks()

    def _define_masks(self):
        self.masks = []
        for block in self.inputs_block:
            mask = np.ones((self.num_inputs,), dtype=np.int64)
            mask[block[0]:block[1]] = 0
            self.masks.append(mask)
        self.masks = np.array(self.masks)

    def set_models(self, methods, hyper_params):
        self.models = []
        for m, p in zip(methods, hyper_params):
            self.models.append(m(**p))

    def train(self, train_inputs, K=10, K_max_run=None, verbose=1):
        self.models = []
        for mask, block, attr_type, method, hyper_params in zip(self.masks, self.inputs_block, self.attr_types,
                        self.methods, self.hyper_params_choices):
            inputs = train_inputs * mask
            targets = train_inputs[:, block[0]:block[1]]
            if attr_type=='c':
                targets = np.argmax(targets, axis=1)
                num_classes = block[1]-block[0]
            elif attr_type=='b':
                targets = targets[:, 0]
                num_classes = 2
            
            model = SklearnConditionalModel(method, hyper_params, num_classes, random=False)
            p, e = model.search_hyper_params(K, inputs, targets, K_max_run=K_max_run)
            print p, e
            self.models.append(model)

    def query(self, query_inputs):
        ret = []
        for model, mask in zip(self.models, self.masks):
            proba = model.query_proba(query_inputs * mask)
            ret.append(proba)
        return ret
