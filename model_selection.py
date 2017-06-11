import numpy as np
import scipy
import copy

class ModelSelection(object):

    def __init__(self, hyper_params_grid):
        self.hyper_params_grid = hyper_params_grid
        self.performance_records = []
        self.without_replacement = True
        self.num_settings = 1
        for param in self.hyper_params_grid:
            if not isinstance(self.hyper_params_grid[param], list):
                self.without_replacement = False
            else:
                self.num_settings *= len(self.hyper_params_grid[param])

        self.cur_hyper_params = {}

    def sample_hyper_params(self):
        self.cur_hyper_params = {}
        while True:
            for param in self.hyper_params_grid:
                idx = np.random.randint(len(self.hyper_params_grid[param]))
                self.cur_hyper_params[param] = self.hyper_params_grid[param][idx]
            if self.without_replacement:
                exist = False
                for item in self.performance_records:
                    if item[0]==self.cur_hyper_params:
                        exist = True
                if not exist:
                    return copy.copy(self.cur_hyper_params)
                if len(self.performance_records)>=self.num_settings:
                    return None

    def record_performance(self, value):
        self.performance_records.append([copy.copy(self.cur_hyper_params), value])

    def find_best_hyper_params(self, ascending=True):
            return sorted(self.performance_records, key=lambda x: x[1], reverse=not ascending)

    def select_model(self, num_sampling, hyper_params_to_performance):
        pass
