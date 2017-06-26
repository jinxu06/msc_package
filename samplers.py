import numpy as np
import tensorflow as tf

class Sampler(object):

    def __init__(self, initial_samples, inputs_block, attr_types):
        self.initial_samples = initial_samples
        self.inputs_block = inputs_block
        self.attr_types = attr_types
        if self.initial_samples is not None:
            self.cur_samples = self.initial_samples.copy()

    def draw_samples_for_one_round(self):
        pass

    def reset_initial_samples(self, initial_samples=None):
        if initial_samples is not None:
            self.initial_samples = initial_samples
        self.cur_samples = self.initial_samples.copy()

    def run_sampling(self, num_round, mix_and_shuffle=True):
        all_samples = []
        for i in range(num_round):
            samples = self.draw_samples_for_one_round()
            all_samples.append(samples)
        all_samples = np.array(all_samples)
        if mix_and_shuffle:
            all_samples = np.reshape(all_samples,
                    (np.size(all_samples)/all_samples.shape[-1], all_samples.shape[-1]))
            np.random.shuffle(all_samples)
        return all_samples

class BlockGibbsSampler(Sampler):

    def __init__(self, initial_samples, inputs_block, attr_types,
                                        cond_func, sampling_order=None):
        assert len(inputs_block)==len(attr_types), \
                        "lengths of inputs_block and attr_types don't agree"
        if sampling_order is None:
            sampling_order = range(len(inputs_block))
        self.sampling_order = sampling_order
        self.cond_func = cond_func
        super(BlockGibbsSampler, self).__init__(initial_samples,
                                                inputs_block, attr_types)


    def draw_samples_for_one_step(self, last_step_samples, cond_prob, block, attr_type):

        samples = last_step_samples.copy()
        if attr_type == 'b':
            arr = []
            for p in cond_prob:
                s = np.random.multinomial(1, pvals=p)
                arr.append(s)
            samples[:, block[0]:block[1]] = np.array(arr)[:, 1:]
        elif attr_type == 'c':
            arr = []
            for p in cond_prob:
                s = np.random.multinomial(1, pvals=p)
                arr.append(s)
            samples[:, block[0]:block[1]] = np.array(arr)
        elif attr_type == 'r':
            arr = []
            for mu, sigma, alpha in zip(list(cond_prob[0]), list(cond_prob[1]), list(cond_prob[2])):
                idx = np.argmax(np.random.multinomial(1, pvals=alpha))
                s = np.random.normal(loc=mu[idx], scale=sigma[idx])
                arr.append(s)
            samples[:, block[0]:block[1]] = np.array(arr)[:, None]
        elif attr_type == 'i':
            attr = []
            for lam, alpha in zip(list(cond_prob[0]), list(cond_prob[1])):
                idx = np.argmax(np.random.multinomial(1, pvals=alpha))
                s = np.random.poisson(lam=lam[idx])
                arr.append(s)
            samples[:, block[0]:block[1]] = np.array(arr)[:, None]
        else:
            raise Exception("attr_type not found")

        return samples

    def draw_samples_for_one_round(self):
        #cond_probs = self.cond_func(self.cur_samples)
        all_samples = []
        for i, o in enumerate(self.sampling_order):
            block = self.inputs_block[o]
            attr_type = self.attr_types[o]
            cond_probs = self.cond_func(self.cur_samples)
            cond_prob = cond_probs[o]
            self.cur_samples = self.draw_samples_for_one_step(self.cur_samples, cond_prob, block, attr_type)
            all_samples.append(self.cur_samples.copy())
            # self.cur_samples = self.initial_samples.copy() ## code testing
        #return np.array(all_samples)
        return all_samples[-1] # Only keep the samples at the end of one round


"""
class BlockGibbsSampler:

    def __init__(self, init, inputs_block, cond_func, sampling_order=None):

        self.inputs_block = inputs_block
        if sampling_order is not None:
            self.sampling_order = sampling_order
        else:
            self.sampling_order = range(len(inputs_block))

        self.cond_func = cond_func
        self.initial_samples = init

    def draw_samples(self, init, cond_prob, block):
        samples = init.copy()
        arr = []
        for c in cond_prob:
            s = np.random.multinomial(1, pvals=c)
            arr.append(s)
        if block[1]-block[0]==1 and np.array(arr).shape[1]==2:
            samples[:, block[0]:block[1]] = np.array(arr)[:,1:]
        else:
            samples[:, block[0]:block[1]] = np.array(arr)
        return samples

    def draw_samples_for_one_round(self, init):
        all_samples = []
        samples = init.copy()
        for o in self.sampling_order:
            block = self.inputs_block[o]
            cond_prob = self.cond_func(samples)[o]
            samples = self.draw_samples(samples, cond_prob, block)
            all_samples.append(samples)
        return np.array(all_samples)

    def run_sampling(self, num_round, init=None, mix_up=False):
        if init is None:
            init = self.initial_samples
        all_samples = []
        for i in range(num_round):
            all_samples.append(self.draw_samples_for_one_round(init))
            init = all_samples[-1][-1]
        all_samples = np.array(all_samples)
        if mix_up:
            all_samples = np.concatenate(all_samples, axis=0)
            all_samples = np.concatenate(all_samples, axis=0)
            np.random.shuffle(all_samples)
        return all_samples


"""

class MUNGESampler(Sampler):

    def __init__(self, initial_samples, inputs_block, attr_types, swap_prob, s):
        self.swap_prob = swap_prob
        self.s = s
        super(MUNGESampler, self).__init__(initial_samples,
                                                inputs_block, attr_types)
        if initial_samples is not None:
            self.initial_distance = self._cal_distance(initial_samples)
            self.cur_distance = self.initial_distance.copy()

    def _cal_distance(self, data):
        size = data.shape[0]
        distance = np.zeros((size, size))
        for i in range(size):
            for j in range(i):
                distance[i, j] = self._distance(data[i], data[j])
                distance[j, i] = distance[i, j]
        return distance

    def _distance(self, x, y):
        c_dis = 0.
        r_dis = 0.
        for i in range(len(self.attr_types)):
            begin = self.inputs_block[i][0]
            end = self.inputs_block[i][1]
            if self.attr_types[i]=='c' or self.attr_types[i]=='b':
                if not np.all(x[begin:end] == y[begin:end]):
                    c_dis += 1
            elif self.attr_types[i]=='r':
                r_dis += (x[begin]-y[begin])**2
            else:
                raise Exception("attr_type not found")

        return c_dis + np.sqrt(r_dis)
        # the distance formulas should be further confirmed

    def _update_distance(self, var_index):
        size = self.cur_samples.shape[0]
        for idx in var_index:
            for i in range(size):
                self.cur_distance[idx, i] = self._distance(self.cur_samples[idx], self.cur_samples[i])
                self.cur_distance[i, idx] = self.cur_distance[idx, i]

    def reset_initial_samples(self, initial_samples=None):
        super(MUNGESampler, self).reset_initial_samples(initial_samples)
        self.initial_distance = self._cal_distance(self.initial_samples)
        self.cur_distance = self.initial_distance.copy()

    def draw_samples_for_one_round(self, reset=True):

        for i in range(self.cur_samples.shape[0]):
            neighbors = np.argmin(np.ma.masked_equal(self.cur_distance, 0.), axis=0)
            for idx, a in enumerate(self.attr_types):
                begin = self.inputs_block[idx][0]
                end = self.inputs_block[idx][1]
                if a=='c' or a=='b':
                    if np.random.rand() < self.swap_prob:
                        if (self.cur_samples[i][begin:end]!=self.cur_samples[neighbors[i]][begin:end]).all():
                            print self.cur_samples[i][begin:end]
                            temp = self.cur_samples[i][begin:end].copy()
                            self.cur_samples[i][begin:end] = self.cur_samples[neighbors[i]][begin:end].copy()
                            self.cur_samples[neighbors[i]][begin:end] = temp
                elif a=='r':
                    pass
                else:
                    raise Exception("attr_type not found")
            self._update_distance([i, neighbors[i]])
        ret_samples = self.cur_samples.copy()
        if reset:
            self.reset_initial_samples()
        return ret_samples



"""

class MUNGE:

    def __init__(self, p, s, inputs_block, attr_types=None):
        self.p = p
        self.s = s
        if attr_types is None:
            attr_types = ['c' for i in range(len(inputs_block))]
        self.attr_types = attr_types
        self.inputs_block = inputs_block

    def feed_initial_data(self, train_data):
        self.initial_data = train_data
        self.data = train_data.copy()
        self.initial_distance = self._cal_distance(self.initial_data)
        self.distance = self.initial_distance.copy()

    def _cal_distance(self, data):
        size = data.shape[0]
        distance = np.zeros((size, size))
        for i in range(size):
            for j in range(i):
                distance[i, j] = self._distance(data[i], data[j])
                distance[j, i] = distance[i, j]
        return distance

    def _distance(self, x, y):
        c_dis = 0.
        r_dis = 0.
        for i in range(len(self.attr_types)):
            begin = self.inputs_block[i][0]
            end = self.inputs_block[i][1]
            if self.attr_types[i]=='c':
                if not np.all(x[begin:end] == y[begin:end]):
                    c_dis += 1
            elif self.attr_types[i]=='r':
                r_dis += (x[begin]-y[begin])**2
            else:
                raise Exception("attr_type not found")

        return c_dis + np.sqrt(r_dis)
        # the distance formulas should be further confirmed

    def _update_distance(self, var_index):
        size = self.data.shape[0]
        for idx in var_index:
            for i in range(size):
                self.distance[idx, i] = self._distance(self.data[idx], self.data[i])
                self.distance[i, idx] = self.distance[idx, i]

    def run(self, num_round):
        all_samples = []
        assert self.data is not None, "need to feed data first"
        for k in range(num_round):
            for i in range(self.data.shape[0]):
                neighbors = np.argmin(self.distance+
                                      np.eye(self.distance.shape[0]) * np.max(self.distance, axis=0), axis=0)
                for a, t in enumerate(self.attr_types):
                    if t=='c':
                        begin = self.inputs_block[a][0]
                        end = self.inputs_block[a][1]
                        if np.random.rand() < self.p:
                            temp = self.data[i][begin:end]
                            self.data[i][begin:end] = self.data[neighbors[i]][begin:end]
                            self.data[neighbors[i]][begin:end] = temp
                self._update_distance([i, neighbors[i]])
            all_samples.append(self.data)
            self.data = self.initial_data.copy()
            self.distance = self.initial_distance.copy()
        return np.concatenate(all_samples, axis=0)

"""
