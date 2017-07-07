import numpy as np

class Sampler(object):

    def __init__(self, inputs_block, attr_types, initial_samples=None, distribution_mapping=None):
        self.mapping = {
            "b": "bernoulli",
            "c": "multinoulli",
            "r": "gaussian mixture",
            "i": "poisson mixture",
            "ib": "negative_binomial mixture"
        }
        if distribution_mapping is not None:
            for key in distribution_mapping:
                self.mapping[key] = distribution_mapping[key]
        self.inputs_block = inputs_block
        self.attr_types = attr_types
        self.initial_samples = initial_samples
        if initial_samples is not None:
            self.cur_samples = self.initial_samples.copy()
        else:
            self.cur_samples = None
        self.max_step = len(attr_types)
        self.cur_round = 0
        self.cur_step = 0

    def reset(self, initial_samples=None):
        if initial_samples is not None:
            self.initial_samples = initial_samples
        self.cur_samples = self.initial_samples.copy()
        self.cur_round = 0
        self.cur_step = 0

    def draw_samples_for_one_step(self):
        pass

    def draw_samples_for_one_round(self, max_step=None):
        self.cur_round += 1
        self.cur_step = 0
        if max_step is None:
            max_step = self.max_step
        else:
            max_step = min(max_step, self.max_step)
        for step in range(max_step):
            self.draw_samples_for_one_step()
        return self.cur_samples.copy()

    def run_sampling(self, num_round=1, skip=0, max_step=None):
        all_samples = []
        for r in range(skip):
            self.draw_samples_for_one_round(max_step)
        for r in range(num_round):
            all_samples.append(self.draw_samples_for_one_round(max_step))
        return np.concatenate(all_samples, axis=0)

class BlockGibbsSampler(Sampler):

    def __init__(self, inputs_block, attr_types, query_func, sampling_order=None, initial_samples=None, distribution_mapping=None):
        if sampling_order is None:
            sampling_order = np.random.permutation(len(attr_types))
        self.sampling_order = sampling_order
        self.query_func = query_func
        super(BlockGibbsSampler, self).__init__(inputs_block, attr_types, initial_samples, distribution_mapping)

    def reset(self, initial_samples=None):
        self.sampling_order = np.random.permutation(len(self.attr_types))
        super(BlockGibbsSampler, self).reset(initial_samples)


    def draw_samples_for_one_step(self):
        o = self.sampling_order[self.cur_step]
        cond_prob = self.query_func(self.cur_samples)[o]
        samples = self.cur_samples.copy()
        attr_type = self.attr_types[o]
        block = self.inputs_block[o]
        if self.mapping[attr_type] == 'bernoulli':
            arr = []
            for p in cond_prob:
                s = np.random.multinomial(1, pvals=p)
                arr.append(s)
            samples[:, block[0]:block[1]] = np.array(arr)[:, 1:]
        elif self.mapping[attr_type] == 'multinoulli':
            arr = []
            for p in cond_prob:
                s = np.random.multinomial(1, pvals=p)
                arr.append(s)
            samples[:, block[0]:block[1]] = np.array(arr)
        elif self.mapping[attr_type] == 'gaussian mixture':
            arr = []
            for mu, sigma, alpha in zip(list(cond_prob[0]), list(cond_prob[1]), list(cond_prob[2])):
                idx = np.argmax(np.random.multinomial(1, pvals=alpha))
                s = np.random.normal(loc=mu[idx], scale=sigma[idx])
                arr.append(s)
            samples[:, block[0]:block[1]] = np.array(arr)[:, None]
        elif self.mapping[attr_type] == 'poisson mixture':
            arr = []
            for lam, alpha in zip(list(cond_prob[0]), list(cond_prob[1])):
                idx = np.argmax(np.random.multinomial(1, pvals=alpha))
                s = np.random.poisson(lam=lam[idx])
                arr.append(s)
            samples[:, block[0]:block[1]] = np.array(arr)[:, None]
        elif self.mapping[attr_type] == 'negative_binomial mixture':
            arr = []
            for offset, count, prob, alpha in zip(list(cond_prob[0]), list(cond_prob[1]), list(cond_prob[2]), list(cond_prob[3])):
                idx = np.argmax(np.random.multinomial(1, pvals=alpha))
                s = np.random.negative_binomial(n=count[idx], p=prob[idx])
                #s += offset[idx]
                arr.append(s)
            samples[:, block[0]:block[1]] = np.array(arr)[:, None]
        else:
            raise Exception("model not found")
        self.cur_samples = samples
        self.cur_step += 1
        return self.cur_samples.copy()

class RandomizedSampler(Sampler):

    def __init__(self, inputs_block, attr_types, query_func, initial_samples=None, distribution_mapping=None):
        self.query_func = query_func
        if initial_samples is not None:
            arr = []
            for i in range(initial_samples.shape[0]):
                arr.append(np.random.permutation(initial_samples.shape[1]))
            self.sampling_order = np.array(arr)
        super(BlockGibbsSampler, self).__init__(inputs_block, attr_types, initial_samples, distribution_mapping)

    def draw_samples_for_one_step(self):
        cond_probs = self.query_func(self.cur_samples)
        samples = self.cur_samples.copy()
        for idx, (sample, order) in enumerate(zip(samples, self.sampling_order)):
            o = order[self.cur_step]
            attr_type = self.attr_types[o]
            block = self.inputs_block[o]
            cond_prob = cond_probs[o]
            if self.mapping[attr_type] == 'bernoulli':
                p = cond_prob[idx]
                s = np.random.multinomial(1, pvals=p)
                sample[block[0]:block[1]] = s[1:]
            elif self.mapping[attr_type] == 'multinoulli':
                p = cond_prob[idx]
                s = np.random.multinomial(1, pvals=p)
                sample[block[0]:block[1]] = s
            elif self.mapping[attr_type] == 'gaussian mixture':
                mu, sigma, alpha = cond_prob[0][idx], cond_prob[1][idx], cond_prob[2][idx]
                i = np.argmax(np.random.multinomial(1, pvals=alpha))
                s = np.random.normal(loc=mu[i], scale=sigma[i])
                sample[block[0]:block[1]] = s
            elif self.mapping[attr_type] == 'poisson mixture':
                lam, alpha = cond_prob[0][idx], cond_prob[1][idx]
                i = np.argmax(np.random.multinomial(1, pvals=alpha))
                s = np.random.poisson(lam=lam[i])
                sample[block[0]:block[1]] = s
            elif self.mapping[attr_type] == 'negative_binomial mixture':
                count, prob, alpha = cond_prob[0][idx], cond_prob[1][idx], cond_prob[2][idx]
                i = np.argmax(np.random.multinomial(1, pvals=alpha))
                s = np.random.negative_binomial(n=count[i], p=prob[i])
                sample[block[0]:block[1]] = s
            else:
                raise Exception("model not found")
        self.cur_samples = samples
        self.cur_step += 1
        return self.cur_samples.copy()

    def reset(self, initial_samples=None):
        super(RandomizedSampler, self).reset(initial_samples)
        if self.initial_samples is not None:        
            arr = []
            for i in range(self.initial_samples.shape[0]):
                arr.append(np.random.permutation(self.initial_samples.shape[1]))
            self.sampling_order = np.array(arr)


"""
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




class RandomSampler(Sampler):

    def __init__(self, initial_samples, inputs_block, attr_types,
                                        cond_func):
        assert len(inputs_block)==len(attr_types), \
                        "lengths of inputs_block and attr_types don't agree"
        self.cond_func = cond_func
        if initial_samples is not None:
            arr = []
            for i in range(initial_samples.shape[0]):
                arr.append(np.random.permutation(initial_samples.shape[1]))
            self.sampling_order = np.array(arr)
        super(RandomSampler, self).__init__(initial_samples,
                                                inputs_block, attr_types)

    def reset_initial_samples(self, initial_samples=None):
        if initial_samples is not None:
            self.initial_samples = initial_samples
            arr = []
            for i in range(initial_samples.shape[0]):
                arr.append(np.random.permutation(initial_samples.shape[1]))
            self.sampling_order = np.array(arr)
        self.cur_samples = self.initial_samples.copy()


    def draw_samples_for_one_step(self, last_step_samples, step):
        samples = last_step_samples.copy()
        cond_probs = self.cond_func(samples)
        for i, (o, sample) in enumerate(zip(self.sampling_order, samples)):
            o = o[step]
            attr_type = self.attr_types[o]
            block = self.inputs_block[o]
            if attr_type == 'r':
                mu, sigma, alpha = cond_probs[o][0][i], cond_probs[o][1][i], cond_probs[o][2][i]
                idx = np.argmax(np.random.multinomial(1, pvals=alpha))
                s = np.random.normal(loc=mu[idx], scale=sigma[idx])
                sample[block[0]:block[1]] = s
            elif attr_type == 'i':
                lam, alpha = cond_probs[o][0][i], cond_probs[o][1][i]
                idx = np.argmax(np.random.multinomial(1, pvals=alpha))
                s = np.random.poisson(lam=lam[idx])
                sample[block[0]:block[1]] = s
            else:
                raise Exception("attr_type not found")
        return samples


    def draw_samples_for_one_round(self):
        for step in range(len(self.inputs_block)):
            self.cur_samples = self.draw_samples_for_one_step(self.cur_samples, step)
        return self.cur_samples


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
            arr = []
            for lam, alpha in zip(list(cond_prob[0]), list(cond_prob[1])):
                idx = np.argmax(np.random.multinomial(1, pvals=alpha))
                s = np.random.poisson(lam=lam[idx])
                arr.append(s)
            samples[:, block[0]:block[1]] = np.array(arr)[:, None]
        elif attr_type == 'ib':
            arr = []
            for offset, count, prob, alpha in zip(list(cond_prob[0]), list(cond_prob[1]), list(cond_prob[2]), list(cond_prob[3])):
                idx = np.argmax(np.random.multinomial(1, pvals=alpha))
                s = np.random.negative_binomial(n=count[idx], p=prob[idx])
                #s += offset[idx]
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
