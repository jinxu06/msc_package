import numpy as np
import tensorflow as tf
from samplers import *

class SyntheticDataGenerator(object):

    def __init__(self, initial_inputs, initial_targets):
        self.initial_inputs = initial_inputs.copy()
        self.initial_targets = initial_targets.copy()

    def reset(self):
        pass

    def run_sampling(num_round, skip, max_step):
        pass

    def generate(self, multiple, weight_ratio, num_round=1, skip=0, max_step=None, include_original_data=False, shuffle=False):
        all_gen_data = []
        if include_original_data:
            train_data = np.concatenate([self.initial_inputs, self.initial_targets[:, None]], axis=1)
            all_gen_data.append(train_data)
        for m in range(multiple):
            inputs, targets = self.run_sampling(num_round, skip, max_step)
            gen_data = np.concatenate([inputs, targets[:, None]], axis=1)
            all_gen_data.append(gen_data)
            self.reset()
            print "gen {0}".format(m+1)
        all_data = np.concatenate(all_gen_data, axis=0)
        sample_weight = np.ones((all_data.shape[0], )) / multiple * weight_ratio
        is_train = np.zeros_like(sample_weight)
        if include_original_data:
            sample_weight[:train_data.shape[0]] = 1.
            is_train[:train_data.shape[0]] = 1.
        all_data = np.concatenate([all_data, sample_weight[:, None], is_train[:, None]], axis=1)
        if shuffle:
            np.random.shuffle(all_data)
        return all_data[:, :-3], all_data[:, -3], all_data[:, -2], all_data[:, -1]

class PerClassSyntheticDataGenerator(SyntheticDataGenerator):

    def __init__(self, initial_inputs, initial_targets, samplers, num_classes=None):
        super(PerClassSyntheticDataGenerator, self).__init__(initial_inputs, initial_targets)
        if num_classes is None:
            num_classes = int(max(initial_targets)) + 1
        self.num_classes = num_classes
        self.samplers = samplers
        for c in range(self.num_classes):
            self.samplers[c].reset(initial_inputs[initial_targets==c])

    def reset(self, initial_inputs=None, initial_targets=None):
        if initial_inputs is not None:
            self.initial_inputs = initial_inputs.copy()
            self.initial_targets = initial_targets.copy()
        for c in range(self.num_classes):
            self.samplers[c].reset(self.initial_inputs[self.initial_targets==c])

    def run_sampling(self, num_round=1, skip=0, max_step=None):

        inputs = np.zeros((self.initial_inputs.shape[0]*num_round, self.initial_inputs.shape[1]))
        targets = np.concatenate([self.initial_targets.copy() for i in range(num_round)], axis=0)
        for c in range(self.num_classes):
            inputs[targets==c] = self.samplers[c].run_sampling(num_round, skip, max_step)
        return inputs, targets



    """
    def generate(self, multiple, weight_ratio, num_round=1, skip=0, max_step=None, include_original_data=False, shuffle=False, test_split=0.3):
        all_gen_data = []
        test_mask = np.zeros((self.initial_inputs.shape[0], ))
        test_mask[-int(test_mask.shape[0]*test_split):] = 1
        if include_original_data:
            train_data = np.concatenate([self.initial_inputs, self.initial_targets[:, None]], axis=1)
            all_gen_data.append(train_data)
        for m in range(multiple):
            inputs, targets = self.run_sampling(num_round, skip, max_step)
            gen_data = np.concatenate([inputs, targets[:, None]], axis=1)
            all_gen_data.append(gen_data)
            self.reset()
            print "gen {0}".format(m+1)
        all_data = np.concatenate(all_gen_data, axis=0)
        sample_weight = np.ones((all_data.shape[0], )) / multiple * weight_ratio
        test_mask = np.concatenate([test_mask.copy() for i in range(all_data.shape[0]/self.initial_inputs.shape[0])], axis=0) * 2
        if include_original_data:
            sample_weight[:train_data.shape[0]] = 1.
            ref = test_mask[:train_data.shape[0]]
            ref[ref==2] = 1
        all_data = np.concatenate([all_data, sample_weight[:, None], test_mask[:, None]], axis=1)
        if shuffle:
            np.random.shuffle(all_data)
        return all_data[:, :-3], all_data[:, -3], all_data[:, -2], all_data[:, -1]
    """


class TargetsAsInputsSyntheticDataGenerator(SyntheticDataGenerator):

    def __init__(self, initial_inputs, initial_targets, sampler, num_classes=None, targets_type='c'):
        super(TargetsAsInputsSyntheticDataGenerator, self).__init__(initial_inputs, initial_targets)
        if targets_type=='c' and num_classes is None:
            num_classes = int(max(initial_targets)) + 1
        self.num_classes = num_classes
        self.sampler = sampler
        self.targets_type = targets_type
        if targets_type=='c':
            encoded_targets = np.zeros((self.initial_targets.shape[0], self.num_classes), dtype=np.int32)
            for c in range(self.num_classes):
                encoded_targets[:, c] = self.initial_targets==c
            encoded_targets = encoded_targets.astype(np.int32)
        self.initial_data = np.concatenate([self.initial_inputs, encoded_targets], axis=1)
        self.sampler.reset(self.initial_data)

    def reset(self, initial_inputs=None, initial_targets=None):
        if initial_inputs is not None:
            if self.targets_type=='c':
                encoded_targets = np.zeros((self.initial_targets.shape[0], self.num_classes), dtype=np.int32)
                for c in range(self.num_classes):
                    encoded_targets[:, c] = self.initial_targets==c
                encoded_targets = encoded_targets.astype(np.int32)
            self.initial_data = np.concatenate([self.initial_inputs, encoded_targets], axis=1)
        self.sampler.reset(self.initial_data)

    def run_sampling(self, num_round=1, skip=0, max_step=None):

        data = self.sampler.run_sampling(num_round, skip, max_step)
        if self.targets_type=='c':
            inputs, targets = data[:, :-self.num_classes], data[:, -self.num_classes:]
            targets = np.argmax(targets, axis=1)
        else:
            inputs, targets = data[:, :-1], data[:, -1]
        return inputs, targets



"""
class SyntheticDataGenerator:

    def __init__(self, inputs, targets, models):
        self.num_attr = len(models)
        self.original_inputs = inputs
        self.original_targets = targets
        self.models = models


    def generate(self, num_round, include_original_data=True):
        samplers = []
        all_samples = []
        if include_original_data:
            samples = np.array(np.concatenate([self.original_inputs, self.original_targets[:,None]], axis=1))
            all_samples.append(samples)
        for i in range(self.num_attr):
            model = self.models[i]
            inputs = self.original_inputs[self.original_targets==i, :]
            sampler = BlockGibbsSampler(inputs, model.inputs_block, model.query)
            samplers.append(sampler)
            samples = sampler.run_sampling(num_round, mix_up=True)
            samples = np.concatenate([samples, np.ones((samples.shape[0],1), dtype=np.int32)*i], axis=1)
            all_samples.append(samples)
        all_samples = np.concatenate(all_samples, axis=0)
        np.random.shuffle(all_samples)
        return all_samples[:, :-1], all_samples[:, -1]


class NonBayesSyntheticDataGenerator(SyntheticDataGenerator):

    def __init__(self):
        pass

    def generate(self, num_round, include_original_data=True):
        pass







class MUNGESyntheticDataGenerator:

    def __init__(self, inputs, targets, models):
        for i, model in enumerate(models):
            model.feed_initial_data(inputs[targets==i, :])
        self.models = models
        self.original_inputs = inputs
        self.original_targets = targets


    def generate(self, num_round, include_original_data=True):
        samples = []
        if include_original_data:
            samples.append(np.array(np.concatenate([self.original_inputs, self.original_targets[:,None]], axis=1)))
        for i, model in enumerate(models):
            inputs = model.run(num_round)
            targets = np.ones((inputs.shape[0], 1), dtype=np.int32)*i
            data = np.concatenate([inputs, targets], axis=1)
            samples.append(data)
        samples = np.concatenate(samples, axis=0)
        np.random.shuffle(samples)
        return samples[:, :-1], samples[:, -1]

"""
