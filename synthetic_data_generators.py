import numpy as np
import tensorflow as tf
from samplers import *

class SyntheticDataGenerator(object):

    def __init__(self, initial_inputs, initial_targets):
        self.initial_inputs = initial_inputs
        self.initial_targets = initial_targets

    def generate(self, num_round, include_initial_data=True):
        pass

class PerClassSyntheticDataGenerator(SyntheticDataGenerator):

    def __init__(self, initial_inputs, initial_targets, samplers, num_classes=None):
        super(PerClassSyntheticDataGenerator, self).__init__(initial_inputs, initial_targets)
        if num_classes is None:
            num_classes = max(initial_targets) + 1
        assert len(samplers)==num_classes, "number of samplers mismatch the targets"
        self.num_classes = num_classes

        self.samplers = samplers
        for c in range(self.num_classes):
            self.samplers[c].reset_initial_samples(initial_inputs[initial_targets==c])

    def generate(self, num_round, include_initial_data=True):
        all_samples = []
        if include_initial_data:
            all_samples.append(np.array(np.concatenate([self.initial_inputs, self.initial_targets[:,None]], axis=1)))
        for c in range(self.num_classes):
            samples = self.samplers[c].run_sampling(num_round)
            samples = np.concatenate([samples, np.ones((samples.shape[0],1), dtype=np.int32)*c], axis=1)
            all_samples.append(samples)
        all_samples = np.concatenate(all_samples, axis=0)
        np.random.shuffle(all_samples)
        return all_samples[:, :-1], all_samples[:, -1]

class TargetsAsInputsSyntheticDataGenerator(SyntheticDataGenerator):

    def __init__(self, initial_inputs, initial_targets, sampler, num_classes=None, targets_type='c'):
        super(TargetsAsInputsSyntheticDataGenerator, self).__init__(initial_inputs, initial_targets)
        if targets_type=='c' and num_classes is None:
            num_classes = max(initial_targets) + 1
        self.num_classes = num_classes
        self.sampler = sampler
        self.targets_type = targets_type
        if targets_type=='c':
            encoded_targets = np.zeros((self.initial_targets.shape[0], self.num_classes), dtype=np.int32)
            for c in range(self.num_classes):
                encoded_targets[:, c] = self.initial_targets==c
            encoded_targets = encoded_targets.astype(np.int32)
        self.initial_data = np.concatenate([self.initial_inputs, encoded_targets], axis=1)
        self.sampler.reset_initial_samples(self.initial_data)

    def generate(self, num_round, include_initial_data=True):
        all_samples = []
        if include_initial_data:
            all_samples.append(np.array(np.concatenate([self.initial_inputs, self.initial_targets[:,None]], axis=1)))

        samples = self.sampler.run_sampling(num_round)
        inputs = samples[:, :-self.num_classes]
        targets = samples[:, -self.num_classes:]
        targets = np.argmax(targets, axis=1)[:, None]
        samples = np.concatenate([inputs, targets], axis=1)
        all_samples.append(samples)
        all_samples = np.concatenate(all_samples, axis=0)
        np.random.shuffle(all_samples)
        return all_samples[:, :-1], all_samples[:, -1]


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